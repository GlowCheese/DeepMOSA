####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    var_15 = '.py'
    var_16 = module_0.Config()
    var_17 = 'non_existent_path.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = 'skip_this.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'non_existent_file.py'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = module_0.Config()
    var_15 = 'test_directory'
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_16, var_14, var_17, var_18)
    var_20 = list(var_19)
    var_21 = 'test_directory/skip_file.py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = [var_15]
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_23, var_25, var_26)
    var_28 = list(var_27)
    var_29 = True
    var_30 = module_0.Config()
    var_31 = 'test_directory_with_links'
    var_32 = [var_31]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_30, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test content'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subfile content'
    var_5 = 'skipped_dir'
    var_6 = 'skipped.py'
    var_7 = '# skipped content'
    var_8 = 'readme.txt'
    var_9 = '# not python'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = 'nonexistent/path'
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_16, var_10, var_17, var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_17)
    assert var_22 == 0
    var_23 = len(var_18)
    assert var_23 == 1
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_16, var_10, var_24, var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = len(var_25)
    assert var_30 == 0
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_16, var_10, var_31, var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = len(var_31)
    assert var_36 == 0
    var_37 = len(var_32)
    assert var_37 == 0



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_6, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test1.py'
    var_13 = 'test2.py'
    var_14 = "print('test1')"
    var_15 = "print('test2')"
    var_16 = 'test.txt'
    var_17 = 'not python'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = set(var_11)
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/nonexistent/path'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = 'test.py'
    var_30 = "print('test')"
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_25, var_32, var_33, var_34)
    var_36 = list(var_35)
    var_37 = [var_27]
    var_38 = 'test.py'
    var_39 = "print('test')"
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_31, var_40, var_41, var_42)
    var_44 = list(var_43)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = 'test_directory/test1.py'
    var_11 = [var_10]
    var_12 = module_1.find(var_11, var_0, var_3, var_4)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = len(var_3)
    assert var_15 == 0
    var_16 = len(var_4)
    assert var_16 == 0
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_0, var_3, var_4)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_3)
    assert var_22 == 0
    var_23 = len(var_4)
    assert var_23 == 1
    var_24 = 'test_directory/subdir'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = [var_1]
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_27, var_26, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = len(var_28)
    assert var_33 == 1
    var_34 = len(var_29)
    assert var_34 == 0
    var_35 = [var_10]
    var_36 = module_0.Config()
    var_37 = [var_1]
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_37, var_36, var_38, var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = len(var_38)
    assert var_43 == 1
    var_44 = len(var_39)
    assert var_44 == 0



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = module_0.Config()
    var_11 = 'test_dir'
    var_12 = [var_11]
    var_13 = []
    var_14 = []
    var_15 = module_1.find(var_12, var_10, var_13, var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    var_18 = '.py'
    var_19 = len(var_13)
    assert var_19 == 0
    var_20 = len(var_14)
    assert var_20 == 0
    var_21 = module_0.Config()
    var_22 = 'non_existent_path'
    var_23 = [var_22]
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_23, var_21, var_24, var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = len(var_25)
    assert var_30 == 1
    var_31 = 'skip_me.py'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = [var_11]
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_34, var_33, var_35, var_36)
    var_38 = list(var_37)
    var_39 = module_0.Config()
    var_40 = [var_1, var_11]
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_40, var_39, var_41, var_42)
    var_44 = list(var_43)
    var_45 = len(var_41)
    assert var_45 == 0
    var_46 = len(var_42)
    assert var_46 == 0



# Parsed testcases at query #7
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
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_0, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'nonexistent_path.py'
    var_13 = [var_12]
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = module_1.find(var_13, var_14, var_15, var_16)
    var_18 = list(var_17)
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = "print('file1')"
    var_22 = "print('file2')"
    var_23 = 'file.txt'
    var_24 = 'text file'
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_13, var_25, var_26, var_27)
    var_29 = list(var_28)
    var_30 = set(var_29)
    var_31 = 'subdir'
    var_32 = 'file.py'
    var_33 = "print('file')"
    var_34 = 'skipped_dir'
    var_35 = 'skipped.py'
    var_36 = "print('skipped')"
    var_37 = [var_34]
    var_38 = module_0.Config()
    var_39 = []
    var_40 = []
    var_41 = 'real_dir'
    var_42 = 'symlink_dir'
    var_43 = 'file.py'
    var_44 = "print('file')"
    var_45 = True
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_37, var_46, var_47, var_48)
    var_50 = list(var_49)
    var_51 = False
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test'
    var_6 = 'non_existent.py'
    var_7 = 'test_file.py'
    var_8 = '# test file'
    var_9 = [var_3, var_6, var_7]
    var_10 = module_1.find(var_9, var_0, var_1, var_2)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 'test.py'
    var_14 = len(var_1)
    assert var_14 == 0



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = '/nonexistent/path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_1)
    assert var_16 == 0
    var_17 = len(var_2)
    assert var_17 == 1
    var_18 = b'# test'
    var_19 = [var_15]
    var_20 = module_1.find(var_19, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = len(var_1)
    assert var_23 == 0
    var_24 = len(var_2)
    assert var_24 == 0
    var_25 = 'skip_me'
    var_26 = '# should be skipped'
    var_27 = module_1.find(var_13, var_0, var_1, var_2)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = len(var_1)
    assert var_30 == 1
    var_31 = len(var_2)
    assert var_31 == 0



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b"print('test')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_6, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test.py'
    var_13 = "print('test')"
    var_14 = 'test.txt'
    var_15 = 'not python'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = len(var_11)
    assert var_19 == 1
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = '/nonexistent/path'
    var_24 = [var_23]
    var_25 = module_1.find(var_24, var_20, var_21, var_22)
    var_26 = list(var_25)
    var_27 = 'test.py'
    var_28 = "print('test')"
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_23, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = [var_25]
    var_36 = 'test1.py'
    var_37 = "print('test1')"
    var_38 = 'test2.py'
    var_39 = "print('test2')"
    var_40 = b"print('standalone')"
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_40, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 3



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = True
    var_6 = "print('test')"
    var_7 = 'non_existent_path.py'
    var_8 = 'single_file.py'
    var_9 = "print('single')"
    var_10 = module_1.find(var_4, var_0, var_1, var_2)
    var_11 = list(var_10)
    var_12 = len(var_1)
    assert var_12 == 0
    var_13 = 'test_dir/test.py'



# Parsed testcases at query #13
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# test1'
    var_16 = '# test2'
    var_17 = 'text'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'test_dir/file1.py'
    var_26 = 'test_dir/file2.py'
    var_27 = 'test_dir/non_py.txt'
    var_28 = 'non_existent_path.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_29, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'test_skip_dir'
    var_36 = '# skip'
    var_37 = [var_35]
    var_38 = 'skip_me.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'test_skip_dir/skip_me.py'
    var_46 = 'test_mixed_dir'
    var_47 = '# mixed'
    var_48 = '# single'
    var_49 = 'single_file.py'
    var_50 = [var_46, var_49]
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = module_1.find(var_50, var_51, var_52, var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'test_mixed_dir/mixed.py'



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subfile'
    var_5 = 'skipped'
    var_6 = 'skipped.py'
    var_7 = '# skipped'
    var_8 = 'nonexistent.py'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = len(var_11)
    assert var_13 == 0
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = []
    var_19 = []
    var_20 = len(var_18)
    assert var_20 == 0
    var_21 = len(var_19)
    assert var_21 == 1
    var_22 = []
    var_23 = []
    var_24 = len(var_22)
    assert var_24 == 1
    var_25 = len(var_23)
    assert var_25 == 1



# Parsed testcases at query #15
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
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_0, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = 'file2.py'
    var_14 = "print('file1')"
    var_15 = "print('file2')"
    var_16 = 'file.txt'
    var_17 = 'text file'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_0, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = sorted(var_22)
    var_24 = '/nonexistent/path'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_25, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'subdir'
    var_32 = 'file.py'
    var_33 = "print('in subdir')"
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_23, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = 'subdir'
    var_41 = 'file.py'
    var_42 = "print('in subdir')"
    var_43 = 'symlink'
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = sorted(var_39)
    var_49 = False
    var_50 = module_0.Config()
    var_51 = []
    var_52 = []
    var_53 = sorted(var_39)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = 'nonexistent_file.py'
    var_3 = 'single_file.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = 'test_dir/subdir'
    var_8 = True
    var_9 = '# test'
    var_10 = '# test'
    var_11 = '# test'
    var_12 = module_1.find(var_4, var_0, var_5, var_6)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = len(var_5)
    assert var_15 == 0
    var_16 = len(var_6)
    assert var_16 == 1
    var_17 = 'test_dir/file1.py'
    var_18 = 'test_dir/subdir/file2.py'



# Parsed testcases at query #17
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# test1'
    var_16 = '# test2'
    var_17 = 'text'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'test_dir/file1.py'
    var_26 = 'test_dir/file2.py'
    var_27 = 'test_dir/non_py.txt'
    var_28 = 'non_existent_path.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_29, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'test_skip_dir'
    var_36 = '# skip'
    var_37 = [var_35]
    var_38 = 'skip_me.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'test_skip_dir/skip_me.py'
    var_46 = 'test_mixed_dir'
    var_47 = '# mixed'
    var_48 = '# mixed file'
    var_49 = 'mixed_file.py'
    var_50 = [var_46, var_49]
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = module_1.find(var_50, var_51, var_52, var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'test_mixed_dir/mixed.py'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = 'not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = '/nonexistent/path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_1)
    assert var_16 == 0
    var_17 = len(var_2)
    assert var_17 == 1
    var_18 = b'# test'
    var_19 = module_1.find(var_18, var_0, var_1, var_2)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = len(var_1)
    assert var_22 == 0
    var_23 = len(var_2)
    assert var_23 == 0
    var_24 = 'skipme'
    var_25 = '# should be skipped'
    var_26 = module_1.find(var_22, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_1)
    assert var_29 == 1
    var_30 = len(var_2)
    assert var_30 == 0



# Parsed testcases at query #19
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'non_existent_file.py'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'subdir'
    var_21 = '# test'
    var_22 = '# test'
    var_23 = '# test'
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_14, var_24, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'test1.py'
    var_31 = 'test3.py'
    var_32 = 'skipdir'
    var_33 = '# test'
    var_34 = '# test'
    var_35 = 'skipfile.py'
    var_36 = [var_35, var_34]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_14, var_37, var_38, var_39)
    var_41 = list(var_40)
    var_42 = 'normal.py'
    var_43 = [var_31]
    var_44 = len(var_38)
    assert var_44 == 2
    var_45 = 'broken_link'
    var_46 = 'non_existent_target'
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = module_1.find(var_14, var_48, var_49, var_50)
    var_52 = list(var_51)



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# test file 1'
    var_7 = '# test file 2'
    var_8 = '# skipped file'
    var_9 = 'skip_me.py'
    var_10 = module_1.find(var_2, var_0, var_3, var_4)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_dir/test1.py'
    var_15 = 'test_dir/test2.py'
    var_16 = 'test_dir/skip_me.py'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_0, var_3, var_4)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = '# single file'
    var_23 = 'single_file.py'
    var_24 = [var_23]
    var_25 = module_1.find(var_24, var_0, var_3, var_4)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# test2'
    var_6 = 'not python'
    var_7 = 'test.py'
    var_8 = 'test2.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = 'nonexistent/path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_0, var_11, var_12)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = b'# test'
    var_21 = []
    var_22 = []
    var_23 = [var_17]
    var_24 = module_1.find(var_23, var_0, var_21, var_22)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_21)
    assert var_27 == 0
    var_28 = len(var_22)
    assert var_28 == 0
    var_29 = 'skipme'
    var_30 = '# should be skipped'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_23, var_32, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = len(var_33)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'sub_dir'
    var_6 = 'file3.py'
    var_7 = '# Another Python file'
    var_8 = 'file4.py'
    var_9 = '# Yet another Python file'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = 'file'
    var_14 = '.py'
    var_15 = len(var_11)
    assert var_15 == 0
    var_16 = len(var_12)
    assert var_16 == 0
    var_17 = []
    var_18 = []
    var_19 = 'nonexistent'
    var_20 = len(var_17)
    assert var_20 == 0
    var_21 = len(var_18)
    assert var_21 == 1
    var_22 = []
    var_23 = []
    var_24 = len(var_22)
    assert var_24 == 0
    var_25 = len(var_23)
    assert var_25 == 0
    var_26 = [var_5]
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = len(var_28)
    assert var_30 == 1
    var_31 = len(var_29)
    assert var_31 == 0



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'sub_dir'
    var_2 = 'file1.py'
    var_3 = '# Python file'
    var_4 = 'file2.txt'
    var_5 = '# Text file'
    var_6 = 'file3.py'
    var_7 = '# Python file in subdir'
    var_8 = 'file4.py'
    var_9 = '# Another Python file'
    var_10 = 'skipped_dir'
    var_11 = 'file5.py'
    var_12 = '# Should be skipped'
    var_13 = 'nonexistent.py'
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 1
    var_18 = len(var_16)
    assert var_18 == 1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = "print('test1')"
    var_7 = "print('test2')"
    var_8 = 'not python'
    var_9 = module_1.find(var_2, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_dir/test1.py'
    var_15 = 'test_dir/test2.py'
    var_16 = 'test_dir/non_python.txt'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_0, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 0
    var_25 = len(var_20)
    assert var_25 == 1
    var_26 = 'single_file.py'
    var_27 = [var_26]
    var_28 = []
    var_29 = []
    var_30 = "print('single')"
    var_31 = module_1.find(var_27, var_0, var_28, var_29)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = len(var_28)
    assert var_34 == 0
    var_35 = len(var_29)
    assert var_35 == 0
    var_36 = 'skip_dir'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = [var_30]
    var_40 = []
    var_41 = []
    var_42 = 'test_dir/skip_dir'
    var_43 = "print('skipped')"
    var_44 = module_1.find(var_39, var_38, var_40, var_41)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 0
    var_47 = len(var_40)
    assert var_47 == 1
    var_48 = len(var_41)
    assert var_48 == 0
    var_49 = 'test_dir/skip_dir/skipped.py'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test1.py'
    var_4 = '# test'
    var_5 = 'test2.py'
    var_6 = 'test.txt'
    var_7 = '# not python'
    var_8 = 'subdir'
    var_9 = 'test3.py'
    var_10 = 'test'
    var_11 = len(var_1)
    assert var_11 == 0
    var_12 = len(var_2)
    assert var_12 == 0
    var_13 = []
    var_14 = []
    var_15 = '/nonexistent/path'
    var_16 = [var_15]
    var_17 = module_1.find(var_16, var_0, var_13, var_14)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = len(var_13)
    assert var_20 == 0
    var_21 = len(var_14)
    assert var_21 == 1
    var_22 = []
    var_23 = []
    var_24 = b'# test'
    var_25 = module_1.find(var_24, var_0, var_22, var_23)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_22)
    assert var_28 == 0
    var_29 = len(var_23)
    assert var_29 == 0
    var_30 = []
    var_31 = []
    var_32 = 'test.py'
    var_33 = var_24 / var_32
    var_34 = '# test'
    var_35 = 'skipdir'
    var_36 = var_21 / var_35
    var_37 = 'test2.py'
    var_38 = var_36 / var_37
    var_39 = list(var_7)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = len(var_30)
    assert var_41 == 1
    var_42 = len(var_31)
    assert var_42 == 0



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = 'not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = 'nonexistent/path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_0, var_11, var_12)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = b'# test'
    var_21 = []
    var_22 = []
    var_23 = [var_17]
    var_24 = module_1.find(var_23, var_0, var_21, var_22)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_21)
    assert var_27 == 0
    var_28 = len(var_22)
    assert var_28 == 0
    var_29 = 'skipme'
    var_30 = '# test'
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_17, var_0, var_31, var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = len(var_31)
    assert var_36 == 1
    var_37 = len(var_32)
    assert var_37 == 0



# Parsed testcases at query #4
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
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_0, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = 'file2.py'
    var_14 = "print('hello')"
    var_15 = "print('world')"
    var_16 = 'file.txt'
    var_17 = 'not python'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_0, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = set(var_22)
    var_24 = '/nonexistent/path'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_25, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'subdir'
    var_32 = 'file.py'
    var_33 = "print('hello')"
    var_34 = 'skip.py'
    var_35 = "print('skip')"
    var_36 = [var_34]
    var_37 = '*/subdir/*'
    var_38 = [var_37]
    var_39 = module_0.Config()
    var_40 = []
    var_41 = []
    var_42 = module_1.find(var_25, var_39, var_40, var_41)
    var_43 = list(var_42)
    var_44 = set(var_40)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test'
    var_6 = 'non_existent.py'
    var_7 = 'single_file.py'
    var_8 = '# single file'
    var_9 = [var_3, var_6, var_7]
    var_10 = module_1.find(var_9, var_0, var_1, var_2)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = len(var_1)
    assert var_13 == 0
    var_14 = 'test.py'



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'test_dir'
    var_6 = 'file1.py'
    var_7 = '# test1'
    var_8 = 'file2.py'
    var_9 = '# test2'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = 'non_existent_path.py'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'skipped.py'
    var_21 = '# skipped'
    var_22 = [var_20]
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_14, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 2



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# test2'
    var_6 = 'text'
    var_7 = 'test.py'
    var_8 = 'test2.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = '/nonexistent/path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_1)
    assert var_16 == 0
    var_17 = len(var_2)
    assert var_17 == 1
    var_18 = b'# test'
    var_19 = module_1.find(var_18, var_0, var_1, var_2)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = len(var_1)
    assert var_22 == 0
    var_23 = len(var_2)
    assert var_23 == 0
    var_24 = 'skipme'
    var_25 = '# test'
    var_26 = module_1.find(var_21, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_1)
    assert var_29 == 1
    var_30 = len(var_2)
    assert var_30 == 0



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test content'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subfile content'
    var_5 = 'skipped_dir'
    var_6 = 'skipped.py'
    var_7 = '# skipped content'
    var_8 = 'readme.md'
    var_9 = '# not python'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = 'nonexistent'
    var_16 = []
    var_17 = []
    var_18 = len(var_16)
    assert var_18 == 0
    var_19 = len(var_17)
    assert var_19 == 1
    var_20 = []
    var_21 = []
    var_22 = len(var_20)
    assert var_22 == 0
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = []
    var_25 = []
    var_26 = len(var_24)
    assert var_26 == 1
    var_27 = len(var_25)
    assert var_27 == 0



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = '# test'
    var_6 = module_1.find(var_4, var_0, var_1, var_2)
    var_7 = list(var_6)
    var_8 = 'test_dir'
    var_9 = '# test1'
    var_10 = '# test2'
    var_11 = 'text'
    var_12 = [var_8]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/file2.py'
    var_18 = 'test_dir/non_py.txt'
    var_19 = 'nonexistent_path.py'
    var_20 = [var_19]
    var_21 = module_1.find(var_20, var_0, var_1, var_2)
    var_22 = list(var_21)
    var_23 = 'skip_me.py'
    var_24 = '# should be skipped'
    var_25 = [var_23]
    var_26 = module_1.find(var_25, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = 'mixed_dir'
    var_29 = '# included'
    var_30 = '# single'
    var_31 = 'single_file.py'
    var_32 = [var_28, var_31]
    var_33 = module_1.find(var_32, var_0, var_1, var_2)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'mixed_dir/included.py'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# not python'
    var_6 = '# test in subdir'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = 'test2.txt'
    var_10 = '/nonexistent/path'
    var_11 = [var_10]
    var_12 = module_1.find(var_11, var_0, var_1, var_2)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = b'# test'
    var_16 = module_1.find(var_15, var_0, var_1, var_2)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'skipme'
    var_20 = '# should be skipped'
    var_21 = module_1.find(var_14, var_0, var_1, var_2)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = any(var_7)



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = module_0.Config()
    var_15 = 'non_existent_path'
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_16, var_14, var_17, var_18)
    var_20 = list(var_19)
    var_21 = 'test_dir/skip_file.py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = [var_8]
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_23, var_25, var_26)
    var_28 = list(var_27)
    var_29 = True
    var_30 = module_0.Config()
    var_31 = 'test_dir_with_links'
    var_32 = [var_31]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_30, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subdir file'
    var_5 = 'skipped_dir'
    var_6 = 'skipped.py'
    var_7 = '# skipped file'
    var_8 = 'readme.md'
    var_9 = '# not python'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = []
    var_16 = []
    var_17 = 'nonexistent/path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_10, var_15, var_16)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_15)
    assert var_22 == 0
    var_23 = len(var_16)
    assert var_23 == 1
    var_24 = []
    var_25 = []
    var_26 = len(var_20)
    assert var_26 == 1
    var_27 = len(var_24)
    assert var_27 == 0
    var_28 = len(var_25)
    assert var_28 == 0
    var_29 = []
    var_30 = []
    var_31 = len(var_20)
    assert var_31 == 0
    var_32 = len(var_29)
    assert var_32 == 1
    var_33 = len(var_30)
    assert var_33 == 0
    var_34 = []
    var_35 = []
    var_36 = len(var_20)
    assert var_36 == 0
    var_37 = len(var_34)
    assert var_37 == 0
    var_38 = len(var_35)
    assert var_38 == 0



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = 'not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = 'nonexistent/path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_4, var_11, var_12)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = b'# test'
    var_21 = []
    var_22 = []
    var_23 = [var_17]
    var_24 = module_1.find(var_23, var_4, var_21, var_22)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_21)
    assert var_27 == 0
    var_28 = len(var_22)
    assert var_28 == 0
    var_29 = 'skipme'
    var_30 = '# test'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_23, var_32, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = len(var_33)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = len(var_3)
    assert var_8 == 1
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = 'non_existent_path'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_11, var_0, var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_12)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 1
    var_19 = 'test_file.py'
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_0, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = module_0.Config()
    var_29 = 'test_directory_non_python'
    var_30 = [var_29]
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_30, var_28, var_31, var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = len(var_31)
    assert var_36 == 0
    var_37 = len(var_32)
    assert var_37 == 0



# Parsed testcases at query #15
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# test1'
    var_16 = '# test2'
    var_17 = 'text'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'test_dir/file1.py'
    var_26 = 'test_dir/file2.py'
    var_27 = 'test_dir/non_py.txt'
    var_28 = 'non_existent_path.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_29, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'test_skip_dir'
    var_36 = '# skip'
    var_37 = [var_35]
    var_38 = 'skip_me.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'test_skip_dir/skip_me.py'



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = '# Not a Python file'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = '# Python file in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Python file in skipped dir'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 0
    var_16 = []
    var_17 = []
    var_18 = 'nonexistent_path'
    var_19 = [var_18]
    var_20 = module_1.find(var_19, var_11, var_16, var_17)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_16)
    assert var_23 == 0
    var_24 = len(var_17)
    assert var_24 == 1
    var_25 = []
    var_26 = []
    var_27 = len(var_21)
    assert var_27 == 1
    var_28 = len(var_25)
    assert var_28 == 0
    var_29 = len(var_26)
    assert var_29 == 0
    var_30 = []
    var_31 = []
    var_32 = len(var_21)
    assert var_32 == 3
    var_33 = len(var_30)
    assert var_33 == 1
    var_34 = len(var_31)
    assert var_34 == 0



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# test'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = 'test2.txt'
    var_10 = len(var_1)
    assert var_10 == 0
    var_11 = len(var_2)
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = 'nonexistent_path'
    var_15 = [var_14]
    var_16 = module_1.find(var_15, var_0, var_12, var_13)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 0
    var_20 = len(var_13)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = []
    var_23 = []
    var_24 = module_1.find(var_21, var_0, var_22, var_23)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = 'skipme'
    var_30 = '# test'
    var_31 = '# test'
    var_32 = module_1.find(var_27, var_0, var_22, var_23)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = len(var_22)
    assert var_35 == 1
    var_36 = len(var_23)
    assert var_36 == 0



# Parsed testcases at query #18
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
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_0, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test1.py'
    var_13 = 'test2.py'
    var_14 = "print('test1')"
    var_15 = "print('test2')"
    var_16 = 'test.txt'
    var_17 = 'not python'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_0, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = set(var_22)
    var_24 = '/nonexistent/path'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_25, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'test.py'
    var_32 = "print('test')"
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_29, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = [var_23]
    var_40 = 'subdir'
    var_41 = 'test.py'
    var_42 = "print('test')"
    var_43 = 'linkdir'
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = 'test1.py'
    var_49 = 'test2.py'
    var_50 = "print('test1')"
    var_51 = "print('test2')"
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = module_1.find(var_25, var_52, var_53, var_54)
    var_56 = list(var_55)
    var_57 = set(var_56)



# Parsed testcases at query #19
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'non_existent_file.py'
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_15, var_16, var_17, var_18)
    var_20 = list(var_19)
    var_21 = 'test_dir'
    var_22 = True
    var_23 = '# test1'
    var_24 = '# test2'
    var_25 = '# test3'
    var_26 = [var_21]
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_26, var_27, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'test_dir/file1.py'
    var_34 = 'test_dir/file2.py'
    var_35 = 'test_dir/file3.txt'
    var_36 = '# skipped'
    var_37 = [var_21]
    var_38 = 'skipped_file.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'test_dir/skipped_file.py'
    var_46 = 'broken_link'
    var_47 = [var_46]
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = module_1.find(var_47, var_48, var_49, var_50)
    var_52 = list(var_51)



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'non_existent_path'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_0, var_10, var_11)
    var_13 = list(var_12)
    var_14 = 'test_file.py'
    var_15 = [var_14]
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_15, var_0, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'skip_directory'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = [var_1]
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_23, var_22, var_24, var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'skip_file.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = [var_1]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 1



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = "print('hello')"
    var_3 = 'file2.txt'
    var_4 = 'not python'
    var_5 = 'sub_dir'
    var_6 = 'file3.py'
    var_7 = "print('world')"
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = len(var_9)
    assert var_11 == 0
    var_12 = len(var_10)
    assert var_12 == 0
    var_13 = 'nonexistent'
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 1
    var_18 = []
    var_19 = []
    var_20 = len(var_18)
    assert var_20 == 0
    var_21 = len(var_19)
    assert var_21 == 0
    var_22 = [var_5]
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = len(var_24)
    assert var_26 == 1
    var_27 = len(var_25)
    assert var_27 == 0



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = 'non_existent_path'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_11, var_0, var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_12)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 1
    var_19 = 'test_file.py'
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_0, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = 'skip_directory'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = [var_1]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = len(var_32)
    assert var_37 == 1
    var_38 = len(var_33)
    assert var_38 == 0



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = 'not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = '/nonexistent/path'
    var_15 = [var_14]
    var_16 = module_1.find(var_15, var_11, var_12, var_13)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 0
    var_20 = len(var_13)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_15, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'skipme'
    var_31 = '# should be skipped'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_15, var_33, var_34, var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = len(var_34)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 0



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# test file 1'
    var_7 = '# test file 2'
    var_8 = '# skipped file'
    var_9 = 'skip_me.py'
    var_10 = module_1.find(var_2, var_0, var_3, var_4)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = len(var_3)
    assert var_13 == 1
    var_14 = len(var_4)
    assert var_14 == 0
    var_15 = 'test_directory/test1.py'
    var_16 = 'test_directory/test2.py'
    var_17 = 'test_directory/skip_me.py'
    var_18 = 'non_existent_path'
    var_19 = [var_18]
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_19, var_0, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = len(var_20)
    assert var_25 == 0
    var_26 = len(var_21)
    assert var_26 == 1
    var_27 = 'test_file.py'
    var_28 = [var_27]
    var_29 = []
    var_30 = []
    var_31 = '# test file'
    var_32 = module_1.find(var_28, var_0, var_29, var_30)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = len(var_29)
    assert var_35 == 0
    var_36 = len(var_30)
    assert var_36 == 0
    var_37 = 'test_symlink_dir'
    var_38 = [var_37]
    var_39 = []
    var_40 = []
    var_41 = '# target file'
    var_42 = 'test_symlink_dir/target.py'
    var_43 = 'test_symlink_dir/link.py'
    var_44 = module_1.find(var_38, var_0, var_39, var_40)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = len(var_39)
    assert var_47 == 0
    var_48 = len(var_40)
    assert var_48 == 0



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = 'test_dir'
    var_11 = [var_10]
    var_12 = module_1.find(var_11, var_7, var_8, var_9)
    var_13 = list(var_12)
    var_14 = '.py'
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = 'non_existent_path'
    var_19 = [var_18]
    var_20 = module_1.find(var_19, var_15, var_16, var_17)
    var_21 = list(var_20)
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = 'skipped_file.py'
    var_26 = [var_25]
    var_27 = True
    var_28 = module_1.find(var_26, var_22, var_23, var_24)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = [var_3, var_10, var_18]
    var_34 = module_1.find(var_33, var_30, var_31, var_32)
    var_35 = list(var_34)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_6, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = '/non/existent/path.py'
    var_16 = [var_15]
    var_17 = module_1.find(var_16, var_12, var_13, var_14)
    var_18 = list(var_17)
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = "print('hello')"
    var_22 = "print('world')"
    var_23 = 'file.txt'
    var_24 = 'not python'
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_16, var_25, var_26, var_27)
    var_29 = sorted(var_28)
    var_30 = 'subdir'
    var_31 = 'file.py'
    var_32 = "print('hello')"
    var_33 = 'skipped.py'
    var_34 = "print('skipped')"
    var_35 = [var_33, var_34]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = sorted(var_37)
    var_40 = 'target'
    var_41 = 'link'
    var_42 = 'file.py'
    var_43 = "print('hello')"
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = list(var_39)
    var_49 = False
    var_50 = module_0.Config()
    var_51 = []
    var_52 = []



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b'test'
    var_7 = module_1.find(var_3, var_0, var_1, var_2)
    var_8 = list(var_7)
    var_9 = 'test.py'
    var_10 = 'test'
    var_11 = 'test.txt'
    var_12 = 'test'
    var_13 = 'subdir'
    var_14 = 'subtest.py'
    var_15 = 'test'
    var_16 = module_1.find(var_3, var_0, var_1, var_2)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = '/nonexistent/path'
    var_20 = [var_19]
    var_21 = module_1.find(var_20, var_0, var_1, var_2)
    var_22 = list(var_21)
    var_23 = 'test.py'
    var_24 = 'test'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = module_1.find(var_20, var_26, var_1, var_2)
    var_28 = list(var_27)
    var_29 = 'test.py'
    var_30 = 'test'
    var_31 = b'test'
    var_32 = module_1.find(var_20, var_26, var_1, var_2)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'sub_dir'
    var_2 = 'skipped_dir'
    var_3 = 'file1.py'
    var_4 = '# test'
    var_5 = 'file2.py'
    var_6 = 'file3.py'
    var_7 = 'file4.py'
    var_8 = 'not_python.txt'
    var_9 = 'not python'
    var_10 = 'nonexistent.py'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = b'# test'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = len(var_2)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'nonexistent/path.py'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 1



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test1.py'
    var_1 = '# Python file'
    var_2 = 'test2.py'
    var_3 = '# Another Python file'
    var_4 = 'test.txt'
    var_5 = '# Not a Python file'
    var_6 = 'subdir'
    var_7 = 'test3.py'
    var_8 = '# Python file in subdir'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = '.py'
    var_13 = len(var_10)
    assert var_13 == 0
    var_14 = len(var_11)
    assert var_14 == 0
    var_15 = []
    var_16 = []
    var_17 = '/nonexistent/path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_9, var_15, var_16)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_15)
    assert var_22 == 0
    var_23 = len(var_16)
    assert var_23 == 1
    var_24 = b'# Python file'
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_9, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = len(var_25)
    assert var_30 == 0
    var_31 = len(var_26)
    assert var_31 == 0
    var_32 = 'test1.py'
    var_33 = var_24 / var_32
    var_34 = '# Python file'
    var_35 = 'skipdir'
    var_36 = var_23 / var_35
    var_37 = 'test2.py'
    var_38 = var_36 / var_37
    var_39 = '# Python file in skipdir'
    var_40 = [var_35]
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_5, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 0
    var_48 = var_45[var_47]
    var_49 = len(var_42)
    assert var_49 == 1
    var_50 = var_42[var_47]
    var_51 = len(var_43)
    assert var_51 == 0



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = module_0.Config()
    var_12 = '/nonexistent/path'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_11, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 0
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_13, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'skipme'
    var_31 = '# test'
    var_32 = '# test'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_13, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 1
    var_41 = len(var_36)
    assert var_41 == 0



# Parsed testcases at query #3
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# test1'
    var_16 = '# test2'
    var_17 = 'not python'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'test_dir/file1.py'
    var_26 = 'test_dir/file2.py'
    var_27 = 'test_dir/non_py.txt'
    var_28 = 'non_existent_path.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_29, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'skip_dir'
    var_36 = '# skip'
    var_37 = [var_35]
    var_38 = 'skip_dir/skip_file.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'real_dir'
    var_46 = '# real'
    var_47 = 'symlink_dir'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.Config()
    var_51 = []
    var_52 = []
    var_53 = module_1.find(var_48, var_50, var_51, var_52)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1
    var_56 = 'real_dir/real_file.py'



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# test file 1'
    var_7 = '# test file 2'
    var_8 = 'not a Python file'
    var_9 = module_1.find(var_2, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_dir/test1.py'
    var_15 = 'test_dir/test2.py'
    var_16 = 'test_dir/ignored.txt'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_0, var_3, var_4)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_3)
    assert var_22 == 0
    var_23 = len(var_4)
    assert var_23 == 1
    var_24 = '# single file'
    var_25 = 'single_file.py'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_0, var_3, var_4)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = len(var_3)
    assert var_30 == 0
    var_31 = len(var_4)
    assert var_31 == 0



# Parsed testcases at query #4
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
    var_6 = b'test'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_0, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test.py'
    var_13 = 'test'
    var_14 = 'test.txt'
    var_15 = 'test'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_0, var_16, var_17, var_18)
    var_20 = list(var_19)
    var_21 = '/non/existent/path'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_22, var_23, var_24, var_25)
    var_27 = list(var_26)
    var_28 = 'test.py'
    var_29 = 'test'
    var_30 = 'skip_me'
    var_31 = 'skipped.py'
    var_32 = 'test'
    var_33 = [var_30]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_22, var_34, var_35, var_36)
    var_38 = list(var_37)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_6, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'file1.py'
    var_14 = 'file2.py'
    var_15 = "print('file1')"
    var_16 = "print('file2')"
    var_17 = 'file.txt'
    var_18 = 'text file'
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = len(var_11)
    assert var_22 == 2
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = '/nonexistent/path'
    var_27 = [var_26]
    var_28 = module_1.find(var_27, var_23, var_24, var_25)
    var_29 = list(var_28)
    var_30 = len(var_25)
    assert var_30 == 1
    var_31 = 'subdir'
    var_32 = 'file.py'
    var_33 = "print('file')"
    var_34 = 'skipped.py'
    var_35 = "print('skipped')"
    var_36 = [var_34]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_30, var_37, var_38, var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = len(var_38)
    assert var_43 == 1
    var_44 = 'target'
    var_45 = 'file.py'
    var_46 = "print('file')"
    var_47 = 'link'
    var_48 = True
    var_49 = module_0.Config()
    var_50 = []
    var_51 = []
    var_52 = module_1.find(var_40, var_49, var_50, var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = '# Python file in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Should be skipped'
    var_11 = [var_8]
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = '/non/existent/path'
    var_18 = []
    var_19 = []
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = []
    var_22 = []



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_directory'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    var_15 = '.py'
    var_16 = module_0.Config()
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = 'test_skip.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8, var_17]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = b"print('hello')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_6, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test1.py'
    var_13 = 'test2.py'
    var_14 = 'test.txt'
    var_15 = "print('test1')"
    var_16 = "print('test2')"
    var_17 = 'not python'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = set(var_11)
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/nonexistent/path'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = 'skip_me'
    var_30 = 'keep_me'
    var_31 = 'test.py'
    var_32 = "print('test')"
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = len(var_35)
    assert var_37 == 1
    var_38 = 0
    var_39 = var_35[var_38]
    var_40 = 'real_dir'
    var_41 = 'link_dir'
    var_42 = 'test.py'
    var_43 = "print('test')"
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = module_1.find(var_31, var_45, var_46, var_47)
    var_49 = list(var_48)
    var_50 = False
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = list(var_37)



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = module_0.Config()
    var_15 = 'non_existent_path'
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_16, var_14, var_17, var_18)
    var_20 = list(var_19)
    var_21 = 'test_dir/skip_file.py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = [var_8]
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_23, var_25, var_26)
    var_28 = list(var_27)
    var_29 = module_0.Config()
    var_30 = [var_1, var_8]
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_30, var_29, var_31, var_32)
    var_34 = list(var_33)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = 'non_existent_path.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_0, var_1, var_2)
    var_9 = list(var_8)
    var_10 = b"print('hello')"
    var_11 = module_1.find(var_7, var_0, var_1, var_2)
    var_12 = list(var_11)
    var_13 = 'test.py'
    var_14 = "print('hello')"
    var_15 = 'test.txt'
    var_16 = 'not python'
    var_17 = 'subdir'
    var_18 = 'sub_test.py'
    var_19 = "print('sub')"
    var_20 = module_1.find(var_7, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'skip.py'
    var_24 = "print('skip')"
    var_25 = 'skip_dir'
    var_26 = 'skip_sub.py'
    var_27 = "print('skip sub')"
    var_28 = module_1.find(var_7, var_0, var_1, var_2)
    var_29 = list(var_28)
    var_30 = len(var_1)
    assert var_30 == 2



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = 'nonexistent_file.py'
    var_3 = 'single_file.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = 'test_dir/subdir'
    var_8 = True
    var_9 = '# test'
    var_10 = '# test'
    var_11 = '# test'
    var_12 = module_1.find(var_4, var_0, var_5, var_6)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = len(var_6)
    assert var_15 == 1
    var_16 = len(var_5)
    assert var_16 == 0
    var_17 = 'test_dir/file1.py'
    var_18 = 'test_dir/subdir/file2.py'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = '# test'
    var_5 = 'test.txt'
    var_6 = 'text'
    var_7 = 'subdir'
    var_8 = 'sub.py'
    var_9 = '# sub'
    var_10 = 'skipped'
    var_11 = 'skipped.py'
    var_12 = '# skipped'
    var_13 = len(var_1)
    assert var_13 == 0
    var_14 = 'nonexistent.py'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = module_0.Config()
    var_12 = 'nonexistent/path'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_11, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 0
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_13, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'skipme'
    var_31 = '# test'
    var_32 = '# test'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_13, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 1
    var_41 = len(var_36)
    assert var_41 == 0



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# test'
    var_7 = '# test'
    var_8 = 'test_dir/skipped_dir'
    var_9 = '# test'
    var_10 = 'skipped_dir'
    var_11 = module_1.find(var_2, var_0, var_3, var_4)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = len(var_3)
    assert var_14 == 1
    var_15 = 'test_dir/test1.py'
    var_16 = 'test_dir/test2.py'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_0, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = len(var_20)
    assert var_24 == 1
    var_25 = 'test_file.py'
    var_26 = [var_25]
    var_27 = []
    var_28 = []
    var_29 = '# test'
    var_30 = module_1.find(var_26, var_0, var_27, var_28)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = 'test_dir_non_py'
    var_34 = [var_33]
    var_35 = []
    var_36 = []
    var_37 = '# test'
    var_38 = module_1.find(var_34, var_0, var_35, var_36)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 0
    var_41 = 'test_dir_non_py/test.txt'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = 'non_existent_file.py'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_0, var_1, var_2)
    var_10 = list(var_9)
    var_11 = 'test_module.py'
    var_12 = var_3 / var_11
    var_13 = '# test'
    var_14 = module_1.find(var_8, var_0, var_1, var_2)
    var_15 = list(var_14)
    var_16 = str(var_12)
    var_17 = 'skip_me.py'
    var_18 = var_3 / var_17
    var_19 = '# skip me'
    var_20 = module_1.find(var_8, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = str(var_18)
    var_23 = 'test_module.py'
    var_24 = var_3 / var_23
    var_25 = '# test'
    var_26 = 'test_file.py'
    var_27 = 'non_existent_file.py'
    var_28 = module_1.find(var_8, var_0, var_1, var_2)
    var_29 = list(var_28)
    var_30 = str(var_24)



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'file3.py'
    var_6 = '# Another Python file'
    var_7 = 'sub_dir'
    var_8 = 'file4.py'
    var_9 = '# Python file in subdirectory'
    var_10 = 'skipped_dir'
    var_11 = 'file5.py'
    var_12 = '# Should be skipped'
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 1
    var_17 = len(var_15)
    assert var_17 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path.py'
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

import isort.settings as module_0

def test_case_0():
    var_0 = b'# Test file'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = len(var_2)
    assert var_6 == 0
    var_7 = len(var_3)
    assert var_7 == 0

import isort.settings as module_0

def test_case_0():
    var_0 = 'real_dir'
    var_1 = 'real_file.py'
    var_2 = '# Real file'
    var_3 = 'symlink_dir'
    var_4 = True
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 0
    var_9 = len(var_7)
    assert var_9 == 0



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = 'skipped.py'
    var_4 = '# skipped file'
    var_5 = 'readme.md'
    var_6 = '# readme'
    var_7 = 'broken_path'
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = len(var_16)
    assert var_18 == 0
    var_19 = []
    var_20 = []
    var_21 = len(var_19)
    assert var_21 == 0
    var_22 = len(var_20)
    assert var_22 == 0



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = '/nonexistent/path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_0, var_11, var_12)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = b'# test'
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_0, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = 'skipme'
    var_29 = '# test'
    var_30 = module_1.find(var_26, var_0, var_21, var_22)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = len(var_21)
    assert var_33 == 1
    var_34 = len(var_22)
    assert var_34 == 0



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'sub.py'
    var_4 = '# sub'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'nonexistent'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = 'link'
    var_18 = []
    var_19 = []
    var_20 = list(var_4)



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = 'nonexistent_file.py'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_11, var_0, var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_12)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 1
    var_19 = 'test_directory/skipped_dir'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = [var_1]
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_22, var_21, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 1
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'test_directory/file1.py'
    var_31 = [var_30]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_21, var_32, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = len(var_32)
    assert var_37 == 0
    var_38 = len(var_33)
    assert var_38 == 0



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = 'nonexistent_file.py'
    var_3 = 'single_file.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = True
    var_8 = '# test'
    var_9 = '# test'
    var_10 = '# test'
    var_11 = module_1.find(var_4, var_0, var_5, var_6)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = len(var_5)
    assert var_14 == 0
    var_15 = len(var_6)
    assert var_15 == 1
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/subdir/file2.py'
    var_18 = 'test_dir/subdir'



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test1.py'
    var_4 = '# test'
    var_5 = 'test2.py'
    var_6 = 'test.txt'
    var_7 = '# not python'
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 0
    var_10 = []
    var_11 = []
    var_12 = '/nonexistent/path'
    var_13 = [var_12]
    var_14 = module_1.find(var_13, var_0, var_10, var_11)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_10)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 1
    var_19 = b'# test'
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_19, var_0, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = len(var_20)
    assert var_25 == 0
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = 'test_dir'
    var_28 = var_19 / var_27
    var_29 = 'test.py'
    var_30 = var_28 / var_29
    var_31 = '# test'
    var_32 = len(var_23)
    assert var_32 == 0
    var_33 = len(var_20)
    assert var_33 == 1
    var_34 = len(var_21)
    assert var_34 == 0



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = []
    var_10 = []
    var_11 = '/nonexistent/path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_4, var_9, var_10)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_10)
    assert var_16 == 1
    var_17 = b'# test'
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_4, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'skipme'
    var_24 = '# should be skipped'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_16, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_27)
    assert var_32 == 1



# Parsed testcases at query #15
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'non_existent_file.py'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'test_directory'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_21, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = [var_20]
    var_28 = 'test_directory/skip_file.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_27, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = [var_20]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_35, var_36, var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = "print('hello')"
    var_5 = 'test.txt'
    var_6 = 'not python'
    var_7 = len(var_1)
    assert var_7 == 0
    var_8 = len(var_2)
    assert var_8 == 0
    var_9 = []
    var_10 = []
    var_11 = '/nonexistent/path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_9, var_10)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_9)
    assert var_16 == 0
    var_17 = len(var_10)
    assert var_17 == 1
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_8, var_0, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = len(var_18)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 0
    var_25 = 'subdir'
    var_26 = 'test.py'
    var_27 = "print('hello')"
    var_28 = module_1.find(var_15, var_0, var_18, var_19)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = len(var_18)
    assert var_31 == 1
    var_32 = len(var_19)
    assert var_32 == 0



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'sub_dir'
    var_2 = 'skipped_dir'
    var_3 = 'file1.py'
    var_4 = '# Python file'
    var_5 = 'file2.py'
    var_6 = 'file3.txt'
    var_7 = '# Not Python'
    var_8 = 'file4.py'
    var_9 = '# Skipped Python file'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = len(var_12)
    assert var_14 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'single_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip_me.py'
    var_2 = 'include_me.py'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_3, var_0, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = len(var_4)
    assert var_9 == 1
    var_10 = len(var_5)
    assert var_10 == 0



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    var_15 = '.py'
    var_16 = module_0.Config()
    var_17 = 'non_existent_path.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = 'skip_this.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = module_0.Config()
    var_9 = 'non_existent_path'
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_10, var_8, var_11, var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = module_0.Config()
    var_17 = 'test_file.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'skip_directory'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = [var_1]
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_27, var_26, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = 'non_existent_file.py'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_0, var_1, var_2)
    var_10 = list(var_9)
    var_11 = 'test_directory'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = 'test_directory_with_skipped'
    var_16 = [var_15]
    var_17 = module_1.find(var_16, var_0, var_1, var_2)
    var_18 = list(var_17)
    var_19 = [var_3, var_11]
    var_20 = module_1.find(var_19, var_0, var_1, var_2)
    var_21 = list(var_20)



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = '# Python file in subdir'
    var_8 = 'skipped_file.py'
    var_9 = '# Should be skipped'
    var_10 = [var_8]
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 0
    var_16 = 'nonexistent.py'
    var_17 = len(var_13)
    assert var_17 == 1
    var_18 = len(var_12)
    assert var_18 == 1
    var_19 = len(var_13)
    assert var_19 == 1
    var_20 = 'symlink_dir'
    var_21 = 'symlink'
    var_22 = [var_2]
    var_23 = module_1.find(var_22, var_11, var_12, var_13)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = b"print('hello')"
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = len(var_2)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = 'nonexistent.py'
    var_13 = [var_12]
    var_14 = module_1.find(var_13, var_9, var_10, var_11)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_10)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 1
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = "print('file1')"
    var_22 = "print('file2')"
    var_23 = 'file.txt'
    var_24 = 'not python'
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_7, var_25, var_26, var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = len(var_26)
    assert var_31 == 0
    var_32 = len(var_27)
    assert var_32 == 0
    var_33 = 'subdir'
    var_34 = 'file.py'
    var_35 = "print('file')"
    var_36 = 'skip.py'
    var_37 = "print('skip')"
    var_38 = module_0.Config()
    var_39 = []
    var_40 = []
    var_41 = module_1.find(var_30, var_38, var_39, var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = len(var_39)
    assert var_44 == 1
    var_45 = len(var_40)
    assert var_45 == 0



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# test2'
    var_3 = 'ignore'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test.py'
    var_8 = 'test2.py'
    var_9 = 'skipdir'
    var_10 = '# skip'
    var_11 = '# skip2'
    var_12 = 'skip.py'
    var_13 = [var_12, var_11]
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = list(var_7)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_15)
    assert var_19 == 2
    var_20 = any(var_8)
    var_21 = module_0.Config()
    var_22 = []
    var_23 = []
    var_24 = '/nonexistent/path'
    var_25 = [var_24]
    var_26 = module_1.find(var_25, var_21, var_22, var_23)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_23)
    assert var_29 == 1
    var_30 = b'# single file'
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_30, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = b'# not python'
    var_38 = module_0.Config()
    var_39 = []
    var_40 = []
    var_41 = module_1.find(var_37, var_38, var_39, var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 0



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = '/non/existent/path'
    var_10 = [var_9]
    var_11 = module_1.find(var_10, var_6, var_7, var_8)
    var_12 = list(var_11)
    var_13 = b"print('hello')"
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = 'file1.py'
    var_18 = 'file2.py'
    var_19 = "print('file1')"
    var_20 = "print('file2')"
    var_21 = 'file.txt'
    var_22 = 'text file'
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_10, var_23, var_24, var_25)
    var_27 = list(var_26)
    var_28 = set(var_27)
    var_29 = 'subdir'
    var_30 = 'file.py'
    var_31 = "print('file')"
    var_32 = 'skipped.py'
    var_33 = "print('skipped')"
    var_34 = [var_32]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_28, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = 'file.py'
    var_41 = "print('file')"
    var_42 = b"print('outside')"
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_18, var_43, var_44, var_45)
    var_47 = list(var_46)
    var_48 = set(var_47)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'skipped_file.py'
    var_6 = '# Should be skipped'
    var_7 = 'sub_dir'
    var_8 = 'file3.py'
    var_9 = '# Python file in subdirectory'
    var_10 = 'symlink_dir'
    var_11 = module_0.Config()
    var_12 = 'nonexistent_file.py'
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = len(var_14)
    assert var_16 == 1



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = 'not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = 'test2.txt'
    var_10 = '/nonexistent/path'
    var_11 = [var_10]
    var_12 = module_1.find(var_11, var_0, var_1, var_2)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = b'# test'
    var_16 = [var_14]
    var_17 = module_1.find(var_16, var_0, var_1, var_2)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'skipme'
    var_21 = '# test'
    var_22 = module_1.find(var_12, var_0, var_1, var_2)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = any(var_17)



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test content'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subfile content'
    var_5 = 'skipped_dir'
    var_6 = 'skipped.py'
    var_7 = '# skipped content'
    var_8 = 'nonexistent.py'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = len(var_11)
    assert var_13 == 0
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = []
    var_19 = []
    var_20 = len(var_18)
    assert var_20 == 0
    var_21 = len(var_19)
    assert var_21 == 1
    var_22 = []
    var_23 = []
    var_24 = len(var_22)
    assert var_24 == 1
    var_25 = len(var_23)
    assert var_25 == 1



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# test file'
    var_7 = '# test file 2'
    var_8 = 'not a python file'
    var_9 = module_1.find(var_2, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_directory/test1.py'
    var_15 = 'test_directory/test2.py'
    var_16 = 'test_directory/non_python.txt'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_0, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 0
    var_25 = len(var_20)
    assert var_25 == 1
    var_26 = 'test_file.py'
    var_27 = [var_26]
    var_28 = []
    var_29 = []
    var_30 = '# test file'
    var_31 = module_1.find(var_27, var_0, var_28, var_29)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = len(var_28)
    assert var_34 == 0
    var_35 = len(var_29)
    assert var_35 == 0
    var_36 = 'skip_me'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = [var_30]
    var_40 = []
    var_41 = []
    var_42 = 'test_directory/skip_me'
    var_43 = '# test file'
    var_44 = '# test file'
    var_45 = module_1.find(var_39, var_38, var_40, var_41)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = len(var_40)
    assert var_48 == 1
    var_49 = len(var_41)
    assert var_49 == 0
    var_50 = 'test_directory/skip_me/test.py'
    var_51 = 'test_directory/test.py'



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '# Python file 1'
    var_2 = 'file2.py'
    var_3 = '# Python file 2'
    var_4 = 'subdir'
    var_5 = 'file3.py'
    var_6 = '# Python file 3'
    var_7 = 'non_python.txt'
    var_8 = '# Not a Python file'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = '.py'
    var_13 = len(var_10)
    assert var_13 == 0
    var_14 = len(var_11)
    assert var_14 == 0
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = '/non/existent/path'
    var_19 = [var_18]
    var_20 = module_1.find(var_19, var_15, var_16, var_17)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_16)
    assert var_23 == 0
    var_24 = len(var_17)
    assert var_24 == 1
    var_25 = 'single_file.py'
    var_26 = var_18 / var_25
    var_27 = '# Single Python file'
    var_28 = module_0.Config()
    var_29 = []
    var_30 = []
    var_31 = str(var_26)
    var_32 = [var_31]
    var_33 = module_1.find(var_32, var_28, var_29, var_30)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = str(var_26)
    var_37 = len(var_29)
    assert var_37 == 0
    var_38 = len(var_30)
    assert var_38 == 0
    var_39 = 'file1.py'
    var_40 = var_18 / var_39
    var_41 = '# Python file 1'
    var_42 = 'skipped_dir'
    var_43 = var_32 / var_42
    var_44 = var_37 / var_42
    var_45 = 'file2.py'
    var_46 = var_44 / var_45
    var_47 = '# Should be skipped'
    var_48 = 'normal_dir'
    var_49 = 'file3.py'
    var_50 = '# Should be included'
    var_51 = [var_42]
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = len(var_34)
    assert var_55 == 2
    var_56 = len(var_53)
    assert var_56 == 1
    var_57 = len(var_54)
    assert var_57 == 0



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    var_15 = '.py'
    var_16 = module_0.Config()
    var_17 = 'non_existent_path.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = 'test_skip.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = 'text'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = module_0.Config()
    var_12 = 'nonexistent_path'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_11, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 0
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_13, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'skipdir'
    var_31 = '# test'
    var_32 = '# test'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_13, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 1
    var_41 = len(var_36)
    assert var_41 == 0



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = '# test'
    var_5 = 'subdir'
    var_6 = 'sub.py'
    var_7 = '# sub'
    var_8 = 'skipped'
    var_9 = 'skipped.py'
    var_10 = '# skipped'
    var_11 = 'nonexistent.py'
    var_12 = '.py'



# Parsed testcases at query #24
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'non_existent_file.py'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'subdir'
    var_21 = '# test'
    var_22 = 'not python'
    var_23 = '# test in subdir'
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_14, var_24, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'test1.py'
    var_31 = 'test3.py'
    var_32 = 'skip_me'
    var_33 = '# test'
    var_34 = '# should be skipped'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_14, var_36, var_37, var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = len(var_37)
    assert var_42 == 1
    var_43 = '# valid'
    var_44 = 'broken_path.py'
    var_45 = 'valid.py'
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_14, var_46, var_47, var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = any(var_41)



# Parsed testcases at query #25
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'subdir'
    var_14 = '# test'
    var_15 = '# not python'
    var_16 = '# test'
    var_17 = module_0.Config()
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_7, var_17, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'test1.py'
    var_24 = 'test3.py'
    var_25 = 'nonexistent_path.py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_26, var_27, var_28, var_29)
    var_31 = list(var_30)
    var_32 = 'skipdir'
    var_33 = '# test'
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_26, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = len(var_36)
    assert var_40 == 1
    var_41 = 'dir1'
    var_42 = 'dir2'
    var_43 = '# test'
    var_44 = 'symlink'
    var_45 = True
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_26, var_46, var_47, var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = '# test'
    var_6 = module_1.find(var_4, var_0, var_1, var_2)
    var_7 = list(var_6)
    var_8 = 'test_dir'
    var_9 = '# test1'
    var_10 = '# test2'
    var_11 = 'text'
    var_12 = [var_8]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = set(var_14)
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/file2.py'
    var_18 = 'test_dir/non_python.txt'
    var_19 = 'non_existent_path.py'
    var_20 = [var_19]
    var_21 = module_1.find(var_20, var_0, var_1, var_2)
    var_22 = list(var_21)
    var_23 = 'skip_me.py'
    var_24 = [var_23]
    var_25 = '# skip'
    var_26 = module_1.find(var_24, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = 'test_mixed'
    var_29 = '# mixed'
    var_30 = '# single'
    var_31 = 'single.py'
    var_32 = [var_28, var_31]
    var_33 = module_1.find(var_32, var_0, var_1, var_2)
    var_34 = list(var_33)
    var_35 = set(var_34)
    var_36 = 'test_mixed/mixed.py'



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '.'
    var_2 = 'test_dir'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = '# test'
    var_8 = '# test'
    var_9 = '# skipped'
    var_10 = 'skipped'
    var_11 = '.py'
    var_12 = module_1.find(var_3, var_0, var_4, var_5)
    var_13 = list(var_12)
    var_14 = len(var_5)
    assert var_14 == 0
    var_15 = 'test_file.py'
    var_16 = 'test_dir/test_file.py'
    var_17 = 'test_dir/skipped_file.py'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = 'subdir'
    var_11 = '# test'
    var_12 = '# not python'
    var_13 = '# test in subdir'
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = module_1.find(var_2, var_14, var_15, var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'test1.py'
    var_21 = 'test3.py'
    var_22 = len(var_15)
    assert var_22 == 0
    var_23 = len(var_16)
    assert var_23 == 0
    var_24 = module_0.Config()
    var_25 = 'nonexistent_path.py'
    var_26 = [var_25]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_24, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_27)
    assert var_32 == 0
    var_33 = len(var_28)
    assert var_33 == 1
    var_34 = 'skip_me.py'
    var_35 = '# should be skipped'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_26, var_37, var_38, var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 0
    var_43 = len(var_38)
    assert var_43 == 1
    var_44 = len(var_39)
    assert var_44 == 0
    var_45 = '# test'
    var_46 = '# test'
    var_47 = module_0.Config()
    var_48 = 'file1.py'
    var_49 = []
    var_50 = []
    var_51 = module_1.find(var_26, var_47, var_49, var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = any(var_43)
    var_55 = 'file2.py'
    var_56 = any(var_31)
    var_57 = len(var_49)
    assert var_57 == 0
    var_58 = len(var_50)
    assert var_58 == 0



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = 'non_existent_path.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_0, var_1, var_2)
    var_9 = list(var_8)
    var_10 = '# test'
    var_11 = 'test_file.py'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = 'test_dir'
    var_16 = '# test'
    var_17 = [var_15]
    var_18 = module_1.find(var_17, var_0, var_1, var_2)
    var_19 = list(var_18)
    var_20 = 'test_dir/test_file.py'
    var_21 = 'skip_me.py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = '# test'
    var_25 = [var_21]
    var_26 = module_1.find(var_25, var_23, var_1, var_2)
    var_27 = list(var_26)
    var_28 = 'test_dir2'
    var_29 = '# test'
    var_30 = '# test'
    var_31 = 'test_file2.py'
    var_32 = 'non_existent.py'
    var_33 = [var_28, var_31, var_32]
    var_34 = module_1.find(var_33, var_23, var_1, var_2)
    var_35 = list(var_34)
    var_36 = 'test_dir2/test_file.py'



# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# Test file'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# Test file 1'
    var_16 = '# Test file 2'
    var_17 = '# Not a Python file'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'test_dir/file1.py'
    var_26 = 'test_dir/file2.py'
    var_27 = 'test_dir/non_python.txt'
    var_28 = 'non_existent_path.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_29, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'test_skip_dir'
    var_36 = '# Should be skipped'
    var_37 = [var_35]
    var_38 = 'skip_me.py'
    var_39 = [var_38]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_37, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'test_skip_dir/skip_me.py'
    var_46 = 'test_mixed_dir'
    var_47 = '# Mixed file'
    var_48 = '# Single file'
    var_49 = 'single_file.py'
    var_50 = [var_46, var_49]
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = module_1.find(var_50, var_51, var_52, var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'test_mixed_dir/mixed_file.py'



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = 'not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = 'test2.txt'
    var_10 = '/nonexistent/path'
    var_11 = [var_10]
    var_12 = module_1.find(var_11, var_0, var_1, var_2)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = len(var_2)
    assert var_15 == 1
    var_16 = b'# test'
    var_17 = [var_14]
    var_18 = module_1.find(var_17, var_0, var_1, var_2)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'skipme'
    var_22 = '# test'
    var_23 = module_1.find(var_12, var_0, var_1, var_2)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = len(var_1)
    assert var_26 == 1



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_7, var_10, var_11)
    var_13 = list(var_12)
    var_14 = 'test_skip.py'
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = [var_14]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_16, var_18, var_19)
    var_21 = list(var_20)
    var_22 = module_0.Config()
    var_23 = 'non_existent.py'
    var_24 = [var_23]
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_22, var_25, var_26)
    var_28 = list(var_27)
    var_29 = 'test_skip_dir'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = [var_1, var_8, var_29, var_23]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)



# Parsed testcases at query #27
#--------------------------


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
    var_8 = '# test'
    var_9 = '# test'
    var_10 = 'not python'
    var_11 = module_1.find(var_2, var_0, var_3, var_4)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = len(var_3)
    assert var_14 == 0
    var_15 = len(var_4)
    assert var_15 == 0
    var_16 = 'skip_me.py'
    var_17 = []
    var_18 = module_1.find(var_2, var_0, var_17, var_4)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = len(var_17)
    assert var_21 == 1
    var_22 = 0
    var_23 = var_17[var_22]
    var_24 = []
    var_25 = 'nonexistent_path'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_0, var_17, var_24)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = len(var_24)
    assert var_30 == 1
    var_31 = 'test_dir/file1.py'
    var_32 = [var_31]
    var_33 = module_1.find(var_32, var_0, var_17, var_24)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = var_34[var_22]
    var_37 = 'file1.py'



# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subdir file'
    var_5 = 'skipped_dir'
    var_6 = 'skipped.py'
    var_7 = '# skipped file'
    var_8 = 'readme.txt'
    var_9 = '# not python'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 1
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = 'nonexistent'
    var_16 = []
    var_17 = []
    var_18 = len(var_16)
    assert var_18 == 0
    var_19 = len(var_17)
    assert var_19 == 1
    var_20 = []
    var_21 = []
    var_22 = len(var_20)
    assert var_22 == 0
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = []
    var_25 = []
    var_26 = len(var_24)
    assert var_26 == 0
    var_27 = len(var_25)
    assert var_27 == 0



# Parsed testcases at query #31
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = 'not python'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = '/nonexistent/path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_4, var_11, var_12)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_11)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = b'# test'
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_4, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = 'skipme'
    var_29 = '# test'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_26, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = len(var_32)
    assert var_37 == 1
    var_38 = len(var_33)
    assert var_38 == 0



# Parsed testcases at query #32
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'test_dir'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'non_existent_path.py'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_21, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = 'skipped_file.py'
    var_28 = [var_27]
    var_29 = [var_27]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_28, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = [var_6, var_13, var_20]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_35, var_36, var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_3, var_0, var_1, var_2)
    var_5 = list(var_4)
    var_6 = 'nonexistent_path.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_0, var_1, var_2)
    var_9 = list(var_8)
    var_10 = '# test'
    var_11 = 'test_file.py'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = 'test_dir'
    var_16 = '# test1'
    var_17 = '# test2'
    var_18 = 'ignore'
    var_19 = [var_15]
    var_20 = module_1.find(var_19, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'test_dir/test1.py'
    var_24 = 'test_dir/test2.py'
    var_25 = 'test_dir/ignore.txt'
    var_26 = 'skip_me.py'
    var_27 = '# skip'
    var_28 = [var_26]
    var_29 = module_1.find(var_28, var_0, var_1, var_2)
    var_30 = list(var_29)
    var_31 = 'link_dir'
    var_32 = '# target'
    var_33 = 'symlink_dir'
    var_34 = [var_33]
    var_35 = module_1.find(var_34, var_0, var_1, var_2)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = 'symlink_dir/target.py'



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = 'nonexistent_file.py'
    var_3 = 'single_file.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = True
    var_8 = '# test'
    var_9 = '# test'
    var_10 = '# test'
    var_11 = module_1.find(var_4, var_0, var_5, var_6)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = len(var_5)
    assert var_14 == 0
    var_15 = len(var_6)
    assert var_15 == 1
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/subdir/file2.py'
    var_18 = 'test_dir/subdir'



# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = '# Python file 1'
    var_7 = '# Python file 2'
    var_8 = '# Not a Python file'
    var_9 = module_1.find(var_2, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_directory/file1.py'
    var_15 = 'test_directory/file2.py'
    var_16 = 'test_directory/file3.txt'
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_0, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 0
    var_25 = len(var_20)
    assert var_25 == 1
    var_26 = 'test_file.py'
    var_27 = [var_26]
    var_28 = []
    var_29 = []
    var_30 = '# Python file'
    var_31 = module_1.find(var_27, var_0, var_28, var_29)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = len(var_28)
    assert var_34 == 0
    var_35 = len(var_29)
    assert var_35 == 0
    var_36 = 'test_skipped_directory'
    var_37 = [var_36]
    var_38 = []
    var_39 = []
    var_40 = '# Python file'
    var_41 = [var_36]
    var_42 = module_0.Config()
    var_43 = module_1.find(var_37, var_42, var_38, var_39)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 0
    var_46 = len(var_38)
    assert var_46 == 1
    var_47 = len(var_39)
    assert var_47 == 0
    var_48 = 'test_skipped_directory/file.py'



# Parsed testcases at query #31
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = True
    var_16 = '# test1'
    var_17 = '# test2'
    var_18 = 'text'
    var_19 = [var_14]
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_19, var_20, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'test_dir/file1.py'
    var_27 = 'test_dir/file2.py'
    var_28 = 'test_dir/non_python.txt'
    var_29 = 'non_existent_path.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_30, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = 'test_skip_dir'
    var_37 = '# skip'
    var_38 = [var_36]
    var_39 = 'skip_me.py'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_38, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = 'test_skip_dir/skip_me.py'
    var_47 = 'test_link_dir'
    var_48 = '# target'
    var_49 = 'test_link'
    var_50 = [var_49]
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = module_1.find(var_50, var_51, var_52, var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = 'test_link_dir/target.py'



# Parsed testcases at query #36
#--------------------------




# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = '# test'
    var_5 = '# not python'
    var_6 = '# test'
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = 'nonexistent_path'
    var_12 = [var_11]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = len(var_1)
    assert var_16 == 0
    var_17 = len(var_2)
    assert var_17 == 1
    var_18 = b'# test'
    var_19 = module_1.find(var_18, var_0, var_1, var_2)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = len(var_1)
    assert var_22 == 0
    var_23 = len(var_2)
    assert var_23 == 0
    var_24 = '# test'
    var_25 = 'skip_me.py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = module_1.find(var_21, var_27, var_1, var_2)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = len(var_1)
    assert var_31 == 1
    var_32 = len(var_2)
    assert var_32 == 0



# Parsed testcases at query #37
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not a Python file'
    var_5 = 'file3.py'
    var_6 = '# Another Python file'
    var_7 = 'sub_dir'
    var_8 = 'file4.py'
    var_9 = '# Python file in subdirectory'
    var_10 = 'skipped_dir'
    var_11 = 'file5.py'
    var_12 = '# Python file in skipped directory'
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = [var_10]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = []
    var_23 = 'non_existent'
    var_24 = len(var_22)
    assert var_24 == 1



# Parsed testcases at query #33
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = '# test'
    var_12 = module_1.find(var_7, var_8, var_9, var_10)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = True
    var_16 = '# test1'
    var_17 = '# test2'
    var_18 = 'text'
    var_19 = [var_14]
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_19, var_20, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'test_dir/file1.py'
    var_27 = 'test_dir/file2.py'
    var_28 = 'test_dir/non_py.txt'
    var_29 = 'non_existent_path.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_30, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = 'test_skip_dir'
    var_37 = '# skip'
    var_38 = [var_36]
    var_39 = 'skip_me.py'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_38, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = 'test_skip_dir/skip_me.py'
    var_47 = 'test_link_dir'
    var_48 = True
    var_49 = '# target'
    var_50 = 'test_link_dir/target.py'
    var_51 = 'test_link_dir/link.py'
    var_52 = [var_49]
    var_53 = module_0.Config()
    var_54 = []
    var_55 = []
    var_56 = module_1.find(var_52, var_53, var_54, var_55)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 2



# Parsed testcases at query #38
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# Test file 1'
    var_2 = 'Not a Python file'
    var_3 = '# Test file 3'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = len(var_5)
    assert var_9 == 0
    var_10 = len(var_6)
    assert var_10 == 0
    var_11 = module_0.Config()
    var_12 = '/nonexistent/path'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_11, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 0
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = b'# Single file test'
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_13, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = 'skipme'
    var_31 = '# Should be skipped'
    var_32 = '# Should be included'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_13, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 1
    var_41 = len(var_36)
    assert var_41 == 0



# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #39
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test'
    var_6 = '# test'
    var_7 = '# ignore'
    var_8 = [var_3]
    var_9 = module_1.find(var_8, var_0, var_1, var_2)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'test_directory'
    var_13 = '.py'
    var_14 = 'test_directory/test1.py'
    var_15 = []
    var_16 = [var_3]
    var_17 = module_1.find(var_16, var_0, var_15, var_2)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = []
    var_22 = 'nonexistent_path'
    var_23 = [var_22]
    var_24 = module_1.find(var_23, var_0, var_15, var_21)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 0
    var_27 = len(var_21)
    assert var_27 == 1
    var_28 = 'test_directory/test2.py'
    var_29 = [var_28]
    var_30 = module_1.find(var_29, var_0, var_15, var_21)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #35
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'test_dir'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'non_existent_path'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_21, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = 'skipped_dir'
    var_28 = [var_27]
    var_29 = [var_27]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_28, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = 'skipped_file.py'
    var_36 = [var_35]
    var_37 = [var_35]
    var_38 = module_0.Config()
    var_39 = []
    var_40 = []
    var_41 = module_1.find(var_36, var_38, var_39, var_40)
    var_42 = list(var_41)
    var_43 = [var_6, var_13, var_20, var_27]
    var_44 = [var_27]
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = module_1.find(var_43, var_45, var_46, var_47)
    var_49 = list(var_48)



# Parsed testcases at query #40
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
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'non_existent_file.py'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = []
    var_17 = []
    var_18 = module_1.find(var_14, var_15, var_16, var_17)
    var_19 = list(var_18)
    var_20 = 'test_directory'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_21, var_22, var_23, var_24)
    var_26 = list(var_25)
    var_27 = '.py'
    var_28 = [var_20]
    var_29 = 'test_directory/skipped_file.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_28, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = [var_6, var_20]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_36, var_37, var_38, var_39)
    var_41 = list(var_40)
    var_42 = 'symlink_directory'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = module_1.find(var_43, var_45, var_46, var_47)
    var_49 = list(var_48)



