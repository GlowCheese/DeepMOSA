####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_7 = 'test_directory'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_0, var_1, var_2)
    var_10 = list(var_9)
    var_11 = len(var_10)
    var_12 = '.py'
    var_13 = 'non_existent_path'
    var_14 = [var_13]
    var_15 = module_1.find(var_14, var_0, var_1, var_2)
    var_16 = list(var_15)
    var_17 = 'skipped_file.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_0, var_1, var_2)
    var_20 = list(var_19)
    var_21 = [var_3, var_7, var_13]
    var_22 = module_1.find(var_21, var_0, var_1, var_2)
    var_23 = list(var_22)



# Parsed testcases at query #2
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
    var_8 = '.py'
    var_9 = 'non_existent_path'
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_10, var_0, var_11, var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'test_file.py'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_0, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'skip_this_dir'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_1]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = 'skip_this_file.py'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = [var_1]
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_34, var_33, var_35, var_36)
    var_38 = list(var_37)



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



# Parsed testcases at query #4
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
    var_11 = '/nonexistent'
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
    var_23 = 'skipme'
    var_24 = '# test'
    var_25 = module_1.find(var_15, var_0, var_1, var_2)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = len(var_1)
    assert var_28 == 1



# Parsed testcases at query #5
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
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'test_dir'
    var_10 = '# test1'
    var_11 = '# test2'
    var_12 = [var_9]
    var_13 = module_1.find(var_12, var_0, var_1, var_2)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/file2.py'
    var_18 = 'non_existent_path.py'
    var_19 = [var_18]
    var_20 = module_1.find(var_19, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_2)
    assert var_23 == 1
    var_24 = 'skip_me.py'
    var_25 = [var_24]
    var_26 = '# skip me'
    var_27 = module_1.find(var_25, var_0, var_1, var_2)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = len(var_1)
    assert var_30 == 1



# Parsed testcases at query #6
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
    var_6 = '/non/existent/path'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_7, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'test_file.py'
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
    var_28 = 'test_directory/skip_me'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_27, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = [var_13, var_6, var_20]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_35, var_36, var_37, var_38)
    var_40 = list(var_39)



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'sub_test.py'
    var_4 = '# sub test'
    var_5 = 'readme.txt'
    var_6 = 'readme'
    var_7 = 'skipped'
    var_8 = 'skipped.py'
    var_9 = '# skipped'
    var_10 = [var_7]
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 0
    var_16 = 'non_existent'
    var_17 = module_0.Config()
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_2, var_17, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_18)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 1
    var_25 = 'single_test.py'
    var_26 = '# single test'
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_2, var_27, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = len(var_28)
    assert var_33 == 0
    var_34 = len(var_29)
    assert var_34 == 0



# Parsed testcases at query #8
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
    var_36 = True
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = module_1.find(var_35, var_37, var_38, var_39)
    var_41 = list(var_40)
    var_42 = [var_6, var_20]
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_42, var_43, var_44, var_45)
    var_47 = list(var_46)



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = 'file1.py'
    var_5 = '# test'
    var_6 = 'file2.txt'
    var_7 = 'not python'
    var_8 = 'skipped_file.py'
    var_9 = '# skipped'
    var_10 = 'subdir'
    var_11 = 'file3.py'
    var_12 = '# subdir test'
    var_13 = len(var_1)
    assert var_13 == 1
    var_14 = len(var_2)
    assert var_14 == 0
    var_15 = len(var_1)
    assert var_15 == 0
    var_16 = len(var_2)
    assert var_16 == 0
    var_17 = 'nonexistent.py'
    var_18 = len(var_1)
    assert var_18 == 0
    var_19 = len(var_2)
    assert var_19 == 1



# Parsed testcases at query #10
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
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = module_0.Config()
    var_24 = [var_8]
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_24, var_23, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    var_30 = module_0.Config()
    var_31 = [var_1, var_8]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)



# Parsed testcases at query #11
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
    var_8 = '.py'
    var_9 = 'non_existent_path'
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_10, var_0, var_11, var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'test_file.py'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_0, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'skip_directory'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = module_0.Config()
    var_33 = 'non_python_directory'
    var_34 = [var_33]
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_34, var_32, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 0



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '# Python file 1'
    var_2 = 'file2.py'
    var_3 = '# Python file 2'
    var_4 = 'file3.txt'
    var_5 = '# Not a Python file'
    var_6 = 'subdir'
    var_7 = 'file4.py'
    var_8 = '# Python file in subdir'
    var_9 = 'skipped_dir'
    var_10 = 'file5.py'
    var_11 = '# Python file in skipped dir'
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = module_0.Config()
    var_18 = '/non/existent/path'
    var_19 = [var_18]
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_19, var_17, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = len(var_20)
    assert var_25 == 0
    var_26 = len(var_21)
    assert var_26 == 1
    var_27 = 'single_file.py'
    var_28 = var_18 / var_27
    var_29 = '# Single Python file'
    var_30 = module_0.Config()
    var_31 = str(var_28)
    var_32 = [var_31]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_30, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = str(var_28)
    var_39 = len(var_33)
    assert var_39 == 0
    var_40 = len(var_34)
    assert var_40 == 0
    var_41 = 'skipped_file.py'
    var_42 = var_18 / var_41
    var_43 = '# Skipped Python file'
    var_44 = module_0.Config()
    var_45 = str(var_42)
    var_46 = str(var_42)
    var_47 = [var_46]
    var_48 = []
    var_49 = []
    var_50 = module_1.find(var_47, var_44, var_48, var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 0
    var_53 = len(var_48)
    assert var_53 == 1
    var_54 = str(var_42)
    var_55 = len(var_49)
    assert var_55 == 0



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



# Parsed testcases at query #14
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
    var_18 = 'test_dir/non_py.txt'
    var_19 = 'nonexistent/path'
    var_20 = [var_19]
    var_21 = module_1.find(var_20, var_0, var_1, var_2)
    var_22 = list(var_21)
    var_23 = 'skip_me.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = '# skip'
    var_27 = [var_23]
    var_28 = module_1.find(var_27, var_25, var_1, var_2)
    var_29 = list(var_28)
    var_30 = 'test_dir2'
    var_31 = '# test'
    var_32 = '# normal'
    var_33 = 'normal_file.py'
    var_34 = 'nonexistent'
    var_35 = [var_30, var_33, var_34]
    var_36 = module_1.find(var_35, var_25, var_1, var_2)
    var_37 = list(var_36)
    var_38 = 'test_dir2/file.py'



# Parsed testcases at query #15
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
    var_7 = 'non_existent_file.py'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = 'test.py'
    var_12 = '# test'
    var_13 = 'subdir'
    var_14 = 'sub_test.py'
    var_15 = '# sub test'
    var_16 = module_1.find(var_8, var_0, var_3, var_4)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'test.py'
    var_20 = '# test'
    var_21 = 'skipped_dir'
    var_22 = 'sub_test.py'
    var_23 = '# sub test'
    var_24 = module_1.find(var_8, var_0, var_3, var_4)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_3)
    assert var_27 == 1
    var_28 = 0
    var_29 = var_3[var_28]
    var_30 = 'broken_link'
    var_31 = 'non_existent_target'
    var_32 = module_1.find(var_8, var_0, var_3, var_4)
    var_33 = list(var_32)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = var_1 / var_5
    var_7 = '# Python file 1'
    var_8 = 'file2.py'
    var_9 = '# Python file 2'
    var_10 = 'file3.txt'
    var_11 = '# Not a Python file'
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = []
    var_15 = []
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = module_1.find(var_17, var_0, var_14, var_15)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = len(var_14)
    assert var_21 == 0
    var_22 = len(var_15)
    assert var_22 == 1
    var_23 = 'test_file.py'
    var_24 = var_1 / var_23
    var_25 = '# Test file'
    var_26 = []
    var_27 = []
    var_28 = str(var_24)
    var_29 = [var_28]
    var_30 = module_1.find(var_29, var_0, var_26, var_27)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = str(var_24)
    var_34 = len(var_26)
    assert var_34 == 0
    var_35 = len(var_27)
    assert var_35 == 0
    var_36 = 'skipped_dir'
    var_37 = var_1 / var_36
    var_38 = 'file.py'
    var_39 = var_37 / var_38
    var_40 = '# Python file'
    var_41 = [var_36]
    var_42 = module_0.Config()
    var_43 = []
    var_44 = []
    var_45 = str(var_37)
    var_46 = [var_45]
    var_47 = module_1.find(var_46, var_42, var_43, var_44)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 0
    var_50 = len(var_43)
    assert var_50 == 1
    var_51 = len(var_44)
    assert var_51 == 0



# Parsed testcases at query #17
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
    assert var_13 == 1



# Parsed testcases at query #18
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
    var_14 = '.py'
    var_15 = module_0.Config()
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_15, var_18, var_19)
    var_21 = list(var_20)
    var_22 = 'skip_me.py'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = [var_22]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_24, var_26, var_27)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = [var_1, var_8, var_16]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# not python'
    var_3 = '# test in subdir'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test1.py'
    var_8 = 'test3.py'
    var_9 = 'test2.txt'
    var_10 = len(var_5)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = '/nonexistent/path'
    var_15 = [var_14]
    var_16 = module_1.find(var_15, var_4, var_12, var_13)
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
    var_24 = [var_18]
    var_25 = module_1.find(var_24, var_4, var_22, var_23)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_22)
    assert var_28 == 0
    var_29 = len(var_23)
    assert var_29 == 0
    var_30 = 'skip_me'
    var_31 = '# should be skipped'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_18, var_33, var_34, var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = len(var_34)
    assert var_39 == 1
    var_40 = len(var_35)
    assert var_40 == 0



# Parsed testcases at query #20
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
    var_7 = 'test_dir'
    var_8 = True
    var_9 = '# test'
    var_10 = 'not python'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = [var_7]
    var_15 = module_1.find(var_14, var_11, var_12, var_13)
    var_16 = list(var_15)
    var_17 = 'test_dir/file1.py'
    var_18 = 'test_dir/file2.txt'
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = 'non_existent_path.py'
    var_23 = [var_22]
    var_24 = module_1.find(var_23, var_19, var_20, var_21)
    var_25 = list(var_24)
    var_26 = 'test_skip_dir'
    var_27 = '# test'
    var_28 = 'skip_me.py'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = [var_26]
    var_34 = module_1.find(var_33, var_30, var_31, var_32)
    var_35 = list(var_34)
    var_36 = 'test_skip_dir/skip_me.py'
    var_37 = 'test_multi_dir'
    var_38 = '# test'
    var_39 = '# test'
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = 'multi_file.py'
    var_44 = [var_37, var_43]
    var_45 = module_1.find(var_44, var_40, var_41, var_42)
    var_46 = list(var_45)
    var_47 = 'test_multi_dir/multi.py'



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'sub_dir'
    var_2 = 'skipped_dir'
    var_3 = 'file1.py'
    var_4 = '# Python file'
    var_5 = 'file2.txt'
    var_6 = '# Not a Python file'
    var_7 = 'file3.py'
    var_8 = '# Python file in subdirectory'
    var_9 = 'file4.py'
    var_10 = '# Python file in skipped directory'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 0

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
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '# Python file'
    var_2 = [var_0]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

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
    var_9 = 'test2.txt'
    var_10 = len(var_5)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 0
    var_12 = module_0.Config()
    var_13 = 'nonexistent'
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 1
    var_18 = 'single.py'
    var_19 = '# test'
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = len(var_22)
    assert var_24 == 0
    var_25 = 'skipme'
    var_26 = '# test'
    var_27 = [var_26]
    var_28 = module_0.Config()
    var_29 = []
    var_30 = []
    var_31 = list(var_24)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = len(var_29)
    assert var_33 == 1
    var_34 = len(var_30)
    assert var_34 == 0



# Parsed testcases at query #23
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
    var_6 = module_1.find(var_3, var_0, var_1, var_2)
    var_7 = list(var_6)
    var_8 = 'test1.py'
    var_9 = 'test2.py'
    var_10 = 'test.txt'
    var_11 = '# test'
    var_12 = '# test'
    var_13 = '# test'
    var_14 = module_1.find(var_3, var_0, var_1, var_2)
    var_15 = list(var_14)
    var_16 = set(var_15)
    var_17 = '/nonexistent/path'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_0, var_1, var_2)
    var_20 = list(var_19)
    var_21 = 'subdir'
    var_22 = 'test.py'
    var_23 = '# test'
    var_24 = module_1.find(var_18, var_0, var_1, var_2)
    var_25 = list(var_24)
    var_26 = len(var_1)
    assert var_26 == 1
    var_27 = 'test.py'
    var_28 = '# test'
    var_29 = module_1.find(var_18, var_0, var_1, var_2)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 1



# Parsed testcases at query #24
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
    var_6 = '# Python file'
    var_7 = 'Not a Python file'
    var_8 = module_1.find(var_2, var_0, var_3, var_4)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = len(var_3)
    assert var_11 == 0
    var_12 = len(var_4)
    assert var_12 == 0
    var_13 = 'test_directory/file1.py'
    var_14 = 'test_directory/file2.txt'
    var_15 = 'non_existent_path'
    var_16 = [var_15]
    var_17 = module_1.find(var_16, var_0, var_3, var_4)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = len(var_3)
    assert var_20 == 0
    var_21 = len(var_4)
    assert var_21 == 1
    var_22 = 'test_file.py'
    var_23 = [var_22]
    var_24 = '# Python file'
    var_25 = module_1.find(var_23, var_0, var_3, var_4)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_3)
    assert var_28 == 0
    var_29 = len(var_4)
    assert var_29 == 0
    var_30 = 'test_skipped_directory'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = [var_30]
    var_34 = []
    var_35 = []
    var_36 = '# Python file'
    var_37 = module_1.find(var_33, var_32, var_34, var_35)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 0
    var_40 = len(var_34)
    assert var_40 == 1
    var_41 = len(var_35)
    assert var_41 == 0
    var_42 = 'test_skipped_directory/file.py'



# Parsed testcases at query #25
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
    var_11 = 'nonexistent.py'
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = []
    var_18 = []
    var_19 = len(var_17)
    assert var_19 == 0
    var_20 = len(var_18)
    assert var_20 == 1
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = len(var_22)
    assert var_24 == 0
    var_25 = []
    var_26 = []
    var_27 = len(var_25)
    assert var_27 == 1
    var_28 = len(var_26)
    assert var_28 == 1



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'sub_test.py'
    var_4 = '# sub test'
    var_5 = 'skip_dir'
    var_6 = 'skip.py'
    var_7 = '# skip'
    var_8 = 'readme.txt'
    var_9 = 'readme'
    var_10 = module_0.Config()
    var_11 = []
    var_12 = []
    var_13 = len(var_11)
    assert var_13 == 0
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = [var_5]
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = len(var_17)
    assert var_19 == 1
    var_20 = len(var_18)
    assert var_20 == 0
    var_21 = module_0.Config()
    var_22 = '/nonexistent/path'
    var_23 = []
    var_24 = []
    var_25 = len(var_23)
    assert var_25 == 0
    var_26 = len(var_24)
    assert var_26 == 1
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = len(var_28)
    assert var_30 == 0
    var_31 = len(var_29)
    assert var_31 == 0
    var_32 = 'link'
    var_33 = True
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = list(var_2)
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = len(var_35)
    assert var_39 == 0
    var_40 = len(var_36)
    assert var_40 == 0



# Parsed testcases at query #27
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
    var_14 = module_0.Config()
    var_15 = 'non_existent_path'
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_16, var_14, var_17, var_18)
    var_20 = list(var_19)
    var_21 = module_0.Config()
    var_22 = [var_8]
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_22, var_21, var_23, var_24)
    var_26 = list(var_25)
    var_27 = module_0.Config()
    var_28 = [var_1, var_8]
    var_29 = []
    var_30 = []
    var_31 = module_1.find(var_28, var_27, var_29, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #28
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
    var_12 = 'test.py'
    var_13 = "print('hello')"
    var_14 = 'test.txt'
    var_15 = 'hello'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_0, var_16, var_17, var_18)
    var_20 = list(var_19)
    var_21 = '/nonexistent/path'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_22, var_23, var_24, var_25)
    var_27 = list(var_26)
    var_28 = 'skip_me'
    var_29 = 'test.py'
    var_30 = "print('hello')"
    var_31 = 'normal.py'
    var_32 = "print('hello')"
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_22, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = 'real'
    var_40 = 'test.py'
    var_41 = "print('hello')"
    var_42 = 'link'
    var_43 = True
    var_44 = module_0.Config()
    var_45 = []
    var_46 = []
    var_47 = module_1.find(var_22, var_44, var_45, var_46)
    var_48 = list(var_47)



# Parsed testcases at query #29
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
    var_17 = 'text file'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = set(var_11)
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/non/existent/path'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = 'test1.py'
    var_30 = 'test2.py'
    var_31 = "print('test1')"
    var_32 = "print('test2')"
    var_33 = 'skipped_dir'
    var_34 = 'test3.py'
    var_35 = "print('test3')"
    var_36 = [var_33]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = set(var_28)



# Parsed testcases at query #30
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
    var_21 = '# Python file 1'
    var_22 = 'Text file'
    var_23 = '# Python file 3'
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_14, var_24, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'file1.py'
    var_31 = 'file3.py'
    var_32 = 'skipdir'
    var_33 = '# Python file 1'
    var_34 = '# Python file 2'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_14, var_36, var_37, var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = 'file1.py'
    var_43 = len(var_37)
    assert var_43 == 1
    var_44 = [var_35, var_6]
    var_45 = [var_6]
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_44, var_46, var_47, var_48)
    var_50 = list(var_49)



# Parsed testcases at query #31
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not Python'
    var_5 = 'sub_dir'
    var_6 = 'file3.py'
    var_7 = '# Python in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Should be skipped'
    var_11 = 'nonexistent.py'
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = []
    var_18 = []
    var_19 = len(var_17)
    assert var_19 == 0
    var_20 = len(var_18)
    assert var_20 == 1
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = len(var_22)
    assert var_24 == 0
    var_25 = []
    var_26 = []
    var_27 = len(var_25)
    assert var_27 == 1
    var_28 = len(var_26)
    assert var_28 == 1



# Parsed testcases at query #32
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
    var_8 = 'test'
    var_9 = len(var_3)
    assert var_9 == 0
    var_10 = len(var_4)
    assert var_10 == 0
    var_11 = 'non_existent_path'
    var_12 = [var_11]
    var_13 = []
    var_14 = []
    var_15 = module_1.find(var_12, var_0, var_13, var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 1
    var_20 = 'test_file.py'
    var_21 = [var_20]
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
    var_29 = 'skip_this_dir'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = [var_1]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = len(var_33)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0



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
    var_24 = '/non/existent/path'
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
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []



# Parsed testcases at query #34
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
    var_14 = '.py'
    var_15 = module_0.Config()
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_15, var_18, var_19)
    var_21 = list(var_20)
    var_22 = module_0.Config()
    var_23 = 'test_skipped_file.py'
    var_24 = [var_23]
    var_25 = []
    var_26 = []
    var_27 = True
    var_28 = module_1.find(var_24, var_22, var_25, var_26)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = [var_1, var_8]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test1.py'
    var_1 = '# test'
    var_2 = 'test2.py'
    var_3 = 'test3.txt'
    var_4 = '# not python'
    var_5 = 'subdir'
    var_6 = 'test4.py'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = '.py'
    var_11 = [var_0]
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = 0
    var_17 = var_13[var_16]
    var_18 = module_0.Config()
    var_19 = '/nonexistent/path'
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_18, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = len(var_22)
    assert var_26 == 1
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_20, var_27, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = var_31[var_16]



# Parsed testcases at query #36
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
    var_14 = 'file.txt'
    var_15 = "print('hello')"
    var_16 = "print('world')"
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
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_25, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = 'subdir'
    var_41 = 'file.py'
    var_42 = "print('hello')"
    var_43 = 'symlink'
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = module_1.find(var_25, var_45, var_46, var_47)
    var_49 = list(var_48)
    var_50 = set(var_49)
    var_51 = []
    var_52 = []
    var_53 = False
    var_54 = module_0.Config()
    var_55 = module_1.find(var_25, var_54, var_51, var_52)
    var_56 = list(var_55)
    var_57 = set(var_56)



# Parsed testcases at query #37
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



# Parsed testcases at query #38
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
    var_26 = module_1.find(var_21, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_1)
    assert var_29 == 1
    var_30 = len(var_2)
    assert var_30 == 0



# Parsed testcases at query #39
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
    var_14 = 'test.txt'
    var_15 = "print('test1')"
    var_16 = "print('test2')"
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
    var_32 = 'test.py'
    var_33 = "print('test')"
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_23, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = len(var_36)
    assert var_40 == 1
    var_41 = 'target'
    var_42 = 'test.py'
    var_43 = "print('test')"
    var_44 = 'symlink'
    var_45 = False
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_40, var_46, var_47, var_48)
    var_50 = list(var_49)



# Parsed testcases at query #40
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = 'file1.py'
    var_5 = '# Python file'
    var_6 = 'file2.txt'
    var_7 = 'Not a Python file'
    var_8 = 'skipped_file.py'
    var_9 = '# Should be skipped'
    var_10 = 'sub_dir'
    var_11 = 'file3.py'
    var_12 = '# Python file in subdirectory'
    var_13 = 'symlink_dir'
    var_14 = 'nonexistent'
    var_15 = len(var_2)
    assert var_15 == 1
    var_16 = len(var_1)
    assert var_16 == 1



# Parsed testcases at query #41
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
    var_8 = '.py'
    var_9 = 'non_existent_path'
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_10, var_0, var_11, var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'test_file.py'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_0, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'skip_directory'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_27)
    var_33 = 'skip_file.py'
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = [var_33]
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_36, var_35, var_37, var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 0



# Parsed testcases at query #42
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'subtest.py'
    var_4 = '# subtest'
    var_5 = 'readme.txt'
    var_6 = 'readme'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = len(var_8)
    assert var_10 == 0
    var_11 = len(var_9)
    assert var_11 == 0
    var_12 = module_0.Config()
    var_13 = []
    var_14 = []
    var_15 = '/non/existent/path'
    var_16 = [var_15]
    var_17 = module_1.find(var_16, var_12, var_13, var_14)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = len(var_13)
    assert var_20 == 0
    var_21 = len(var_14)
    assert var_21 == 1
    var_22 = b'# test'
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = module_1.find(var_22, var_23, var_24, var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = len(var_24)
    assert var_29 == 0
    var_30 = len(var_25)
    assert var_30 == 0



# Parsed testcases at query #43
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# not python'
    var_3 = '# test in subdir'
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
    var_32 = '# test in skipped dir'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_13, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = 'test1.py'
    var_41 = len(var_35)
    assert var_41 == 1
    var_42 = len(var_36)
    assert var_42 == 0



# Parsed testcases at query #44
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
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = module_0.Config()
    var_24 = [var_8]
    var_25 = []
    var_26 = []
    var_27 = 'test_dir/skip_me'
    var_28 = module_1.find(var_24, var_23, var_25, var_26)
    var_29 = list(var_28)
    var_30 = len(var_29)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)



# Parsed testcases at query #45
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not Python'
    var_5 = 'sub_dir'
    var_6 = 'file3.py'
    var_7 = '# Python in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Should be skipped'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = 'single.py'
    var_16 = '# Single file'
    var_17 = []
    var_18 = []
    var_19 = 'nonexistent/path'
    var_20 = [var_19]
    var_21 = []
    var_22 = module_1.find(var_20, var_11, var_21, var_13)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = len(var_13)
    assert var_25 == 1
    var_26 = []
    var_27 = []
    var_28 = len(var_23)
    assert var_28 == 0



# Parsed testcases at query #46
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
    var_20 = []
    var_21 = []
    var_22 = b'# test'
    var_23 = [var_17]
    var_24 = module_1.find(var_23, var_0, var_20, var_21)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_20)
    assert var_27 == 0
    var_28 = len(var_21)
    assert var_28 == 0
    var_29 = []
    var_30 = []
    var_31 = 'skipme'
    var_32 = '# test'
    var_33 = module_1.find(var_17, var_0, var_29, var_30)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = len(var_29)
    assert var_36 == 1
    var_37 = len(var_30)
    assert var_37 == 0



# Parsed testcases at query #47
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
    var_23 = 'test_skip_directory'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_26, var_25, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_27)
    var_32 = module_0.Config()
    var_33 = [var_1, var_8]
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_33, var_32, var_34, var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)



# Parsed testcases at query #48
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
    var_7 = '# Python file in subdirectory'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Python file in skipped directory'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 1
    var_15 = len(var_13)
    assert var_15 == 0

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
    var_0 = 'test_file.py'
    var_1 = '# Test file'
    var_2 = module_0.Config()
    var_3 = [var_0]
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_3, var_2, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0



# Parsed testcases at query #49
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = 'skipped_dir'
    var_4 = 'skipped.py'
    var_5 = '# skipped file'
    var_6 = 'nonexistent.py'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = len(var_8)
    assert var_10 == 1
    var_11 = len(var_9)
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = len(var_13)
    assert var_15 == 1
    var_16 = []
    var_17 = []
    var_18 = len(var_16)
    assert var_18 == 0
    var_19 = len(var_17)
    assert var_19 == 0
    var_20 = []
    var_21 = []
    var_22 = len(var_20)
    assert var_22 == 1
    var_23 = len(var_21)
    assert var_23 == 1



# Parsed testcases at query #50
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
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
    var_15 = module_0.Config()
    var_16 = 'test_file.txt'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_15, var_18, var_19)
    var_21 = list(var_20)
    var_22 = 'test_skipped_dir'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = [var_22]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_24, var_26, var_27)
    var_29 = list(var_28)
    var_30 = 'test_skipped_file.py'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = [var_30]
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_33, var_32, var_34, var_35)
    var_37 = list(var_36)



# Parsed testcases at query #51
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
    var_29 = 'test_skip.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_28, var_31, var_32, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #52
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
    var_10 = 'test_directory/skip_file.py'
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = [var_1]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_12, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = len(var_14)
    assert var_19 == 1
    var_20 = len(var_15)
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
    var_31 = module_0.Config()
    var_32 = 'test_directory/file1.py'
    var_33 = [var_32]
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_33, var_31, var_34, var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0
    var_40 = len(var_35)
    assert var_40 == 0
    var_41 = True
    var_42 = module_0.Config()
    var_43 = 'test_directory_with_symlinks'
    var_44 = [var_43]
    var_45 = []
    var_46 = []
    var_47 = module_1.find(var_44, var_42, var_45, var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = len(var_45)
    assert var_50 == 0
    var_51 = len(var_46)
    assert var_51 == 0



# Parsed testcases at query #53
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = 'skipped.py'
    var_4 = '# skipped file'
    var_5 = 'nonexistent.py'
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []



# Parsed testcases at query #54
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
    var_6 = '# test'
    var_7 = 'test_file.py'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_8, var_9, var_10, var_11)
    var_13 = list(var_12)
    var_14 = 'test_dir'
    var_15 = '# test1'
    var_16 = '# test2'
    var_17 = '# not python'
    var_18 = [var_14]
    var_19 = module_0.Config()
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_18, var_19, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = '.py'
    var_26 = 'test_dir/file1.py'
    var_27 = 'test_dir/file2.py'
    var_28 = 'test_dir/file3.txt'
    var_29 = 'non_existent_path.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_30, var_31, var_32, var_33)
    var_35 = list(var_34)
    var_36 = '# skip'
    var_37 = 'skip_me.py'
    var_38 = [var_37]
    var_39 = [var_37]
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = module_1.find(var_38, var_40, var_41, var_42)
    var_44 = list(var_43)
    var_45 = '# mixed'
    var_46 = 'mixed_dir'
    var_47 = '# mixed inner'
    var_48 = 'mixed_file.py'
    var_49 = 'non_existent.py'
    var_50 = [var_48, var_46, var_49]
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = module_1.find(var_50, var_51, var_52, var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'mixed_dir/mixed_inner.py'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test file'
    var_6 = 'test_file.py'
    var_7 = '# test file'
    var_8 = 'non_existent.py'
    var_9 = 'skipped.py'
    var_10 = '# skipped file'
    var_11 = 'broken_link.py'
    var_12 = 'non_existent_target.py'
    var_13 = [var_3, var_6, var_8, var_11]
    var_14 = module_1.find(var_13, var_0, var_1, var_2)
    var_15 = list(var_14)
    var_16 = 'test.py'



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
    var_23 = module_0.Config()
    var_24 = 'skipped_directory'
    var_25 = [var_24]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_23, var_26, var_27)
    var_29 = list(var_28)
    var_30 = len(var_26)
    var_31 = module_0.Config()
    var_32 = [var_1, var_8]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)



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
    var_5 = '# test2'
    var_6 = 'text'
    var_7 = 'test.py'
    var_8 = 'test2.py'
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = 'nonexistent/path'
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



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test1.py'
    var_2 = '# Python file 1'
    var_3 = 'test2.py'
    var_4 = '# Python file 2'
    var_5 = 'test.txt'
    var_6 = '# Not a Python file'
    var_7 = 'subdir'
    var_8 = 'test3.py'
    var_9 = '# Python file 3'
    var_10 = 'skipped_dir'
    var_11 = 'test4.py'
    var_12 = '# Python file in skipped dir'
    var_13 = [var_10]
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 1
    var_18 = len(var_16)
    assert var_18 == 0
    var_19 = module_0.Config()
    var_20 = 'non_existent_path'
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = module_1.find(var_21, var_19, var_22, var_23)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = len(var_23)
    assert var_28 == 1
    var_29 = 'test_file.py'
    var_30 = var_20 / var_29
    var_31 = '# Single Python file'
    var_32 = module_0.Config()
    var_33 = str(var_30)
    var_34 = [var_33]
    var_35 = []
    var_36 = []
    var_37 = module_1.find(var_34, var_32, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = str(var_30)
    var_41 = len(var_35)
    assert var_41 == 0
    var_42 = len(var_36)
    assert var_42 == 0
    var_43 = 'skipped_file.py'
    var_44 = var_20 / var_43
    var_45 = '# Skipped Python file'
    var_46 = [var_43]
    var_47 = module_0.Config()
    var_48 = str(var_44)
    var_49 = [var_48]
    var_50 = []
    var_51 = []
    var_52 = module_1.find(var_49, var_47, var_50, var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 0
    var_55 = len(var_50)
    assert var_55 == 1
    var_56 = str(var_44)
    var_57 = len(var_51)
    assert var_57 == 0



# Parsed testcases at query #5
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
    var_14 = '.py'
    var_15 = module_0.Config()
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_15, var_18, var_19)
    var_21 = list(var_20)
    var_22 = 'skip_me.py'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = [var_22]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_24, var_26, var_27)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = 'non_existent'
    var_32 = [var_1, var_8, var_31]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_30, var_33, var_34)
    var_36 = list(var_35)



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
    var_21 = module_0.Config()
    var_22 = [var_15]
    var_23 = []
    var_24 = []
    var_25 = module_1.find(var_22, var_21, var_23, var_24)
    var_26 = list(var_25)
    var_27 = module_0.Config()
    var_28 = [var_15]
    var_29 = []
    var_30 = []
    var_31 = module_1.find(var_28, var_27, var_29, var_30)
    var_32 = list(var_31)



# Parsed testcases at query #7
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
    var_10 = 'test.py'
    var_11 = "print('hello')"
    var_12 = 'test.txt'
    var_13 = 'not python'
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = module_0.Config()
    var_18 = []
    var_19 = []
    var_20 = '/non/existent/path'
    var_21 = [var_20]
    var_22 = module_1.find(var_21, var_17, var_18, var_19)
    var_23 = list(var_22)
    var_24 = b"print('hello')"
    var_25 = 'test.py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = 'skip_me'
    var_31 = 'test.py'
    var_32 = "print('hello')"
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = list(var_20)
    var_38 = any(var_21)



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
    var_5 = '# test file'
    var_6 = 'non_existent_path'
    var_7 = 'direct_file.py'
    var_8 = '# direct file'
    var_9 = [var_3, var_6, var_7]
    var_10 = module_1.find(var_9, var_0, var_1, var_2)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 'test.py'
    var_14 = 'direct_file.py'
    var_15 = len(var_1)
    assert var_15 == 0



# Parsed testcases at query #9
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
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_14, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = len(var_36)
    assert var_40 == 1
    var_41 = 'dir1'
    var_42 = '# test'
    var_43 = 'symlink'
    var_44 = True
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = module_1.find(var_14, var_45, var_46, var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1



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



# Parsed testcases at query #11
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
    var_16 = 'skipped_directory'
    var_17 = [var_16]
    var_18 = module_0.Config()
    var_19 = [var_1]
    var_20 = []
    var_21 = []
    var_22 = module_1.find(var_19, var_18, var_20, var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = module_0.Config()
    var_26 = 'test_file.py'
    var_27 = [var_26]
    var_28 = []
    var_29 = []
    var_30 = module_1.find(var_27, var_25, var_28, var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = True
    var_34 = module_0.Config()
    var_35 = 'symlink_directory'
    var_36 = [var_35]
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_36, var_34, var_37, var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test file'
    var_6 = '# not a Python file'
    var_7 = 'non_existent_path.py'
    var_8 = 'direct_test.py'
    var_9 = '# direct test file'
    var_10 = 'test.py'
    var_11 = 'test.txt'



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
    var_8 = '.py'
    var_9 = len(var_3)
    assert var_9 == 0
    var_10 = len(var_4)
    assert var_10 == 0
    var_11 = 'non_existent_path'
    var_12 = [var_11]
    var_13 = []
    var_14 = []
    var_15 = module_1.find(var_12, var_0, var_13, var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 1
    var_20 = 'test_file.py'
    var_21 = [var_20]
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
    var_29 = 'test_skipped_directory'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = [var_29]
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_32, var_31, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = len(var_33)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0
    var_40 = module_0.Config()
    var_41 = 'test_non_python_directory'
    var_42 = [var_41]
    var_43 = []
    var_44 = []
    var_45 = module_1.find(var_42, var_40, var_43, var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 0
    var_48 = len(var_43)
    assert var_48 == 0
    var_49 = len(var_44)
    assert var_49 == 0



# Parsed testcases at query #14
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
    var_20 = 'file1.py'
    var_21 = 'w'
    var_22 = open(var_6, var_21)
    var_23 = 'file2.py'
    var_24 = 'file3.txt'
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_14, var_25, var_26, var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = '.py'
    var_32 = 'file1.py'
    var_33 = 'w'
    var_34 = open(var_6, var_33)
    var_35 = 'file2.py'
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = module_1.find(var_14, var_36, var_37, var_38)
    var_40 = list(var_39)
    var_41 = len(var_37)
    assert var_41 == 1
    var_42 = 0
    var_43 = var_37[var_42]
    var_44 = 'broken_symlink.py'
    var_45 = 'non_existent.py'
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_14, var_46, var_47, var_48)
    var_50 = list(var_49)
    var_51 = len(var_48)
    assert var_51 == 1



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
    var_14 = 'non_existent_file.py'
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = module_1.find(var_15, var_16, var_17, var_18)
    var_20 = list(var_19)
    var_21 = 'test_dir'
    var_22 = '# test1'
    var_23 = '# test2'
    var_24 = 'text'
    var_25 = [var_21]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_25, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = '.py'
    var_33 = 'test_dir/file1.py'
    var_34 = 'test_dir/file2.py'
    var_35 = 'test_dir/non_py_file.txt'
    var_36 = 'test_dir/skip_dir'
    var_37 = '# test'
    var_38 = [var_21]
    var_39 = 'skip_dir'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_38, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = len(var_42)
    assert var_46 == 1
    var_47 = 'test_dir/skip_dir/file.py'
    var_48 = 'non_existent.py'
    var_49 = [var_6, var_48, var_21]
    var_50 = '# test'
    var_51 = '# test'
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = module_1.find(var_49, var_52, var_53, var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = 'test_dir/file.py'



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '# Python file 1'
    var_2 = 'file2.py'
    var_3 = '# Python file 2'
    var_4 = 'file3.txt'
    var_5 = '# Not a Python file'
    var_6 = 'subdir'
    var_7 = 'file4.py'
    var_8 = '# Python file 4'
    var_9 = 'skipped_dir'
    var_10 = 'file5.py'
    var_11 = '# Python file 5'
    var_12 = [var_9]
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 1
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = module_0.Config()
    var_19 = '/non/existent/path'
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_18, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 1
    var_28 = 'single_file.py'
    var_29 = var_19 / var_28
    var_30 = '# Single Python file'
    var_31 = module_0.Config()
    var_32 = str(var_29)
    var_33 = [var_32]
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_33, var_31, var_34, var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = str(var_29)
    var_40 = len(var_34)
    assert var_40 == 0
    var_41 = len(var_35)
    assert var_41 == 0
    var_42 = 'dir1'
    var_43 = var_19 / var_42
    var_44 = 'file1.py'
    var_45 = var_43 / var_44
    var_46 = '# Python file 1'
    var_47 = 'symlink'
    var_48 = var_39 / var_47
    var_49 = True
    var_50 = module_0.Config()
    var_51 = str(var_48)
    var_52 = [var_51]
    var_53 = []
    var_54 = []
    var_55 = module_1.find(var_52, var_50, var_53, var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = len(var_53)
    assert var_58 == 0
    var_59 = len(var_54)
    assert var_59 == 0



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
    var_5 = True
    var_6 = '# Python file 1'
    var_7 = '# Not a Python file'
    var_8 = '# Python file 2'
    var_9 = module_1.find(var_2, var_0, var_3, var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = 'test_directory/file1.py'
    var_15 = 'test_directory/file2.txt'
    var_16 = 'test_directory/file3.py'
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
    var_30 = '# Test file'
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
    var_39 = 'test_skip_directory'
    var_40 = [var_39]
    var_41 = []
    var_42 = []
    var_43 = 'test_skip_directory/skip_me'
    var_44 = '# Skipped file'
    var_45 = '# Not skipped file'
    var_46 = module_1.find(var_40, var_38, var_41, var_42)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = len(var_41)
    assert var_49 == 1
    var_50 = len(var_42)
    assert var_50 == 0
    var_51 = 'test_skip_directory/skip_me/file.py'
    var_52 = 'test_skip_directory/file.py'



# Parsed testcases at query #18
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
    var_38 = 'real_dir'
    var_39 = 'symlink_dir'
    var_40 = 'test.py'
    var_41 = "print('test')"
    var_42 = True
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_31, var_43, var_44, var_45)
    var_47 = list(var_46)
    var_48 = False
    var_49 = module_0.Config()
    var_50 = []
    var_51 = []
    var_52 = list(var_37)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_directory'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = var_1 / var_5
    var_7 = '# test'
    var_8 = 'file2.py'
    var_9 = 'file3.txt'
    var_10 = len(var_3)
    assert var_10 == 0
    var_11 = len(var_4)
    assert var_11 == 0
    var_12 = 'non_existent_path'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = module_1.find(var_13, var_0, var_14, var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_14)
    assert var_19 == 0
    var_20 = len(var_15)
    assert var_20 == 1
    var_21 = b'# test'
    var_22 = []
    var_23 = []
    var_24 = module_1.find(var_13, var_0, var_22, var_23)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = 'skip_me'
    var_30 = var_24 / var_29
    var_31 = 'file.py'
    var_32 = var_30 / var_31
    var_33 = '# test'
    var_34 = [var_29]
    var_35 = module_0.Config()
    var_36 = len(var_25)
    assert var_36 == 0
    var_37 = len(var_22)
    assert var_37 == 1
    var_38 = len(var_23)
    assert var_38 == 0



# Parsed testcases at query #20
#--------------------------




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
    var_47 = 'test_mixed_dir'
    var_48 = '# mixed'
    var_49 = '# mixed file'
    var_50 = 'test_mixed_file.py'
    var_51 = [var_47, var_50]
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = module_1.find(var_51, var_52, var_53, var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = 'test_mixed_dir/mixed.py'



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'file1.py'
    var_2 = '# Python file'
    var_3 = 'file2.txt'
    var_4 = 'Not Python'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = '# Python in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Should be skipped'
    var_11 = 'broken_link'
    var_12 = 'nonexistent'
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = []
    var_18 = []
    var_19 = len(var_17)
    assert var_19 == 0
    var_20 = []
    var_21 = []
    var_22 = len(var_20)
    assert var_22 == 0
    var_23 = len(var_21)
    assert var_23 == 0
    var_24 = []
    var_25 = []
    var_26 = 'test_file.py'
    var_27 = '# Direct file'
    var_28 = 'missing'



# Parsed testcases at query #23
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
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_25, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = 'file.py'
    var_41 = "print('hello')"
    var_42 = [var_41]
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_25, var_43, var_44, var_45)
    var_47 = list(var_46)



# Parsed testcases at query #25
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
    var_8 = '# Text file'
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
    var_19 = module_1.find(var_18, var_0, var_3, var_4)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = len(var_3)
    assert var_22 == 0
    var_23 = len(var_4)
    assert var_23 == 1
    var_24 = 'test_file.py'
    var_25 = [var_24]
    var_26 = '# Python file'
    var_27 = module_1.find(var_25, var_0, var_3, var_4)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = len(var_3)
    assert var_30 == 0
    var_31 = len(var_4)
    assert var_31 == 0
    var_32 = 'skip_me.py'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = 'test_skip_directory'
    var_36 = [var_35]
    var_37 = []
    var_38 = []
    var_39 = '# Skipped Python file'
    var_40 = '# Kept Python file'
    var_41 = module_1.find(var_36, var_34, var_37, var_38)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = len(var_37)
    assert var_44 == 1
    var_45 = 'test_skip_directory/skip_me.py'
    var_46 = len(var_38)
    assert var_46 == 0
    var_47 = 'test_skip_directory/keep_me.py'



# Parsed testcases at query #26
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
    var_8 = '# Text file'
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
    var_36 = 'skip_me'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = 'test_skip_directory'
    var_40 = [var_39]
    var_41 = []
    var_42 = []
    var_43 = 'test_skip_directory/skip_me'
    var_44 = '# Python file 1'
    var_45 = '# Python file 2'
    var_46 = module_1.find(var_40, var_38, var_41, var_42)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = len(var_41)
    assert var_49 == 1
    var_50 = len(var_42)
    assert var_50 == 0
    var_51 = 'test_skip_directory/file1.py'
    var_52 = 'test_skip_directory/skip_me/file2.py'



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = '# test'
    var_5 = 'skipped.py'
    var_6 = '# skipped'
    var_7 = 'nested'
    var_8 = 'nested.py'
    var_9 = '# nested'
    var_10 = 'nonexistent.py'
    var_11 = 'skipped'
    var_12 = '.py'
    var_13 = len(var_1)
    assert var_13 == 1
    var_14 = len(var_2)
    assert var_14 == 1



# Parsed testcases at query #28
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
    var_17 = 'non_existent_path.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = 'skip_me.py'
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



# Parsed testcases at query #29
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
    var_10 = []
    var_11 = []
    var_12 = 'nonexistent_path'
    var_13 = [var_12]
    var_14 = module_1.find(var_13, var_0, var_10, var_11)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = len(var_11)
    assert var_17 == 1
    var_18 = []
    var_19 = []
    var_20 = b'# test'
    var_21 = module_1.find(var_20, var_0, var_18, var_19)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = []
    var_25 = []
    var_26 = 'skip_me'
    var_27 = '# should be skipped'
    var_28 = module_1.find(var_23, var_0, var_24, var_25)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = len(var_24)
    assert var_31 == 1



# Parsed testcases at query #30
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
    var_14 = 'test_directory/file1.py'
    var_15 = 'test_directory/file2.py'
    var_16 = [var_14, var_15]
    var_17 = sorted(var_13)
    var_18 = sorted(var_16)
    var_19 = module_0.Config()
    var_20 = 'non_existent_path'
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = module_1.find(var_21, var_19, var_22, var_23)
    var_25 = list(var_24)
    var_26 = module_0.Config()
    var_27 = 'test_skipped_file.py'
    var_28 = [var_27]
    var_29 = []
    var_30 = []
    var_31 = True
    var_32 = module_1.find(var_28, var_26, var_29, var_30)
    var_33 = list(var_32)
    var_34 = module_0.Config()
    var_35 = [var_1, var_8]
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_35, var_34, var_36, var_37)
    var_39 = list(var_38)
    var_40 = [var_1, var_14, var_15]
    var_41 = sorted(var_39)
    var_42 = sorted(var_40)



# Parsed testcases at query #31
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
    var_10 = 'test_file.py'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_11, var_0, var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = len(var_12)
    assert var_17 == 0
    var_18 = len(var_13)
    assert var_18 == 0
    var_19 = 'non_existent_path'
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = module_1.find(var_20, var_0, var_21, var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = len(var_21)
    assert var_26 == 0
    var_27 = len(var_22)
    assert var_27 == 1
    var_28 = 'skip_me'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = [var_23]
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



# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_directory'
    var_4 = True
    var_5 = '# test'
    var_6 = 'test_file.py'
    var_7 = '# test'
    var_8 = 'non_existent.py'
    var_9 = 'skipped.py'
    var_10 = '# skipped'
    var_11 = 'broken_link.py'
    var_12 = 'non_existent.py'
    var_13 = 'test.py'



# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
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
    var_15 = "print('hello')"
    var_16 = "print('world')"
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
    var_30 = 'skipped.py'
    var_31 = 'skipped_dir'
    var_32 = "print('hello')"
    var_33 = "print('skipped')"
    var_34 = [var_30, var_31]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_21, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = set(var_36)
    var_41 = 'test.py'
    var_42 = "print('hello')"
    var_43 = b"print('world')"
    var_44 = module_0.Config()
    var_45 = []
    var_46 = []
    var_47 = module_1.find(var_43, var_44, var_45, var_46)
    var_48 = list(var_47)
    var_49 = set(var_48)



# Parsed testcases at query #35
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
    var_9 = 'test2.txt'
    var_10 = len(var_5)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = '/nonexistent/path'
    var_15 = [var_14]
    var_16 = module_1.find(var_15, var_4, var_12, var_13)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = len(var_12)
    assert var_19 == 0
    var_20 = len(var_13)
    assert var_20 == 1
    var_21 = b'# Single file test'
    var_22 = []
    var_23 = []
    var_24 = module_1.find(var_21, var_4, var_22, var_23)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_22)
    assert var_27 == 0
    var_28 = len(var_23)
    assert var_28 == 0
    var_29 = 'skip_me'
    var_30 = '# Should be skipped'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = module_1.find(var_27, var_32, var_33, var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = len(var_33)
    assert var_38 == 1
    var_39 = len(var_34)
    assert var_39 == 0



# Parsed testcases at query #36
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
    var_24 = 'not python'
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_13, var_25, var_26, var_27)
    var_29 = list(var_28)
    var_30 = set(var_29)
    var_31 = 'subdir'
    var_32 = 'file.py'
    var_33 = "print('in subdir')"
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_13, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = 'file.py'
    var_41 = "print('file')"
    var_42 = b"print('other')"
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_13, var_43, var_44, var_45)
    var_47 = list(var_46)
    var_48 = set(var_47)



# Parsed testcases at query #37
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
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_18, var_16, var_19, var_20)
    var_22 = list(var_21)
    var_23 = module_0.Config()
    var_24 = 'skipped_file.py'
    var_25 = [var_24]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_23, var_26, var_27)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = [var_1, var_8]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #38
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
    var_9 = '/nonexistent'
    var_10 = [var_9]
    var_11 = module_1.find(var_10, var_4, var_5, var_6)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = len(var_6)
    assert var_14 == 1
    var_15 = [var_3]
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = len(var_12)
    assert var_19 == 1
    var_20 = len(var_17)
    assert var_20 == 1
    var_21 = len(var_12)
    assert var_21 == 1



# Parsed testcases at query #39
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
    var_18 = b'# single file'
    var_19 = [var_15]
    var_20 = module_1.find(var_19, var_0, var_1, var_2)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = len(var_1)
    assert var_23 == 0
    var_24 = len(var_2)
    assert var_24 == 1
    var_25 = 'skipme'
    var_26 = '# should be skipped'
    var_27 = module_1.find(var_13, var_0, var_1, var_2)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = len(var_1)
    assert var_30 == 1



# Parsed testcases at query #40
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
    var_7 = '# Python file in subdir'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = '# Python file in skipped dir'
    var_11 = [var_8]
    var_12 = False
    var_13 = module_0.Config()
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 1
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = 'nonexistent'
    var_19 = []
    var_20 = []
    var_21 = len(var_19)
    assert var_21 == 0
    var_22 = len(var_20)
    assert var_22 == 1
    var_23 = []
    var_24 = []
    var_25 = len(var_23)
    assert var_25 == 0
    var_26 = len(var_24)
    assert var_26 == 0
    var_27 = []
    var_28 = []
    var_29 = len(var_27)
    assert var_29 == 1
    var_30 = len(var_28)
    assert var_30 == 1



# Parsed testcases at query #41
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
    var_12 = 'file1.py'
    var_13 = 'file2.py'
    var_14 = "print('file1')"
    var_15 = "print('file2')"
    var_16 = 'file.txt'
    var_17 = 'text file'
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
    var_29 = 'subdir'
    var_30 = 'file.py'
    var_31 = "print('in subdir')"
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_27, var_33, var_34, var_35)
    var_37 = list(var_36)
    var_38 = 'file.py'
    var_39 = "print('in dir')"
    var_40 = b"print('separate')"
    var_41 = module_0.Config()
    var_42 = []
    var_43 = []
    var_44 = module_1.find(var_40, var_41, var_42, var_43)
    var_45 = list(var_44)
    var_46 = set(var_45)



# Parsed testcases at query #42
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'subdir'
    var_1 = '# test'
    var_2 = '# test2'
    var_3 = 'not python'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test.py'
    var_8 = 'test2.py'
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



# Parsed testcases at query #43
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
    var_30 = '# test'
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_26, var_0, var_31, var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = 'test.py'
    var_37 = len(var_31)
    assert var_37 == 1
    var_38 = len(var_32)
    assert var_38 == 0



# Parsed testcases at query #44
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = '# test'
    var_5 = 'subdir'
    var_6 = 'subfile.py'
    var_7 = '# subfile'
    var_8 = 'skipped_dir'
    var_9 = 'skipped.py'
    var_10 = '# skipped'
    var_11 = 'readme.txt'
    var_12 = 'readme'
    var_13 = 'nonexistent.py'



# Parsed testcases at query #45
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = '# test content'
    var_6 = module_1.find(var_4, var_0, var_1, var_2)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = len(var_1)
    assert var_9 == 0
    var_10 = len(var_2)
    assert var_10 == 0
    var_11 = 'test_dir'
    var_12 = '# test content'
    var_13 = '# test content'
    var_14 = '# test content'
    var_15 = [var_11]
    var_16 = module_1.find(var_15, var_0, var_1, var_2)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = len(var_1)
    assert var_19 == 0
    var_20 = len(var_2)
    assert var_20 == 0
    var_21 = 'test_dir/file1.py'
    var_22 = 'test_dir/file2.py'
    var_23 = 'test_dir/ignore.txt'
    var_24 = 'non_existent_path.py'
    var_25 = [var_24]
    var_26 = module_1.find(var_25, var_0, var_1, var_2)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = len(var_1)
    assert var_29 == 0
    var_30 = len(var_2)
    assert var_30 == 1
    var_31 = 'skip_me.py'
    var_32 = [var_31]
    var_33 = '# test content'
    var_34 = module_1.find(var_32, var_0, var_1, var_2)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = len(var_1)
    assert var_37 == 1
    var_38 = len(var_2)
    assert var_38 == 0



# Parsed testcases at query #46
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
    var_14 = '.py'
    var_15 = module_0.Config()
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_15, var_18, var_19)
    var_21 = list(var_20)
    var_22 = 'skip_dir'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = [var_8]
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_25, var_24, var_26, var_27)
    var_29 = list(var_28)
    var_30 = module_0.Config()
    var_31 = [var_1, var_8]
    var_32 = []
    var_33 = []
    var_34 = module_1.find(var_31, var_30, var_32, var_33)
    var_35 = list(var_34)



# Parsed testcases at query #47
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
    var_13 = 'test_dir/test1.py'
    var_14 = 'test_dir/test2.py'
    var_15 = 'test_dir/skip_me.py'
    var_16 = 'non_existent_path'
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = module_1.find(var_17, var_0, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = 'test_file.py'
    var_24 = [var_23]
    var_25 = []
    var_26 = []
    var_27 = '# test file'
    var_28 = module_1.find(var_24, var_0, var_25, var_26)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = module_0.Config()
    var_32 = [var_27]
    var_33 = []
    var_34 = []
    var_35 = 'test_dir/subdir'
    var_36 = '# test file 3'
    var_37 = '# test file 4'
    var_38 = module_1.find(var_32, var_31, var_33, var_34)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = 'test_dir/subdir/test3.py'
    var_42 = 'test_dir/test4.py'



# Parsed testcases at query #48
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
    var_10 = 'file1.py'
    var_11 = 'file2.py'
    var_12 = "print('hello')"
    var_13 = "print('world')"
    var_14 = 'file.txt'
    var_15 = 'not python'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = set(var_5)
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = '/nonexistent/path'
    var_24 = [var_23]
    var_25 = module_1.find(var_24, var_20, var_21, var_22)
    var_26 = list(var_25)
    var_27 = 'subdir'
    var_28 = 'file.py'
    var_29 = "print('hello')"
    var_30 = 'skipped.py'
    var_31 = "print('skipped')"
    var_32 = [var_30]
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = module_1.find(var_23, var_33, var_34, var_35)
    var_37 = list(var_36)
    var_38 = 'target'
    var_39 = 'link'
    var_40 = 'file.py'
    var_41 = "print('hello')"
    var_42 = True
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = module_1.find(var_36, var_43, var_44, var_45)
    var_47 = list(var_46)
    var_48 = False
    var_49 = module_0.Config()
    var_50 = []
    var_51 = []



# Parsed testcases at query #49
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test content'
    var_2 = 'subdir'
    var_3 = 'subfile.py'
    var_4 = '# subfile content'
    var_5 = 'skipped'
    var_6 = 'skipped.py'
    var_7 = '# skipped content'
    var_8 = 'nonexistent.py'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = len(var_10)
    assert var_12 == 1
    var_13 = len(var_11)
    assert var_13 == 1



