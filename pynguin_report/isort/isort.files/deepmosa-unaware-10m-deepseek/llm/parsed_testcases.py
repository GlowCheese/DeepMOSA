####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/non/existent/path.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipped'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = '.py'
    var_25 = []
    var_26 = []
    var_27 = module_1.find(var_9, var_14, var_25, var_26)
    var_28 = list(var_27)
    var_29 = len(var_25)
    assert var_29 == 1
    var_30 = 'single.py'
    var_31 = "print('single')"
    var_32 = 'subdir'
    var_33 = 'nested.py'
    var_34 = "print('nested')"
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = [var_9, var_27]
    var_39 = module_1.find(var_38, var_35, var_36, var_37)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = 'original'
    var_43 = 'file.py'
    var_44 = "print('original')"
    var_45 = 'link'
    var_46 = False
    var_47 = '.py'
    var_48 = []
    var_49 = []
    var_50 = [var_38]
    var_51 = 'script.py'
    var_52 = "print('python')"
    var_53 = 'data.txt'
    var_54 = 'data'
    var_55 = 'notes.md'
    var_56 = '# Notes'
    var_57 = module_0.Config()
    var_58 = []
    var_59 = []
    var_60 = [var_41]
    var_61 = module_1.find(var_60, var_57, var_58, var_59)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = module_0.Config()
    var_65 = []
    var_66 = []
    var_67 = module_1.find(var_51, var_64, var_65, var_66)
    var_68 = list(var_67)



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipped_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'normal_dir'
    var_18 = "print('normal')"
    var_19 = 'skipped'
    var_20 = '.py'
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/non/existent/path'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'test.py'
    var_32 = var_27 / var_31
    var_33 = "print('hello')"
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = str(var_32)
    var_38 = '/invalid/path'
    var_39 = [var_37, var_38]
    var_40 = module_1.find(var_39, var_34, var_35, var_36)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = str(var_32)
    var_44 = 'script.py'
    var_45 = "print('py')"
    var_46 = 'script.js'
    var_47 = "console.log('js')"
    var_48 = False
    var_49 = '.py'
    var_50 = []
    var_51 = []
    var_52 = module_1.find(var_10, var_34, var_50, var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'source'
    var_56 = 'file.py'
    var_57 = "print('source')"
    var_58 = 'link'
    var_59 = False
    var_60 = '.py'
    var_61 = []
    var_62 = []
    var_63 = [var_10]
    var_64 = module_1.find(var_63, var_34, var_61, var_62)
    var_65 = list(var_64)
    var_66 = len(var_65)
    var_67 = module_0.Config()
    var_68 = []
    var_69 = []
    var_70 = []
    var_71 = module_1.find(var_70, var_67, var_68, var_69)
    var_72 = list(var_71)



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip.py'
    var_15 = "print('skip')"
    var_16 = 'include.py'
    var_17 = "print('include')"
    var_18 = 'skip_dir'
    var_19 = 'inside.py'
    var_20 = "print('inside')"
    var_21 = 'skip'
    var_22 = '.py'
    var_23 = []
    var_24 = []
    var_25 = len(var_23)
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = '/non/existent/path.py'
    var_30 = [var_29]
    var_31 = module_1.find(var_30, var_26, var_27, var_28)
    var_32 = list(var_31)
    var_33 = 'test.py'
    var_34 = "print('test')"
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = '/invalid/path.py'
    var_39 = list(var_17)
    var_40 = 'normal.py'
    var_41 = "print('normal')"
    var_42 = False
    var_43 = '.py'
    var_44 = []
    var_45 = []
    var_46 = [var_17]
    var_47 = module_1.find(var_46, var_35, var_44, var_45)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = 'root.py'
    var_51 = "print('root')"
    var_52 = 'subdir'
    var_53 = 'sub.py'
    var_54 = "print('sub')"
    var_55 = module_0.Config()
    var_56 = []
    var_57 = []
    var_58 = [var_19]
    var_59 = module_1.find(var_58, var_55, var_56, var_57)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = any(var_21)
    var_63 = module_0.Config()
    var_64 = []
    var_65 = []
    var_66 = []
    var_67 = module_1.find(var_66, var_63, var_64, var_65)
    var_68 = list(var_67)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = 'file1.py'
    var_7 = 'file2.py'
    var_8 = 'file3.txt'
    var_9 = []
    var_10 = []
    var_11 = True
    var_12 = 'skipme'
    var_13 = var_5 / var_12
    var_14 = 'file.py'
    var_15 = var_13 / var_14
    var_16 = 'normal'
    var_17 = []
    var_18 = []
    var_19 = str(var_15)
    var_20 = []
    var_21 = []
    var_22 = '/non/existent/path'
    var_23 = [var_22]
    var_24 = False
    var_25 = True
    var_26 = 'single.py'
    var_27 = var_12 / var_26
    var_28 = 'subdir'
    var_29 = var_7 / var_28
    var_30 = 'nested.py'
    var_31 = var_29 / var_30
    var_32 = []
    var_33 = []
    var_34 = str(var_27)
    var_35 = str(var_29)
    var_36 = [var_34, var_35]
    var_37 = list(var_19)
    var_38 = str(var_27)
    var_39 = str(var_31)
    var_40 = False
    var_41 = 'script.py'
    var_42 = var_25 / var_41
    var_43 = 'doc.txt'
    var_44 = var_26 / var_43
    var_45 = []
    var_46 = []
    var_47 = list(var_30)
    var_48 = str(var_42)
    var_49 = str(var_44)
    var_50 = False
    var_51 = True
    var_52 = 'dir1'
    var_53 = var_41 / var_52
    var_54 = 'file.py'
    var_55 = var_53 / var_54
    var_56 = []
    var_57 = []
    var_58 = list(var_30)
    var_59 = str(var_55)
    var_60 = True
    var_61 = 'skip_this.py'
    var_62 = var_51 / var_61
    var_63 = 'normal.py'
    var_64 = var_52 / var_63
    var_65 = []
    var_66 = []
    var_67 = list(var_30)
    var_68 = str(var_64)
    var_69 = str(var_62)
    var_70 = any(var_35)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = '.py'
    var_15 = 'skip.py'
    var_16 = "print('skip')"
    var_17 = 'include.py'
    var_18 = "print('include')"
    var_19 = '.py'
    var_20 = 'skip'
    var_21 = []
    var_22 = []
    var_23 = [var_10]
    var_24 = module_1.find(var_23, var_11, var_21, var_22)
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = len(var_21)
    assert var_27 == 1
    var_28 = module_0.Config()
    var_29 = []
    var_30 = []
    var_31 = '/nonexistent/file.py'
    var_32 = [var_31]
    var_33 = module_1.find(var_32, var_28, var_29, var_30)
    var_34 = list(var_33)
    var_35 = 'test.py'
    var_36 = "print('hello')"
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = '/nonexistent/file.py'
    var_41 = list(var_18)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = 'file1.py'
    var_44 = "print('1')"
    var_45 = 'subdir'
    var_46 = 'file2.py'
    var_47 = "print('2')"
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = [var_10]
    var_52 = module_1.find(var_51, var_48, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = '.py'
    var_56 = 'file1.py'
    var_57 = "print('1')"
    var_58 = '.py'
    var_59 = False
    var_60 = []
    var_61 = []
    var_62 = [var_46]
    var_63 = module_1.find(var_62, var_48, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = module_0.Config()
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = module_1.find(var_69, var_66, var_67, var_68)
    var_71 = list(var_70)



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'file3.txt'
    var_10 = 'not python'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipped.py'
    var_15 = 'normal.py'
    var_16 = "print('skipped')"
    var_17 = "print('normal')"
    var_18 = '.py'
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_9, var_11, var_19, var_20)
    var_22 = list(var_21)
    var_23 = [var_10]
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/non/existent/path.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'single.py'
    var_32 = 'subdir'
    var_33 = 'nested.py'
    var_34 = "print('single')"
    var_35 = "print('nested')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = [var_9, var_21]
    var_40 = module_1.find(var_39, var_36, var_37, var_38)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 'source'
    var_44 = 'link'
    var_45 = 'test.py'
    var_46 = "print('test')"
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = [var_44]
    var_52 = module_1.find(var_51, var_48, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 0
    var_56 = var_53[var_55]
    var_57 = 'test.py'
    var_58 = 'normal'
    var_59 = 'skipped'
    var_60 = 'file1.py'
    var_61 = "print('1')"
    var_62 = 'file2.py'
    var_63 = "print('2')"
    var_64 = '.py'
    var_65 = []
    var_66 = []
    var_67 = len(var_53)
    assert var_67 == 1
    var_68 = module_0.Config()
    var_69 = []
    var_70 = []
    var_71 = []
    var_72 = module_1.find(var_71, var_68, var_69, var_70)
    var_73 = list(var_72)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'file1.py'
    var_5 = var_0 / var_4
    var_6 = 'file2.py'
    var_7 = 'subdir'
    var_8 = 'file3.py'
    var_9 = []
    var_10 = []
    var_11 = str(var_5)
    var_12 = 'skip'
    var_13 = 'file1.py'
    var_14 = var_0 / var_13
    var_15 = 'skip_file.py'
    var_16 = 'skip_dir'
    var_17 = var_12 / var_16
    var_18 = 'file3.py'
    var_19 = var_17 / var_18
    var_20 = []
    var_21 = []
    var_22 = str(var_14)
    var_23 = [var_22]
    var_24 = sorted(var_20)
    var_25 = str(var_17)
    var_26 = []
    var_27 = []
    var_28 = '/non/existent/path.py'
    var_29 = [var_28]
    var_30 = list(var_18)
    var_31 = 'dir_file.py'
    var_32 = var_0 / var_31
    var_33 = 'subdir'
    var_34 = var_15 / var_33
    var_35 = 'sub_file.py'
    var_36 = var_34 / var_35
    var_37 = []
    var_38 = []
    var_39 = str(var_32)
    var_40 = str(var_34)
    var_41 = [var_39, var_40]
    var_42 = str(var_32)
    var_43 = str(var_36)
    var_44 = [var_42, var_43]
    var_45 = sorted(var_44)
    var_46 = sorted(var_30)
    var_47 = []
    var_48 = []
    var_49 = 'target'
    var_50 = var_0 / var_49
    var_51 = 'target.py'
    var_52 = var_50 / var_51
    var_53 = 'link'
    var_54 = []
    var_55 = []
    var_56 = [var_39]
    var_57 = list(var_41)
    var_58 = str(var_52)
    var_59 = [var_58]
    var_60 = []
    var_61 = []
    var_62 = []
    var_63 = list(var_46)
    var_64 = '.py'
    var_65 = 'file1.txt'
    var_66 = var_0 / var_65
    var_67 = 'file2.md'
    var_68 = []
    var_69 = []
    var_70 = list(var_28)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = ''
    var_7 = 'file2.py'
    var_8 = 'not_python.txt'
    var_9 = module_0.Config()
    var_10 = []
    var_11 = []
    var_12 = 'skip_me'
    var_13 = 'file.py'
    var_14 = ''
    var_15 = [var_12]
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = [var_8]
    var_20 = module_1.find(var_19, var_16, var_17, var_18)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_17)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_26)
    assert var_32 == 1
    var_33 = 'single.py'
    var_34 = ''
    var_35 = 'subdir'
    var_36 = 'another.py'
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = '/nonexistent'
    var_41 = [var_19, var_20, var_40]
    var_42 = module_1.find(var_41, var_37, var_38, var_39)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = len(var_39)
    assert var_45 == 1
    var_46 = 'source'
    var_47 = 'file.py'
    var_48 = ''
    var_49 = 'link'
    var_50 = True
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = [var_20]
    var_55 = module_1.find(var_54, var_51, var_52, var_53)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = 'skip.py'
    var_59 = ''
    var_60 = 'keep.py'
    var_61 = [var_58]
    var_62 = module_0.Config()
    var_63 = []
    var_64 = []
    var_65 = [var_49]
    var_66 = module_1.find(var_65, var_62, var_63, var_64)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 1
    var_69 = len(var_63)
    assert var_69 == 1



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'test1.py'
    var_5 = 'test2.py'
    var_6 = 'not_python.txt'
    var_7 = ''
    var_8 = ''
    var_9 = ''
    var_10 = []
    var_11 = []
    var_12 = 'skipme'
    var_13 = 'test1.py'
    var_14 = 'test2.py'
    var_15 = ''
    var_16 = ''
    var_17 = len(var_10)
    assert var_17 == 1
    var_18 = []
    var_19 = []
    var_20 = '/nonexistent/path/file.py'
    var_21 = [var_20]
    var_22 = list(var_13)
    var_23 = []
    var_24 = []
    var_25 = 'normal.py'
    var_26 = 'skipfile.py'
    var_27 = ''
    var_28 = ''
    var_29 = list(var_14)
    var_30 = len(var_23)
    assert var_30 == 1
    var_31 = []
    var_32 = []
    var_33 = list(var_26)
    var_34 = []
    var_35 = []
    var_36 = 'test1.py'
    var_37 = 'test2.py'
    var_38 = ''
    var_39 = ''
    var_40 = '/nonexistent'
    var_41 = list(var_13)
    var_42 = sorted(var_41)
    var_43 = sorted(var_30)
    var_44 = []
    var_45 = []
    var_46 = 'source'
    var_47 = 'link'
    var_48 = 'test1.py'
    var_49 = ''
    var_50 = list(var_17)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = []
    var_53 = []
    var_54 = 'source'
    var_55 = 'link'
    var_56 = 'test1.py'
    var_57 = ''
    var_58 = []
    var_59 = 'test1.py'
    var_60 = [var_59]
    var_61 = list(var_43)
    var_62 = len(var_61)
    assert var_62 == 1



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip.py'
    var_15 = "print('skip')"
    var_16 = 'include.py'
    var_17 = "print('include')"
    var_18 = '.py'
    var_19 = 'skip'
    var_20 = []
    var_21 = []
    var_22 = [var_10]
    var_23 = module_1.find(var_22, var_11, var_20, var_21)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = len(var_20)
    assert var_26 == 1
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = '/non/existent/path.py'
    var_31 = [var_30]
    var_32 = module_1.find(var_31, var_27, var_28, var_29)
    var_33 = list(var_32)
    var_34 = 'valid.py'
    var_35 = var_30 / var_34
    var_36 = "print('valid')"
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = str(var_35)
    var_41 = '/invalid/path.py'
    var_42 = [var_40, var_41]
    var_43 = module_1.find(var_42, var_37, var_38, var_39)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = str(var_35)
    var_47 = 'subdir'
    var_48 = 'nested.py'
    var_49 = "print('nested')"
    var_50 = 'root.py'
    var_51 = "print('root')"
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = [var_23]
    var_56 = module_1.find(var_55, var_52, var_53, var_54)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 2
    var_59 = 'link'
    var_60 = 'target'
    var_61 = 'linked.py'
    var_62 = "print('linked')"
    var_63 = True
    var_64 = module_0.Config()
    var_65 = []
    var_66 = []
    var_67 = [var_36]
    var_68 = module_1.find(var_67, var_64, var_65, var_66)
    var_69 = list(var_68)
    var_70 = 'linked.py'
    var_71 = any(var_49)
    var_72 = module_0.Config()
    var_73 = []
    var_74 = []
    var_75 = []
    var_76 = module_1.find(var_75, var_72, var_73, var_74)
    var_77 = list(var_76)
    var_78 = 'script.py'
    var_79 = "print('python')"
    var_80 = 'data.txt'
    var_81 = 'text data'
    var_82 = 'notes.md'
    var_83 = '# Markdown'
    var_84 = module_0.Config()
    var_85 = []
    var_86 = []
    var_87 = [var_23]
    var_88 = module_1.find(var_87, var_84, var_85, var_86)
    var_89 = list(var_88)
    var_90 = len(var_89)
    assert var_90 == 1



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/non/existent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = len(var_7)
    assert var_13 == 1
    var_14 = 'file1.py'
    var_15 = "print('1')"
    var_16 = 'file2.py'
    var_17 = "print('2')"
    var_18 = 'not_python.txt'
    var_19 = 'text'
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = len(var_11)
    assert var_23 == 2
    var_24 = 'skipme'
    var_25 = 'file.py'
    var_26 = "print('skipped')"
    var_27 = [var_24]
    var_28 = module_0.Config()
    var_29 = []
    var_30 = []
    var_31 = module_1.find(var_18, var_28, var_29, var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = len(var_29)
    assert var_34 == 1
    var_35 = 'subdir'
    var_36 = 'root.py'
    var_37 = "print('root')"
    var_38 = 'nested.py'
    var_39 = "print('nested')"
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = [var_33]
    var_44 = module_1.find(var_43, var_40, var_41, var_42)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = 'file1.py'
    var_48 = "print('1')"
    var_49 = 'subdir'
    var_50 = 'file2.py'
    var_51 = "print('2')"
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = str(var_33)
    var_56 = [var_55, var_44]
    var_57 = module_1.find(var_56, var_52, var_53, var_54)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = any(var_23)
    var_61 = 'real'
    var_62 = 'link'
    var_63 = 'test.py'
    var_64 = "print('test')"
    var_65 = True
    var_66 = module_0.Config()
    var_67 = []
    var_68 = []
    var_69 = [var_50]
    var_70 = module_1.find(var_69, var_66, var_67, var_68)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 1
    var_73 = 'file1.py'
    var_74 = "print('1')"
    var_75 = 'file2.py'
    var_76 = "print('2')"
    var_77 = module_0.Config()
    var_78 = []
    var_79 = []
    var_80 = [var_70]
    var_81 = module_1.find(var_80, var_77, var_78, var_79)
    var_82 = next(var_81)
    var_83 = list(var_81)
    var_84 = len(var_83)
    assert var_84 == 1



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/non/existent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = "print('1')"
    var_14 = 'file2.py'
    var_15 = "print('2')"
    var_16 = 'subdir'
    var_17 = 'file3.py'
    var_18 = "print('3')"
    var_19 = 'not_python.txt'
    var_20 = 'text'
    var_21 = module_0.Config()
    var_22 = []
    var_23 = []
    var_24 = len(var_11)
    assert var_24 == 3
    var_25 = [Path(p) for p in var_11]
    var_26 = 'skip_dir'
    var_27 = 'file.py'
    var_28 = "print('skipped')"
    var_29 = 'keep_dir'
    var_30 = "print('kept')"
    var_31 = [var_26]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = [var_19]
    var_36 = module_1.find(var_35, var_32, var_33, var_34)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = len(var_33)
    assert var_39 == 1
    var_40 = 'single.py'
    var_41 = "print('single')"
    var_42 = 'subdir'
    var_43 = 'nested.py'
    var_44 = "print('nested')"
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = len(var_37)
    assert var_48 == 2
    var_49 = [Path(p) for p in var_37]
    var_50 = 'source'
    var_51 = 'source_file.py'
    var_52 = "print('source')"
    var_53 = 'link'
    var_54 = True
    var_55 = module_0.Config()
    var_56 = []
    var_57 = []
    var_58 = len(var_37)
    var_59 = any(var_30)
    var_60 = module_0.Config()
    var_61 = []
    var_62 = []
    var_63 = module_1.find(var_50, var_60, var_61, var_62)
    var_64 = list(var_63)
    var_65 = 'skip_me.py'
    var_66 = "print('skip')"
    var_67 = 'keep_me.py'
    var_68 = "print('keep')"
    var_69 = [var_65]
    var_70 = module_0.Config()
    var_71 = []
    var_72 = []
    var_73 = module_1.find(var_54, var_70, var_71, var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = len(var_71)
    assert var_76 == 1



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipped_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'normal_dir'
    var_18 = "print('normal')"
    var_19 = '.py'
    var_20 = 'skipped'
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/non/existent/path'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'single.py'
    var_32 = "print('single')"
    var_33 = 'subdir'
    var_34 = 'nested.py'
    var_35 = "print('nested')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = len(var_30)
    assert var_39 == 2
    var_40 = any(var_20)
    var_41 = 'source'
    var_42 = 'file.py'
    var_43 = "print('source')"
    var_44 = 'link'
    var_45 = '.py'
    var_46 = False
    var_47 = []
    var_48 = []
    var_49 = [var_10]
    var_50 = module_1.find(var_49, var_36, var_47, var_48)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = 'script.py'
    var_54 = "print('python')"
    var_55 = 'data.txt'
    var_56 = 'text data'
    var_57 = 'notes.md'
    var_58 = '# Markdown'
    var_59 = '.py'
    var_60 = False
    var_61 = []
    var_62 = []
    var_63 = list(var_19)
    var_64 = len(var_63)
    assert var_64 == 1



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = 'not_python.txt'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = '.py'
    var_11 = 'skip_me'
    var_12 = 'file.py'
    var_13 = lambda p: var_11 in str(p)
    var_14 = True
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 1
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = '/non/existent/path.py'
    var_22 = [var_21]
    var_23 = module_1.find(var_22, var_18, var_19, var_20)
    var_24 = list(var_23)
    var_25 = 'valid.py'
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = '/invalid/path.py'
    var_30 = [var_23, var_29]
    var_31 = module_1.find(var_30, var_26, var_27, var_28)
    var_32 = list(var_31)
    var_33 = [var_14]
    var_34 = 'subdir'
    var_35 = 'file.py'
    var_36 = False
    var_37 = True
    var_38 = []
    var_39 = []
    var_40 = [var_33]
    var_41 = module_1.find(var_40, var_26, var_38, var_39)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = 'python.py'
    var_45 = 'text.txt'
    var_46 = False
    var_47 = '.py'
    var_48 = lambda f: f.endswith(var_47)
    var_49 = []
    var_50 = []
    var_51 = [var_41]
    var_52 = module_1.find(var_51, var_26, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = var_53[var_46]
    var_56 = 'dir1'
    var_57 = 'file.py'
    var_58 = False
    var_59 = True
    var_60 = []
    var_61 = []
    var_62 = [var_47, var_48]
    var_63 = module_1.find(var_62, var_26, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = list(var_1)
    var_5 = '.py'
    var_6 = lambda x: x.endswith(var_5)
    var_7 = []
    var_8 = []
    var_9 = 'file1.py'
    var_10 = 'file2.py'
    var_11 = 'file3.txt'
    var_12 = ''
    var_13 = set(var_4)
    var_14 = 'skipme'
    var_15 = lambda x: var_14 in str(x)
    var_16 = []
    var_17 = []
    var_18 = 'skipme'
    var_19 = 'file.py'
    var_20 = ''
    var_21 = list(var_14)
    var_22 = any(var_15)
    var_23 = []
    var_24 = []
    var_25 = '/nonexistent/path/file.py'
    var_26 = [var_25]
    var_27 = lambda x: x.endswith(var_19)
    var_28 = []
    var_29 = []
    var_30 = 'dir_file.py'
    var_31 = ''
    var_32 = 'subdir'
    var_33 = 'subdir_file.py'
    var_34 = ''
    var_35 = list(var_15)
    var_36 = []
    var_37 = []
    var_38 = 'target'
    var_39 = 'link'
    var_40 = 'file.py'
    var_41 = ''
    var_42 = []
    var_43 = 'file.py'
    var_44 = [var_43]
    assert var_44 == 1
    var_45 = list(var_15)
    var_46 = True
    var_47 = []
    var_48 = []
    var_49 = 'dir1'
    var_50 = 'dir2'
    var_51 = 'link_to_dir1'
    var_52 = 'file.py'
    var_53 = ''
    var_54 = [var_53, var_50]
    var_55 = []
    var_56 = []
    var_57 = [var_52]
    var_58 = [var_51]
    var_59 = []
    var_60 = []
    var_61 = [var_52]
    var_62 = list(var_50)
    var_63 = lambda x: x.endswith(var_44)
    var_64 = []
    var_65 = []
    var_66 = list(var_50)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = '.py'
    var_6 = lambda x: x.endswith(var_5)
    var_7 = 'file1.py'
    var_8 = 'file2.py'
    var_9 = 'file3.txt'
    var_10 = []
    var_11 = []
    var_12 = 'skipme'
    var_13 = lambda x: var_12 in str(x)
    var_14 = True
    var_15 = 'normal'
    var_16 = 'file1.py'
    var_17 = 'file2.py'
    var_18 = []
    var_19 = []
    var_20 = len(var_18)
    assert var_20 == 1
    var_21 = []
    var_22 = []
    var_23 = '/nonexistent/file.py'
    var_24 = [var_23]
    var_25 = list(var_16)
    var_26 = False
    var_27 = '.py'
    var_28 = lambda x: x.endswith(var_27)
    var_29 = 'dir1'
    var_30 = 'file1.py'
    var_31 = 'file2.py'
    var_32 = var_24 / var_31
    var_33 = []
    var_34 = []
    var_35 = str(var_32)
    var_36 = sorted(var_25)
    var_37 = str(var_32)
    var_38 = [var_20, var_37]
    var_39 = sorted(var_38)
    var_40 = False
    var_41 = '.py'
    var_42 = lambda x: x.endswith(var_41)
    var_43 = 'file1.py'
    var_44 = 'file2.txt'
    var_45 = 'file3.py'
    var_46 = var_24 / var_45
    var_47 = []
    var_48 = []
    var_49 = sorted(var_25)
    var_50 = str(var_46)
    var_51 = [var_36, var_50]
    var_52 = sorted(var_51)
    var_53 = False
    var_54 = '.py'
    var_55 = lambda x: x.endswith(var_54)
    var_56 = 'subdir'
    var_57 = 'file1.py'
    var_58 = []
    var_59 = []
    var_60 = list(var_17)
    var_61 = 'skip'
    var_62 = lambda x: var_61 in str(x)
    var_63 = '.py'
    var_64 = lambda x: x.endswith(var_63)
    var_65 = 'skip_me.py'
    var_66 = var_56 / var_65
    var_67 = 'normal.py'
    var_68 = var_57 / var_67
    var_69 = []
    var_70 = []
    var_71 = str(var_68)
    var_72 = [var_71]
    var_73 = len(var_69)
    assert var_73 == 1



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/non/existent/path.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipped_dir'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = 'normal_dir'
    var_25 = "print('normal')"
    var_26 = [var_21]
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = len(var_20)
    assert var_30 == 1
    var_31 = len(var_28)
    var_32 = 'single.py'
    var_33 = "print('single')"
    var_34 = 'subdir'
    var_35 = 'nested.py'
    var_36 = "print('nested')"
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = len(var_20)
    assert var_40 == 2
    var_41 = any(var_26)
    var_42 = any(var_30)
    var_43 = 'target'
    var_44 = 'linked_file.py'
    var_45 = "print('linked')"
    var_46 = 'link'
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = module_1.find(var_10, var_48, var_49, var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = module_0.Config()
    var_55 = []
    var_56 = []
    var_57 = []
    var_58 = module_1.find(var_57, var_54, var_55, var_56)
    var_59 = list(var_58)
    var_60 = 'test.txt'
    var_61 = 'not python'
    var_62 = module_0.Config()
    var_63 = []
    var_64 = []
    var_65 = [var_57]
    var_66 = module_1.find(var_65, var_62, var_63, var_64)
    var_67 = list(var_66)
    var_68 = [var_46]



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'keep_dir'
    var_18 = "print('kept')"
    var_19 = '.py'
    var_20 = []
    var_21 = []
    var_22 = len(var_20)
    var_23 = module_0.Config()
    var_24 = []
    var_25 = []
    var_26 = '/non/existent/path.py'
    var_27 = [var_26]
    var_28 = module_1.find(var_27, var_23, var_24, var_25)
    var_29 = list(var_28)
    var_30 = 'test.py'
    var_31 = var_26 / var_30
    var_32 = "print('hello')"
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = str(var_31)
    var_37 = '/invalid/path.py'
    var_38 = [var_36, var_37]
    var_39 = module_1.find(var_38, var_33, var_34, var_35)
    var_40 = list(var_39)
    var_41 = str(var_31)
    var_42 = [var_41]
    var_43 = 'real_dir'
    var_44 = 'file.py'
    var_45 = "print('test')"
    var_46 = False
    var_47 = '.py'
    var_48 = []
    var_49 = []
    var_50 = module_1.find(var_10, var_33, var_48, var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = 'script.py'
    var_54 = "print('py')"
    var_55 = 'script.js'
    var_56 = "console.log('js')"
    var_57 = False
    var_58 = '.py'
    var_59 = []
    var_60 = []
    var_61 = module_1.find(var_10, var_33, var_59, var_60)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'dir1'
    var_65 = 'subdir'
    var_66 = var_39 / var_65
    var_67 = 'nested.py'
    var_68 = var_66 / var_67
    var_69 = "print('nested')"
    var_70 = 'dir2'
    var_71 = 'another.py'
    var_72 = "print('another')"
    var_73 = module_0.Config()
    var_74 = []
    var_75 = []
    var_76 = len(var_62)
    assert var_76 == 2



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipme'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = [var_14]
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_9, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_19)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'single.py'
    var_32 = "print('single')"
    var_33 = 'subdir'
    var_34 = 'nested.py'
    var_35 = "print('nested')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = '/fake'
    var_40 = [var_21, var_23, var_39]
    var_41 = module_1.find(var_40, var_36, var_37, var_38)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'real'
    var_45 = 'file.py'
    var_46 = "print('real')"
    var_47 = 'link'
    var_48 = True
    var_49 = module_0.Config()
    var_50 = []
    var_51 = []
    var_52 = [var_21]
    var_53 = module_1.find(var_52, var_49, var_50, var_51)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1
    var_56 = 'script.py'
    var_57 = "print('python')"
    var_58 = 'data.txt'
    var_59 = 'data'
    var_60 = 'notes.md'
    var_61 = '# Notes'
    var_62 = module_0.Config()
    var_63 = []
    var_64 = []
    var_65 = [var_55]
    var_66 = module_1.find(var_65, var_62, var_63, var_64)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 1
    var_69 = module_0.Config()
    var_70 = []
    var_71 = []
    var_72 = module_1.find(var_56, var_69, var_70, var_71)
    var_73 = list(var_72)



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipme'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = [var_14]
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = list(var_9)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_19)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_26)
    assert var_32 == 1
    var_33 = 'test.py'
    var_34 = var_27 / var_33
    var_35 = "print('hello')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = str(var_34)
    var_40 = '/nonexistent/file.py'
    var_41 = [var_39, var_40]
    var_42 = module_1.find(var_41, var_36, var_37, var_38)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = str(var_34)
    var_46 = len(var_38)
    assert var_46 == 1
    var_47 = 'source'
    var_48 = 'link'
    var_49 = True
    var_50 = 'file.py'
    var_51 = "print('linked')"
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = [var_41]
    var_56 = module_1.find(var_55, var_52, var_53, var_54)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = 'a'
    var_60 = 'b'
    var_61 = var_33 / var_60
    var_62 = 'c'
    var_63 = var_61 / var_62
    var_64 = True
    var_65 = 'deep.py'
    var_66 = var_63 / var_65
    var_67 = "print('deep')"
    var_68 = module_0.Config()
    var_69 = []
    var_70 = []
    var_71 = len(var_57)
    assert var_71 == 1
    var_72 = 'skip.py'
    var_73 = "print('skip')"
    var_74 = [var_72]
    var_75 = module_0.Config()
    var_76 = []
    var_77 = []
    var_78 = str(var_34)
    var_79 = [var_78]
    var_80 = module_1.find(var_79, var_75, var_76, var_77)
    var_81 = list(var_80)
    var_82 = len(var_81)
    assert var_82 == 0
    var_83 = len(var_76)
    assert var_83 == 1



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'keep_dir'
    var_18 = "print('kept')"
    var_19 = []
    var_20 = []
    var_21 = len(var_19)
    assert var_21 == 1
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/nonexistent/file.py'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = 'test.py'
    var_31 = var_25 / var_30
    var_32 = "print('hello')"
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = str(var_31)
    var_37 = '/nonexistent/file.py'
    var_38 = [var_36, var_37]
    var_39 = module_1.find(var_38, var_33, var_34, var_35)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = str(var_31)
    var_43 = 'source'
    var_44 = 'link'
    var_45 = 'file.py'
    var_46 = "print('test')"
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = [var_44]
    var_52 = module_1.find(var_51, var_48, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'skip.py'
    var_56 = "print('skip')"
    var_57 = 'keep.py'
    var_58 = "print('keep')"
    var_59 = str(var_41)
    var_60 = [var_59]
    var_61 = module_0.Config()
    var_62 = []
    var_63 = []
    var_64 = len(var_53)
    assert var_64 == 1
    var_65 = len(var_62)
    assert var_65 == 1
    var_66 = module_0.Config()
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = module_1.find(var_69, var_66, var_67, var_68)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 0



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipme'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = [var_14]
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_9, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = len(var_19)
    assert var_24 == 1
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = '/nonexistent/file.py'
    var_29 = [var_28]
    var_30 = module_1.find(var_29, var_25, var_26, var_27)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = 'dir1'
    var_34 = 'file1.py'
    var_35 = "print('1')"
    var_36 = 'file2.py'
    var_37 = "print('2')"
    var_38 = module_0.Config()
    var_39 = []
    var_40 = []
    var_41 = [var_21, var_23]
    var_42 = module_1.find(var_41, var_38, var_39, var_40)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = 'source'
    var_46 = 'file.py'
    var_47 = "print('test')"
    var_48 = 'link'
    var_49 = True
    var_50 = module_0.Config()
    var_51 = []
    var_52 = []
    var_53 = [var_21]
    var_54 = module_1.find(var_53, var_50, var_51, var_52)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = 'test.txt'
    var_58 = 'not python'
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_32]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 0
    var_66 = 'skip_this.py'
    var_67 = "print('skipped')"
    var_68 = [var_66]
    var_69 = module_0.Config()
    var_70 = []
    var_71 = []
    var_72 = [var_62]
    var_73 = module_1.find(var_72, var_69, var_70, var_71)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 0
    var_76 = len(var_70)
    assert var_76 == 1



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipme'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = [var_14]
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_9, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_19)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'single.py'
    var_32 = "print('single')"
    var_33 = 'subdir'
    var_34 = 'nested.py'
    var_35 = "print('nested')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = [var_9, var_21]
    var_40 = module_1.find(var_39, var_36, var_37, var_38)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 'real'
    var_44 = 'file.py'
    var_45 = "print('real')"
    var_46 = 'link'
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = [var_21]
    var_52 = module_1.find(var_51, var_48, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'script.py'
    var_56 = "print('python')"
    var_57 = 'document.txt'
    var_58 = 'not python'
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_47]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'skip_me.py'
    var_67 = "print('skip')"
    var_68 = 'include_me.py'
    var_69 = "print('include')"
    var_70 = [var_66]
    var_71 = module_0.Config()
    var_72 = []
    var_73 = []
    var_74 = module_1.find(var_47, var_71, var_72, var_73)
    var_75 = list(var_74)
    var_76 = len(var_75)
    assert var_76 == 1
    var_77 = len(var_72)
    assert var_77 == 1
    var_78 = module_0.Config()
    var_79 = []
    var_80 = []
    var_81 = module_1.find(var_66, var_78, var_79, var_80)
    var_82 = list(var_81)
    var_83 = 'dir1'
    var_84 = 'dir2'
    var_85 = 'link_to_dir2'
    var_86 = 'link_to_dir1'
    var_87 = True
    var_88 = module_0.Config()
    var_89 = []
    var_90 = []
    var_91 = [var_74]
    var_92 = module_1.find(var_91, var_88, var_89, var_90)
    var_93 = list(var_92)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = 'file1.py'
    var_7 = "print('1')"
    var_8 = 'file2.py'
    var_9 = "print('2')"
    var_10 = 'not_py.txt'
    var_11 = 'text'
    var_12 = False
    var_13 = '.py'
    var_14 = lambda f: f.endswith(var_13)
    var_15 = []
    var_16 = []
    var_17 = 'skip_me'
    var_18 = 'file.py'
    var_19 = "print('skipped')"
    var_20 = 'ok.py'
    var_21 = "print('ok')"
    var_22 = lambda p: var_17 in str(p)
    var_23 = True
    var_24 = []
    var_25 = []
    var_26 = [var_12]
    var_27 = list(var_14)
    var_28 = len(var_24)
    var_29 = False
    var_30 = True
    var_31 = []
    var_32 = []
    var_33 = '/nonexistent/file.py'
    var_34 = [var_33]
    var_35 = list(var_19)
    var_36 = 'test.py'
    var_37 = "print('hello')"
    var_38 = False
    var_39 = True
    var_40 = []
    var_41 = []
    var_42 = '/invalid/path.py'
    var_43 = list(var_21)
    var_44 = 'real'
    var_45 = 'file.py'
    var_46 = "print('real')"
    var_47 = 'link'
    var_48 = False
    var_49 = True
    var_50 = []
    var_51 = []
    var_52 = [var_22]
    var_53 = list(var_12)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'dir1'
    var_56 = 'file1.py'
    var_57 = "print('1')"
    var_58 = 'dir2'
    var_59 = False
    var_60 = True
    var_61 = []
    var_62 = []
    var_63 = [var_22, var_52]
    var_64 = list(var_54)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'script.py'
    var_67 = "print('py')"
    var_68 = 'data.txt'
    var_69 = 'text'
    var_70 = 'notes.md'
    var_71 = '# Notes'
    var_72 = False
    var_73 = '.py'
    var_74 = lambda f: f.endswith(var_73)
    var_75 = []
    var_76 = []
    var_77 = len(var_64)
    assert var_77 == 1
    var_78 = var_64[var_72]



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'test1.py'
    var_6 = "print('test1')"
    var_7 = 'test2.py'
    var_8 = "print('test2')"
    var_9 = 'not_python.txt'
    var_10 = 'not python'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip_me'
    var_15 = 'test.py'
    var_16 = "print('skipped')"
    var_17 = [var_14]
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = module_1.find(var_9, var_18, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_19)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/non/existent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'valid.py'
    var_32 = "print('valid')"
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = '/invalid/path.py'
    var_37 = list(var_17)
    var_38 = 'source'
    var_39 = 'link'
    var_40 = 'test.py'
    var_41 = "print('test')"
    var_42 = True
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = [var_39]
    var_47 = module_1.find(var_46, var_43, var_44, var_45)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = 'subdir'
    var_51 = 'nested'
    var_52 = 'root.py'
    var_53 = "print('root')"
    var_54 = 'sub.py'
    var_55 = "print('sub')"
    var_56 = 'deep.py'
    var_57 = "print('deep')"
    var_58 = 'ignore.txt'
    var_59 = 'ignore me'
    var_60 = module_0.Config()
    var_61 = []
    var_62 = []
    var_63 = 'skip_this.py'
    var_64 = 'keep_this.py'
    var_65 = "print('skip')"
    var_66 = "print('keep')"
    var_67 = [var_63]
    var_68 = module_0.Config()
    var_69 = []
    var_70 = []
    var_71 = module_1.find(var_54, var_68, var_69, var_70)
    var_72 = list(var_71)
    var_73 = [var_55]
    var_74 = len(var_69)
    assert var_74 == 1



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/non/existent/path.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipme'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = [var_21]
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = list(var_9)
    var_29 = len(var_26)
    assert var_29 == 1
    var_30 = 'root.py'
    var_31 = "print('root')"
    var_32 = 'subdir'
    var_33 = 'nested.py'
    var_34 = "print('nested')"
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_10, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = 'file1.py'
    var_42 = var_30 / var_41
    var_43 = "print('1')"
    var_44 = 'file2.py'
    var_45 = var_32 / var_44
    var_46 = "print('2')"
    var_47 = module_0.Config()
    var_48 = []
    var_49 = []
    var_50 = module_1.find(var_34, var_47, var_48, var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = any(var_38)
    var_54 = 'real'
    var_55 = 'file.py'
    var_56 = "print('real')"
    var_57 = 'link'
    var_58 = True
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_50]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    var_66 = 'valid.py'
    var_67 = var_54 / var_66
    var_68 = "print('valid')"
    var_69 = module_0.Config()
    var_70 = []
    var_71 = []
    var_72 = str(var_67)
    var_73 = '/invalid/path.py'
    var_74 = [var_72, var_73]
    var_75 = module_1.find(var_74, var_69, var_70, var_71)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 1



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/non/existent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = len(var_7)
    assert var_13 == 1
    var_14 = 'file1.py'
    var_15 = ''
    var_16 = 'file2.py'
    var_17 = 'not_python.txt'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = len(var_11)
    assert var_21 == 2
    var_22 = 'skipme'
    var_23 = 'file.py'
    var_24 = ''
    var_25 = [var_22]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = [var_17]
    var_30 = module_1.find(var_29, var_26, var_27, var_28)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = len(var_27)
    assert var_33 == 1
    var_34 = 'skip.py'
    var_35 = ''
    var_36 = [var_34]
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = [var_24]
    var_41 = module_1.find(var_40, var_37, var_38, var_39)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 0
    var_44 = len(var_38)
    assert var_44 == 1
    var_45 = 'dir1'
    var_46 = 'file1.py'
    var_47 = ''
    var_48 = 'dir2'
    var_49 = 'file2.py'
    var_50 = 'single.py'
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = len(var_42)
    assert var_54 == 3
    var_55 = 'target'
    var_56 = 'linked.py'
    var_57 = ''
    var_58 = 'link'
    var_59 = False
    var_60 = module_0.Config()
    var_61 = []
    var_62 = []
    var_63 = [var_30]
    var_64 = module_1.find(var_63, var_60, var_61, var_62)
    var_65 = list(var_64)
    var_66 = len(var_65)
    assert var_66 == 1
    var_67 = 'valid.py'
    var_68 = ''
    var_69 = module_0.Config()
    var_70 = []
    var_71 = []
    var_72 = '/invalid/path.py'
    var_73 = [var_36, var_72]
    var_74 = module_1.find(var_73, var_69, var_70, var_71)
    var_75 = list(var_74)
    var_76 = len(var_75)
    assert var_76 == 1
    var_77 = len(var_71)
    assert var_77 == 1
    var_78 = module_0.Config()
    var_79 = []
    var_80 = []
    var_81 = []
    var_82 = module_1.find(var_81, var_78, var_79, var_80)
    var_83 = list(var_82)
    var_84 = len(var_83)
    assert var_84 == 0



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipped_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'normal_dir'
    var_18 = "print('normal')"
    var_19 = []
    var_20 = []
    var_21 = len(var_19)
    assert var_21 == 1
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/non/existent/path.py'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = 'dir1'
    var_30 = 'file1.py'
    var_31 = "print('1')"
    var_32 = 'file2.py'
    var_33 = "print('2')"
    var_34 = module_0.Config()
    var_35 = []
    var_36 = []
    var_37 = len(var_28)
    assert var_37 == 2
    var_38 = 'real_dir'
    var_39 = 'file.py'
    var_40 = "print('test')"
    var_41 = 'link_dir'
    var_42 = 'real_dir'
    var_43 = True
    var_44 = module_0.Config()
    var_45 = []
    var_46 = []
    var_47 = module_1.find(var_40, var_44, var_45, var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    var_50 = 'file1.py'
    var_51 = "print('1')"
    var_52 = 'file2.py'
    var_53 = "print('2')"
    var_54 = [var_40]
    var_55 = module_0.Config()
    var_56 = []
    var_57 = []
    var_58 = module_1.find(var_49, var_55, var_56, var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = len(var_56)
    assert var_61 == 1
    var_62 = module_0.Config()
    var_63 = []
    var_64 = []
    var_65 = []
    var_66 = module_1.find(var_65, var_62, var_63, var_64)
    var_67 = list(var_66)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skip_dir'
    var_15 = 'file.py'
    var_16 = "print('skip')"
    var_17 = 'keep_dir'
    var_18 = "print('keep')"
    var_19 = []
    var_20 = []
    var_21 = len(var_19)
    assert var_21 == 1
    var_22 = module_0.Config()
    var_23 = []
    var_24 = []
    var_25 = '/non/existent/path.py'
    var_26 = [var_25]
    var_27 = module_1.find(var_26, var_22, var_23, var_24)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0
    var_30 = len(var_24)
    assert var_30 == 1
    var_31 = 'test.py'
    var_32 = "print('hello')"
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = '/invalid/path.py'
    var_37 = [var_29, var_36]
    var_38 = module_1.find(var_37, var_33, var_34, var_35)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = len(var_35)
    assert var_41 == 1
    var_42 = 'source'
    var_43 = 'file.py'
    var_44 = "print('source')"
    var_45 = 'link'
    var_46 = True
    var_47 = module_0.Config()
    var_48 = []
    var_49 = []
    var_50 = [var_41]
    var_51 = module_1.find(var_50, var_47, var_48, var_49)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = 'skip.py'
    var_55 = "print('skip')"
    var_56 = 'keep.py'
    var_57 = "print('keep')"
    var_58 = [var_45]
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_46]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = len(var_60)
    assert var_66 == 1
    var_67 = module_0.Config()
    var_68 = []
    var_69 = []
    var_70 = []
    var_71 = module_1.find(var_70, var_67, var_68, var_69)
    var_72 = list(var_71)
    var_73 = len(var_72)
    assert var_73 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'nonexistent.py'
    var_7 = [var_6]
    var_8 = 'file1.py'
    var_9 = 'file2.py'
    var_10 = 'not_python.txt'
    var_11 = '.py'
    var_12 = []
    var_13 = []
    var_14 = 'skipped_dir'
    var_15 = 'file.py'
    var_16 = []
    var_17 = []
    var_18 = len(var_16)
    assert var_18 == 1
    var_19 = 'skipped.py'
    var_20 = []
    var_21 = []
    var_22 = list(var_7)
    var_23 = len(var_20)
    assert var_23 == 1
    var_24 = 'dir1'
    var_25 = 'file1.py'
    var_26 = 'dir2'
    var_27 = 'file2.py'
    var_28 = '.py'
    var_29 = []
    var_30 = []
    var_31 = 'real_dir'
    var_32 = 'file.py'
    var_33 = 'link_dir'
    var_34 = '.py'
    var_35 = []
    var_36 = []
    var_37 = []
    var_38 = 'file.py'
    var_39 = [var_38]
    var_40 = (var_2, var_37, var_39)
    var_41 = [var_23]
    var_42 = list(var_33)
    var_43 = True
    var_44 = []
    var_45 = []
    var_46 = 'subdir'
    var_47 = [var_46]
    var_48 = []
    var_49 = list(var_48)
    var_50 = []
    var_51 = []
    var_52 = 'valid.py'
    var_53 = 'invalid.py'
    var_54 = [var_52, var_53]
    var_55 = list(var_43)
    var_56 = []
    var_57 = []
    var_58 = []



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/nonexistent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = "print('1')"
    var_14 = 'file2.py'
    var_15 = "print('2')"
    var_16 = 'not_python.txt'
    var_17 = 'text'
    var_18 = module_0.Config()
    var_19 = []
    var_20 = []
    var_21 = len(var_11)
    assert var_21 == 2
    var_22 = 'skipme'
    var_23 = 'file.py'
    var_24 = "print('skipped')"
    var_25 = [var_22]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = module_1.find(var_16, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_27)
    assert var_31 == 1
    var_32 = 'single.py'
    var_33 = "print('single')"
    var_34 = 'subdir'
    var_35 = 'subfile.py'
    var_36 = "print('sub')"
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = [var_29, var_31]
    var_41 = module_1.find(var_40, var_37, var_38, var_39)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'real'
    var_45 = 'file.py'
    var_46 = "print('real')"
    var_47 = True
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = [var_36]
    var_52 = module_1.find(var_51, var_48, var_49, var_50)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'script.py'
    var_56 = "print('py')"
    var_57 = 'data.txt'
    var_58 = 'text'
    var_59 = 'notes.md'
    var_60 = '# Markdown'
    var_61 = module_0.Config()
    var_62 = []
    var_63 = []
    var_64 = [var_41]
    var_65 = module_1.find(var_64, var_61, var_62, var_63)
    var_66 = list(var_65)
    var_67 = len(var_66)
    assert var_67 == 1
    var_68 = module_0.Config()
    var_69 = []
    var_70 = []
    var_71 = module_1.find(var_55, var_68, var_69, var_70)
    var_72 = list(var_71)
    var_73 = 'skip.py'
    var_74 = "print('skip')"
    var_75 = [var_73]
    var_76 = module_0.Config()
    var_77 = []
    var_78 = []
    var_79 = [var_57]
    var_80 = module_1.find(var_79, var_76, var_77, var_78)
    var_81 = list(var_80)
    var_82 = len(var_77)
    assert var_82 == 1



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/non/existent/path.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipped'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = [var_21]
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = module_1.find(var_9, var_25, var_26, var_27)
    var_29 = list(var_28)
    var_30 = len(var_26)
    assert var_30 == 1
    var_31 = 'source'
    var_32 = 'file.py'
    var_33 = "print('source')"
    var_34 = 'link'
    var_35 = True
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = [var_28]
    var_40 = module_1.find(var_39, var_36, var_37, var_38)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = 'file1.py'
    var_44 = "print('1')"
    var_45 = 'subdir'
    var_46 = 'file2.py'
    var_47 = "print('2')"
    var_48 = module_0.Config()
    var_49 = []
    var_50 = []
    var_51 = str(var_39)
    var_52 = [var_51, var_42]
    var_53 = module_1.find(var_52, var_48, var_49, var_50)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = 'script.py'
    var_57 = "print('python')"
    var_58 = 'data.txt'
    var_59 = 'text data'
    var_60 = module_0.Config()
    var_61 = []
    var_62 = []
    var_63 = [var_47]
    var_64 = module_1.find(var_63, var_60, var_61, var_62)
    var_65 = list(var_64)
    var_66 = len(var_65)
    assert var_66 == 1
    var_67 = 'skipped.py'
    var_68 = "print('skipped')"
    var_69 = 'regular.py'
    var_70 = "print('regular')"
    var_71 = [var_67]
    var_72 = module_0.Config()
    var_73 = []
    var_74 = []
    var_75 = [var_35]
    var_76 = module_1.find(var_75, var_72, var_73, var_74)
    var_77 = list(var_76)
    var_78 = len(var_77)
    assert var_78 == 1
    var_79 = len(var_73)
    assert var_79 == 1



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'file3.txt'
    var_10 = 'not python'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'skipped_dir'
    var_15 = 'file.py'
    var_16 = "print('skipped')"
    var_17 = 'normal_dir'
    var_18 = "print('normal')"
    var_19 = 'skipped'
    var_20 = '.py'
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/non/existent/path.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'test.py'
    var_32 = "print('hello')"
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = '/non/existent/path.py'
    var_37 = list(var_16)
    var_38 = len(var_37)
    var_39 = 'subdir'
    var_40 = 'file.py'
    var_41 = "print('test')"
    var_42 = False
    var_43 = '.py'
    var_44 = []
    var_45 = []
    var_46 = [var_17]
    var_47 = list(var_10)
    var_48 = list(var_18)
    var_49 = len(var_47)
    var_50 = len(var_48)
    var_51 = module_0.Config()
    var_52 = []
    var_53 = []
    var_54 = []
    var_55 = module_1.find(var_54, var_51, var_52, var_53)
    var_56 = list(var_55)
    var_57 = 'test.txt'
    var_58 = 'not python'
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_55]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/non/existent/file.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = 'skipme'
    var_23 = 'file.py'
    var_24 = "print('skipped')"
    var_25 = [var_22]
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = list(var_9)
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = len(var_27)
    assert var_31 == 1
    var_32 = 'single.py'
    var_33 = "print('single')"
    var_34 = 'subdir'
    var_35 = 'another.py'
    var_36 = "print('another')"
    var_37 = module_0.Config()
    var_38 = []
    var_39 = []
    var_40 = [var_30, var_31]
    var_41 = module_1.find(var_40, var_37, var_38, var_39)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'source'
    var_45 = 'file.py'
    var_46 = "print('source')"
    var_47 = 'link'
    var_48 = True
    var_49 = module_0.Config()
    var_50 = []
    var_51 = []
    var_52 = [var_30]
    var_53 = module_1.find(var_52, var_49, var_50, var_51)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1
    var_56 = 'skip_this.py'
    var_57 = "print('skip')"
    var_58 = 'keep_this.py'
    var_59 = "print('keep')"
    var_60 = [var_56]
    var_61 = module_0.Config()
    var_62 = []
    var_63 = []
    var_64 = module_1.find(var_36, var_61, var_62, var_63)
    var_65 = list(var_64)
    var_66 = len(var_65)
    assert var_66 == 1
    var_67 = len(var_62)
    assert var_67 == 1



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = 'not_python.txt'
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = 'skipped.py'
    var_11 = 'included.py'
    var_12 = '.py'
    var_13 = 'skipped'
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 1
    var_17 = module_0.Config()
    var_18 = []
    var_19 = []
    var_20 = '/non/existent/path.py'
    var_21 = [var_20]
    var_22 = module_1.find(var_21, var_17, var_18, var_19)
    var_23 = list(var_22)
    var_24 = 'valid.py'
    var_25 = var_20 / var_24
    var_26 = module_0.Config()
    var_27 = []
    var_28 = []
    var_29 = str(var_25)
    var_30 = '/invalid/path.py'
    var_31 = [var_29, var_30]
    var_32 = module_1.find(var_31, var_26, var_27, var_28)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = str(var_25)
    var_36 = 'normal.py'
    var_37 = '.py'
    var_38 = False
    var_39 = []
    var_40 = []
    var_41 = []
    var_42 = 'normal.py'
    var_43 = [var_42]
    var_44 = (var_36, var_41, var_43)
    var_45 = [var_31]
    var_46 = module_1.find(var_45, var_26, var_39, var_40)
    var_47 = list(var_46)
    var_48 = True
    var_49 = 'skipped_dir'
    var_50 = 'file.py'
    var_51 = 'normal.py'
    var_52 = '.py'
    var_53 = []
    var_54 = []
    var_55 = [var_48]
    var_56 = module_1.find(var_55, var_26, var_53, var_54)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = len(var_53)
    assert var_59 == 1
    var_60 = 'dir1'
    var_61 = 'file.py'
    var_62 = '.py'
    var_63 = False
    var_64 = []
    var_65 = []
    var_66 = set()
    var_67 = [var_60, var_41]
    var_68 = module_1.find(var_67, var_26, var_64, var_65)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = 'script.py'
    var_72 = 'data.txt'
    var_73 = 'notes.md'
    var_74 = '.py'
    var_75 = False
    var_76 = []
    var_77 = []
    var_78 = [var_56]
    var_79 = module_1.find(var_78, var_26, var_76, var_77)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 1
    var_82 = module_0.Config()
    var_83 = []
    var_84 = []
    var_85 = [var_71]
    var_86 = module_1.find(var_85, var_82, var_83, var_84)
    var_87 = list(var_86)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = 'file1.py'
    var_7 = "print('1')"
    var_8 = 'file2.py'
    var_9 = "print('2')"
    var_10 = 'not_py.txt'
    var_11 = 'text'
    var_12 = False
    var_13 = '.py'
    var_14 = lambda f: f.endswith(var_13)
    var_15 = []
    var_16 = []
    var_17 = 'skipdir'
    var_18 = 'file.py'
    var_19 = "print('skipped')"
    var_20 = 'keepdir'
    var_21 = "print('kept')"
    var_22 = True
    var_23 = []
    var_24 = []
    var_25 = [var_12]
    var_26 = list(var_14)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = len(var_23)
    var_29 = False
    var_30 = True
    var_31 = []
    var_32 = []
    var_33 = '/nonexistent/file.py'
    var_34 = [var_33]
    var_35 = list(var_19)
    var_36 = 'single.py'
    var_37 = "print('single')"
    var_38 = 'subdir'
    var_39 = 'nested.py'
    var_40 = "print('nested')"
    var_41 = False
    var_42 = True
    var_43 = []
    var_44 = []
    var_45 = [var_11, var_22]
    var_46 = list(var_25)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = 'real'
    var_49 = 'file.py'
    var_50 = "print('real')"
    var_51 = 'link'
    var_52 = False
    var_53 = True
    var_54 = []
    var_55 = []
    var_56 = [var_11]
    var_57 = list(var_45)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = 'file.py'
    var_60 = "print('python')"
    var_61 = 'file.txt'
    var_62 = 'text'
    var_63 = False
    var_64 = '.py'
    var_65 = lambda f: f.endswith(var_64)
    var_66 = []
    var_67 = []
    var_68 = [var_56]
    var_69 = list(var_58)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = var_69[var_63]



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = 'file1.py'
    var_15 = "print('1')"
    var_16 = 'skipme'
    var_17 = 'file2.py'
    var_18 = "print('2')"
    var_19 = [var_16]
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = len(var_21)
    assert var_23 == 1
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = len(var_26)
    assert var_32 == 1
    var_33 = 'test.py'
    var_34 = var_27 / var_33
    var_35 = "print('hello')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = str(var_34)
    var_40 = '/nonexistent/file.py'
    var_41 = [var_39, var_40]
    var_42 = module_1.find(var_41, var_36, var_37, var_38)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = str(var_34)
    var_46 = len(var_38)
    assert var_46 == 1
    var_47 = 'source'
    var_48 = 'file.py'
    var_49 = "print('test')"
    var_50 = 'link'
    var_51 = True
    var_52 = module_0.Config()
    var_53 = []
    var_54 = []
    var_55 = [var_45]
    var_56 = module_1.find(var_55, var_52, var_53, var_54)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = 'test.txt'
    var_60 = var_47 / var_59
    var_61 = 'text content'
    var_62 = module_0.Config()
    var_63 = []
    var_64 = []
    var_65 = str(var_60)
    var_66 = [var_65]
    var_67 = module_1.find(var_66, var_62, var_63, var_64)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 0
    var_70 = module_0.Config()
    var_71 = []
    var_72 = []
    var_73 = module_1.find(var_47, var_70, var_71, var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 0
    var_76 = 'root.py'
    var_77 = "print('root')"
    var_78 = 'sub1'
    var_79 = 'file1.py'
    var_80 = "print('1')"
    var_81 = 'sub2'
    var_82 = 'file2.py'
    var_83 = "print('2')"
    var_84 = module_0.Config()
    var_85 = []
    var_86 = []
    var_87 = len(var_74)
    assert var_87 == 3



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/nonexistent/file.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipped'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = [var_21]
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = list(var_9)
    var_29 = len(var_26)
    assert var_29 == 1
    var_30 = 'root.py'
    var_31 = "print('root')"
    var_32 = 'subdir'
    var_33 = 'nested.py'
    var_34 = "print('nested')"
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = module_1.find(var_10, var_35, var_36, var_37)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = 'valid.py'
    var_42 = "print('valid')"
    var_43 = module_0.Config()
    var_44 = []
    var_45 = []
    var_46 = str(var_32)
    var_47 = '/invalid/path.py'
    var_48 = [var_46, var_47]
    var_49 = module_1.find(var_48, var_43, var_44, var_45)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'real'
    var_53 = 'file.py'
    var_54 = "print('real')"
    var_55 = 'link'
    var_56 = True
    var_57 = module_0.Config()
    var_58 = []
    var_59 = []
    var_60 = [var_51]
    var_61 = module_1.find(var_60, var_57, var_58, var_59)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'script.py'
    var_65 = "print('python')"
    var_66 = 'data.txt'
    var_67 = 'text data'
    var_68 = 'notes.md'
    var_69 = '# Markdown'
    var_70 = module_0.Config()
    var_71 = []
    var_72 = []
    var_73 = module_1.find(var_63, var_70, var_71, var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = '.py'
    var_15 = 'skipme'
    var_16 = 'test.py'
    var_17 = "print('skipped')"
    var_18 = 'ok.py'
    var_19 = "print('ok')"
    var_20 = [var_15]
    var_21 = module_0.Config()
    var_22 = []
    var_23 = []
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = '/nonexistent/file.py'
    var_28 = [var_27]
    var_29 = module_1.find(var_28, var_24, var_25, var_26)
    var_30 = list(var_29)
    var_31 = 'file1.py'
    var_32 = "print('1')"
    var_33 = 'subdir'
    var_34 = 'file2.py'
    var_35 = "print('2')"
    var_36 = module_0.Config()
    var_37 = []
    var_38 = []
    var_39 = '/nonexistent'
    var_40 = len(var_30)
    assert var_40 == 2
    var_41 = 'real'
    var_42 = 'test.py'
    var_43 = "print('test')"
    var_44 = 'link'
    var_45 = True
    var_46 = module_0.Config()
    var_47 = []
    var_48 = []
    var_49 = module_1.find(var_39, var_46, var_47, var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 0
    var_53 = var_50[var_52]
    var_54 = 'subdir'
    var_55 = 'test.py'
    var_56 = "print('test')"
    var_57 = 'selflink'
    var_58 = True
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = module_1.find(var_39, var_59, var_60, var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = 'script.py'
    var_66 = "print('py')"
    var_67 = 'script.js'
    var_68 = "console.log('js')"
    var_69 = 'script.txt'
    var_70 = 'text'
    var_71 = module_0.Config()
    var_72 = []
    var_73 = []
    var_74 = [var_64]
    var_75 = module_1.find(var_74, var_71, var_72, var_73)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 1
    var_78 = 0
    var_79 = var_76[var_78]
    var_80 = '.py'
    var_81 = 'skip_this.py'
    var_82 = "print('skip')"
    var_83 = 'keep_this.py'
    var_84 = "print('keep')"
    var_85 = [var_81]
    var_86 = module_0.Config()
    var_87 = []
    var_88 = []
    var_89 = module_1.find(var_69, var_86, var_87, var_88)
    var_90 = list(var_89)
    var_91 = [var_70]
    var_92 = [var_64]



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'non_existent.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_3, var_4, var_5)
    var_9 = list(var_8)
    var_10 = 'file1.py'
    var_11 = 'file2.py'
    var_12 = 'file3.txt'
    var_13 = var_7 / var_12
    var_14 = 'subdir'
    var_15 = 'subfile.py'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = 'skip_me'
    var_20 = 'skipped.py'
    var_21 = 'normal'
    var_22 = 'normal.py'
    var_23 = True
    var_24 = []
    var_25 = []
    var_26 = list(var_14)
    var_27 = 'dir_file.py'
    var_28 = 'subdir'
    var_29 = 'subfile.py'
    var_30 = module_0.Config()
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_23, var_30, var_31, var_32)
    var_34 = list(var_33)
    var_35 = sorted(var_34)
    var_36 = 'script.py'
    var_37 = 'notes.txt'
    var_38 = False
    var_39 = []
    var_40 = []
    var_41 = module_1.find(var_12, var_30, var_39, var_40)
    var_42 = list(var_41)
    var_43 = 'main'
    var_44 = 'main.py'
    var_45 = 'link'
    var_46 = True
    var_47 = False
    var_48 = []
    var_49 = []
    var_50 = [var_23]
    var_51 = list(var_34)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = module_0.Config()
    var_54 = []
    var_55 = []
    var_56 = []
    var_57 = module_1.find(var_56, var_53, var_54, var_55)
    var_58 = list(var_57)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/nonexistent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = "print('1')"
    var_14 = 'file2.py'
    var_15 = "print('2')"
    var_16 = 'subdir'
    var_17 = 'file3.py'
    var_18 = "print('3')"
    var_19 = 'not_python.txt'
    var_20 = 'text'
    var_21 = module_0.Config()
    var_22 = []
    var_23 = []
    var_24 = len(var_11)
    assert var_24 == 3
    var_25 = [Path(p) for p in var_11]
    var_26 = 'skipped_dir'
    var_27 = 'file.py'
    var_28 = "print('skipped')"
    var_29 = 'normal_dir'
    var_30 = "print('normal')"
    var_31 = [var_26]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = [var_19]
    var_36 = module_1.find(var_35, var_32, var_33, var_34)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = len(var_33)
    var_40 = 'single.py'
    var_41 = "print('single')"
    var_42 = 'mydir'
    var_43 = 'inside.py'
    var_44 = "print('inside')"
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = len(var_37)
    assert var_48 == 2
    var_49 = [Path(p) for p in var_37]
    var_50 = 'real'
    var_51 = 'file.py'
    var_52 = "print('real')"
    var_53 = 'link'
    var_54 = True
    var_55 = module_0.Config()
    var_56 = []
    var_57 = []
    var_58 = len(var_37)
    var_59 = False
    var_60 = module_0.Config()
    var_61 = []
    var_62 = []
    var_63 = [var_18]
    var_64 = module_1.find(var_63, var_60, var_61, var_62)
    var_65 = list(var_64)
    var_66 = module_0.Config()
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = module_1.find(var_69, var_66, var_67, var_68)
    var_71 = list(var_70)
    var_72 = 'skipped.py'
    var_73 = "print('skipped')"
    var_74 = 'normal.py'
    var_75 = "print('normal')"
    var_76 = [var_72]
    var_77 = module_0.Config()
    var_78 = []
    var_79 = []
    var_80 = module_1.find(var_54, var_77, var_78, var_79)
    var_81 = list(var_80)
    var_82 = len(var_81)
    assert var_82 == 1
    var_83 = len(var_78)
    assert var_83 == 1



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/nonexistent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = 'file1.py'
    var_13 = ''
    var_14 = 'file2.py'
    var_15 = 'not_python.txt'
    var_16 = module_0.Config()
    var_17 = []
    var_18 = []
    var_19 = len(var_11)
    assert var_19 == 2
    var_20 = 'skipme'
    var_21 = 'file.py'
    var_22 = ''
    var_23 = [var_20]
    var_24 = module_0.Config()
    var_25 = []
    var_26 = []
    var_27 = [var_15]
    var_28 = module_1.find(var_27, var_24, var_25, var_26)
    var_29 = list(var_28)
    var_30 = len(var_25)
    assert var_30 == 1
    var_31 = 'subdir'
    var_32 = 'root.py'
    var_33 = ''
    var_34 = 'nested.py'
    var_35 = module_0.Config()
    var_36 = []
    var_37 = []
    var_38 = [var_28]
    var_39 = module_1.find(var_38, var_35, var_36, var_37)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = any(var_19)
    var_43 = 'single.py'
    var_44 = ''
    var_45 = 'mydir'
    var_46 = 'inside.py'
    var_47 = module_0.Config()
    var_48 = []
    var_49 = []
    var_50 = [var_27, var_28]
    var_51 = module_1.find(var_50, var_47, var_48, var_49)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = any(var_19)
    var_55 = 'real'
    var_56 = 'file.py'
    var_57 = ''
    var_58 = True
    var_59 = module_0.Config()
    var_60 = []
    var_61 = []
    var_62 = [var_15]
    var_63 = module_1.find(var_62, var_59, var_60, var_61)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = "print('1')"
    var_7 = 'file2.py'
    var_8 = "print('2')"
    var_9 = 'not_python.txt'
    var_10 = 'text'
    var_11 = module_0.Config()
    var_12 = []
    var_13 = []
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = '/nonexistent/file.py'
    var_18 = [var_17]
    var_19 = module_1.find(var_18, var_14, var_15, var_16)
    var_20 = list(var_19)
    var_21 = 'skipme'
    var_22 = 'file.py'
    var_23 = "print('skipped')"
    var_24 = [var_21]
    var_25 = module_0.Config()
    var_26 = []
    var_27 = []
    var_28 = list(var_9)
    var_29 = len(var_26)
    assert var_29 == 1
    var_30 = 'skip_me.py'
    var_31 = "print('skip')"
    var_32 = [var_30]
    var_33 = module_0.Config()
    var_34 = []
    var_35 = []
    var_36 = [var_23]
    var_37 = module_1.find(var_36, var_33, var_34, var_35)
    var_38 = list(var_37)
    var_39 = len(var_34)
    assert var_39 == 1
    var_40 = 'dir1'
    var_41 = 'file1.py'
    var_42 = "print('1')"
    var_43 = 'file2.py'
    var_44 = "print('2')"
    var_45 = module_0.Config()
    var_46 = []
    var_47 = []
    var_48 = [var_29, var_10]
    var_49 = module_1.find(var_48, var_45, var_46, var_47)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'source'
    var_53 = 'file.py'
    var_54 = "print('test')"
    var_55 = 'link'
    var_56 = True
    var_57 = module_0.Config()
    var_58 = []
    var_59 = []
    var_60 = [var_29]
    var_61 = module_1.find(var_60, var_57, var_58, var_59)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'script.py'
    var_65 = "print('python')"
    var_66 = 'data.txt'
    var_67 = 'text data'
    var_68 = 'notes.md'
    var_69 = '# Markdown'
    var_70 = module_0.Config()
    var_71 = []
    var_72 = []
    var_73 = module_1.find(var_63, var_70, var_71, var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = module_0.Config()
    var_77 = []
    var_78 = []
    var_79 = module_1.find(var_64, var_76, var_77, var_78)
    var_80 = list(var_79)
    var_81 = module_0.Config()
    var_82 = []
    var_83 = []
    var_84 = '/fake/path1.py'
    var_85 = '/another/fake/path2.py'
    var_86 = [var_84, var_85]
    var_87 = module_1.find(var_86, var_81, var_82, var_83)
    var_88 = list(var_87)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = '/non/existent/file.py'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = len(var_7)
    assert var_13 == 1
    var_14 = 'file1.py'
    var_15 = "print('1')"
    var_16 = 'file2.py'
    var_17 = "print('2')"
    var_18 = 'not_python.txt'
    var_19 = 'text'
    var_20 = module_0.Config()
    var_21 = []
    var_22 = []
    var_23 = len(var_11)
    assert var_23 == 2
    var_24 = 'skipped_dir'
    var_25 = 'file.py'
    var_26 = var_12 / var_25
    var_27 = "print('skipped')"
    var_28 = 'normal_dir'
    var_29 = "print('normal')"
    var_30 = str(var_23)
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = []
    var_34 = []
    var_35 = len(var_11)
    assert var_35 == 1
    var_36 = len(var_33)
    assert var_36 == 1
    var_37 = 'file1.py'
    var_38 = "print('1')"
    var_39 = 'dir1'
    var_40 = 'file2.py'
    var_41 = "print('2')"
    var_42 = module_0.Config()
    var_43 = []
    var_44 = []
    var_45 = len(var_11)
    assert var_45 == 2
    var_46 = any(var_30)
    var_47 = 'script.py'
    var_48 = "print('python')"
    var_49 = 'data.txt'
    var_50 = 'text'
    var_51 = 'notes.md'
    var_52 = '# markdown'
    var_53 = module_0.Config()
    var_54 = []
    var_55 = []
    var_56 = module_1.find(var_45, var_53, var_54, var_55)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = 'skipped.py'
    var_60 = "print('skipped')"
    var_61 = 'normal.py'
    var_62 = "print('normal')"
    var_63 = [var_50]
    var_64 = module_0.Config()
    var_65 = []
    var_66 = []
    var_67 = [var_51]
    var_68 = module_1.find(var_67, var_64, var_65, var_66)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = len(var_65)
    assert var_71 == 1
    var_72 = module_0.Config()
    var_73 = []
    var_74 = []
    var_75 = module_1.find(var_59, var_72, var_73, var_74)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 0
    var_78 = 'source'
    var_79 = 'file.py'
    var_80 = "print('source')"
    var_81 = 'link'
    var_82 = True
    var_83 = module_0.Config()
    var_84 = []
    var_85 = []
    var_86 = [var_67]
    var_87 = module_1.find(var_86, var_83, var_84, var_85)
    var_88 = list(var_87)
    var_89 = len(var_88)
    var_90 = any(var_45)



