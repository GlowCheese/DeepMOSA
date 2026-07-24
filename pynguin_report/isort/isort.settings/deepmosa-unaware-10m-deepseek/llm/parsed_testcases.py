####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = 'file1.py'
    var_2 = 'file2.txt'
    var_3 = [var_1, var_2]
    var_4 = "[tool.isort]\nprofile = 'black'"
    var_5 = 'pyproject.toml'
    var_6 = []
    var_7 = 'pyproject.toml'
    var_8 = [var_7]
    var_9 = 'subdir1'
    var_10 = 'subdir2'
    var_11 = '.isort.cfg'
    var_12 = 'pyproject.toml'
    var_13 = 'setup.cfg'
    var_14 = len(var_9)
    assert var_14 == 3
    var_15 = '.isort.cfg'
    var_16 = []
    var_17 = '.isort.cfg'
    var_18 = [var_17]
    var_19 = '.isort.cfg'
    var_20 = 'pyproject.toml'
    var_21 = 'setup.cfg'
    var_22 = [var_17, var_3, var_12]
    var_23 = []
    var_24 = '.isort.cfg'
    var_25 = 'pyproject.toml'
    var_26 = 'setup.cfg'
    var_27 = [var_24, var_25, var_26]
    var_28 = len(var_23)
    assert var_28 == 1
    var_29 = 'pyproject.toml'
    var_30 = []
    var_31 = 'pyproject.toml'
    var_32 = [var_31]
    var_33 = 'level1'
    var_34 = 'level2'
    var_35 = '.isort.cfg'
    var_36 = 'pyproject.toml'
    var_37 = len(var_33)
    assert var_37 == 2



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = {}
    var_4 = module_0.Config()
    var_5 = 'skip_dir'
    var_6 = {var_5}
    assert var_6 is True
    var_7 = module_0.Config()
    var_8 = var_3 / var_5
    var_9 = 'test.py'
    var_10 = var_8 / var_9
    var_11 = var_7.is_skipped(var_10)
    assert var_11 is True
    var_12 = '*.pyc'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = '/test/*.py'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = 'test'
    var_19 = var_5 / var_18
    var_20 = 'file.py'
    var_21 = var_19 / var_20
    var_22 = var_17.is_skipped(var_21)
    assert var_22 is True
    var_23 = module_0.Config()
    var_24 = '/non/existent/file.py'
    var_25 = 'skip_me'
    var_26 = {var_25}
    var_27 = module_0.Config()
    var_28 = var_3 / var_25
    var_29 = var_27.is_skipped(var_28)
    assert var_29 is True
    var_30 = 'project'
    var_31 = var_25 / var_30
    var_32 = str(var_31)
    var_33 = module_0.Config()
    var_34 = 'test.py'
    var_35 = var_31 / var_34
    var_36 = var_33.is_skipped(var_35)
    assert var_36 is False
    var_37 = '.git'
    var_38 = var_25 / var_37
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'config'
    var_42 = var_38 / var_41
    var_43 = var_40.is_skipped(var_38)
    assert var_43 is True
    var_44 = module_0.Config()
    var_45 = 'real.py'
    var_46 = var_25 / var_45
    var_47 = 'link.py'
    var_48 = var_39 / var_47
    var_49 = var_44.is_skipped(var_48)
    assert var_49 is False
    var_50 = 'skip1.py'
    var_51 = {var_50}
    var_52 = 'skip2.py'
    var_53 = {var_52}
    var_54 = module_0.Config()
    var_55 = 'skip1.py'
    var_56 = var_25 / var_55
    var_57 = 'skip2.py'
    var_58 = var_39 / var_57
    var_59 = 'normal.py'
    var_60 = var_43 / var_59
    var_61 = var_54.is_skipped(var_56)
    assert var_61 is True
    var_62 = var_54.is_skipped(var_58)
    assert var_62 is True
    var_63 = var_54.is_skipped(var_60)
    assert var_63 is False
    var_64 = '*.tmp'
    var_65 = {var_64}
    var_66 = '*.bak'
    var_67 = {var_66}
    var_68 = module_0.Config()
    var_69 = 'file.tmp'
    var_70 = var_25 / var_69
    var_71 = 'file.bak'
    var_72 = var_39 / var_71
    var_73 = 'file.py'
    var_74 = var_43 / var_73
    var_75 = var_68.is_skipped(var_70)
    assert var_75 is True
    var_76 = var_68.is_skipped(var_72)
    assert var_76 is True
    var_77 = var_68.is_skipped(var_74)
    assert var_77 is False



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'skip_dir'
    var_7 = {var_6}
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = var_3 / var_6
    var_10 = 'test.py'
    var_11 = var_9 / var_10
    var_12 = var_8.is_skipped(var_11)
    assert var_12 is True
    var_13 = '*.pyc'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = '**/test/*.py'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = 'test'
    var_20 = var_6 / var_19
    var_21 = 'file.py'
    var_22 = var_20 / var_21
    var_23 = var_18.is_skipped(var_22)
    assert var_23 is True
    var_24 = module_0.Config()
    var_25 = '/non/existent/file.py'
    var_26 = 'skip_me'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = var_3 / var_26
    var_30 = var_28.is_skipped(var_29)
    assert var_30 is True
    var_31 = 'git'
    var_32 = 'init'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = 'untracked.py'
    var_36 = var_23 / var_35
    var_37 = var_28.is_skipped(var_36)
    assert var_37 is True
    var_38 = 'test.py'
    var_39 = var_31 / var_38
    var_40 = False
    var_41 = module_0.Config()
    var_42 = var_41.is_skipped(var_39)
    assert var_42 is False
    var_43 = 'real.py'
    assert var_43 is True
    var_44 = var_31 / var_43
    var_45 = 'link.py'
    var_46 = var_40 / var_45
    var_47 = module_0.Config()
    var_48 = var_47.is_skipped(var_46)
    assert var_48 is False
    var_49 = 'base.py'
    var_50 = {var_49}
    var_51 = 'extra.py'
    var_52 = {var_51}
    var_53 = module_0.Config()
    var_54 = 'extra.py'
    var_55 = var_43 / var_54
    var_56 = var_53.is_skipped(var_55)
    assert var_56 is True
    var_57 = {var_56}
    var_58 = '*.pyo'
    var_59 = {var_58}
    var_60 = module_0.Config()
    var_61 = 'folder\\file.py'
    var_62 = {var_61}
    var_63 = module_0.Config()
    var_64 = 'folder'
    var_65 = var_31 / var_64
    var_66 = 'file.py'
    var_67 = var_65 / var_66
    var_68 = var_63.is_skipped(var_67)
    assert var_68 is True
    var_69 = '.git'
    var_70 = var_31 / var_69
    var_71 = True
    var_72 = module_0.Config()
    var_73 = var_72.is_skipped(var_70)
    assert var_73 is True
    var_74 = 'git'
    var_75 = 'init'
    var_76 = [var_74, var_75]
    var_77 = True
    var_78 = 'tracked.py'
    var_79 = var_68 / var_78
    var_80 = 'add'
    var_81 = [var_74, var_80, var_78]
    var_82 = 'commit'
    var_83 = '-m'
    var_84 = 'test'
    var_85 = [var_74, var_82, var_83, var_84]
    var_86 = var_72.is_skipped(var_79)
    assert var_86 is False



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyx'
    var_3 = [var_1, var_2]
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyx'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'txt'
    var_9 = 'md'
    var_10 = [var_8, var_9]
    var_11 = 'test.txt'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.md'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = [var_1]
    var_16 = '#!/usr/bin/env python\n'
    assert var_16 is True
    var_17 = 'import os\n'
    assert var_17 is False
    var_18 = 'non_existent.py'
    var_19 = var_0.is_supported_filetype(var_18)
    assert var_19 is False
    var_20 = None
    var_21 = var_0.is_supported_filetype(var_20)
    assert var_21 is False
    var_22 = 'Just some text\n'
    assert var_22 is False



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'skip_dir'
    var_7 = {var_6}
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = var_3 / var_6
    var_10 = 'test.py'
    var_11 = var_9 / var_10
    var_12 = var_8.is_skipped(var_11)
    assert var_12 is True
    var_13 = '*.pyc'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = '/test/*.py'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = 'test'
    var_20 = var_6 / var_19
    var_21 = 'file.py'
    var_22 = var_20 / var_21
    var_23 = var_18.is_skipped(var_22)
    assert var_23 is True
    var_24 = module_0.Config()
    var_25 = '/non/existent/file.py'
    var_26 = 'skip_me'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = var_3 / var_26
    var_30 = var_28.is_skipped(var_29)
    assert var_30 is True
    var_31 = 'relative/path.py'
    var_32 = {var_31}
    var_33 = 'relative'
    assert var_33 is True
    var_34 = var_3 / var_33
    var_35 = 'path.py'
    var_36 = var_34 / var_35
    var_37 = True
    var_38 = var_28.is_skipped(var_36)
    assert var_38 is True
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'test.py'
    assert var_41 is True
    var_42 = var_32 / var_41
    var_43 = var_40.is_skipped(var_42)
    var_44 = module_0.Config()
    var_45 = 'real.py'
    var_46 = var_39 / var_45
    var_47 = 'link.py'
    var_48 = var_33 / var_47
    var_49 = var_44.is_skipped(var_48)
    assert var_49 is False
    var_50 = 'skip1.py'
    var_51 = {var_50}
    var_52 = 'skip2.py'
    var_53 = {var_52}
    var_54 = module_0.Config()
    var_55 = {var_47}
    var_56 = '*.pyo'
    var_57 = {var_56}
    var_58 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_dir'
    var_10 = var_1 / var_9
    var_11 = 'test.py'
    var_12 = var_10 / var_11
    var_13 = var_6.is_skipped(var_12)
    assert var_13 is True
    assert var_13 is False
    var_14 = module_0.Config()
    var_15 = '*.pyc'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = module_0.Config()
    var_19 = '/mock/git/folder'
    var_20 = module_0.Config()
    var_21 = [var_7]
    var_22 = 'test_dir'
    var_23 = var_1 / var_22
    var_24 = var_20.is_skipped(var_23)
    assert var_24 is True
    var_25 = module_0.Config()
    var_26 = '/non/existent/file.py'
    var_27 = module_0.Config()
    var_28 = 'subdir/file.py'
    var_29 = [var_28]
    var_30 = '/base/dir/subdir/file.py'
    var_31 = module_0.Config()
    var_32 = '/test/*.py'
    var_33 = [var_32]



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_dir'
    var_7 = {var_6}
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = 'file.py'
    assert var_9 is True
    var_10 = var_3 / var_9
    var_11 = True
    var_12 = var_8.is_skipped(var_10)
    assert var_12 is True
    var_13 = '*.pyc'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'skip_dir'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = module_0.Config()
    var_20 = '/non/existent/path'
    var_21 = 'relative/path.py'
    var_22 = {var_21}
    var_23 = 'relative'
    assert var_23 is True
    var_24 = 'path.py'
    var_25 = var_9 / var_24
    var_26 = True
    var_27 = var_19.is_skipped(var_25)
    assert var_27 is True
    var_28 = 'base.py'
    var_29 = {var_28}
    var_30 = 'extended.py'
    var_31 = {var_30}
    var_32 = module_0.Config()
    var_33 = 'extended.py'
    assert var_33 is True
    var_34 = '*.log'
    var_35 = {var_34}
    var_36 = '*.tmp'
    var_37 = {var_36}
    var_38 = module_0.Config()
    var_39 = module_0.Config()
    var_40 = 'real.py'
    var_41 = var_21 / var_40
    var_42 = 'link.py'
    var_43 = var_23 / var_42
    var_44 = var_39.is_skipped(var_43)
    assert var_44 is False
    var_45 = True
    var_46 = module_0.Config()
    var_47 = 'ignored.py'
    var_48 = var_44 / var_47
    var_49 = var_46.is_skipped(var_48)
    assert var_49 is True
    var_50 = True
    var_51 = module_0.Config()
    var_52 = '.git'
    var_53 = var_44 / var_52
    var_54 = var_51.is_skipped(var_53)
    assert var_54 is True



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    assert var_4 is True
    var_5 = module_0.Config()
    var_6 = 'skip_dir'
    var_7 = {var_6}
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = var_3 / var_6
    var_10 = 'test.py'
    var_11 = var_9 / var_10
    var_12 = var_8.is_skipped(var_11)
    assert var_12 is True
    var_13 = '*.txt'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = {var_7}
    var_17 = module_0.Config()
    var_18 = module_0.Config()
    var_19 = '/non/existent/path.py'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = 'test.py'
    var_23 = var_7 / var_22
    var_24 = var_21.is_skipped(var_23)
    assert var_24 is True
    var_25 = 'extended.py'
    var_26 = {var_25}
    var_27 = module_0.Config()
    var_28 = 'extended.py'
    var_29 = var_7 / var_28
    var_30 = var_27.is_skipped(var_29)
    assert var_30 is True
    var_31 = '*.md'
    var_32 = {var_31}
    var_33 = module_0.Config()
    var_34 = var_33.is_skipped(var_29)
    assert var_34 is True
    var_35 = 'skip_me'
    var_36 = var_20 / var_35
    var_37 = {var_35}
    var_38 = module_0.Config()
    var_39 = var_38.is_skipped(var_36)
    assert var_39 is True
    var_40 = 'relative/path.py'
    var_41 = {var_40}
    var_42 = 'relative'
    var_43 = var_28 / var_42
    var_44 = 'path.py'
    var_45 = var_43 / var_44
    var_46 = True
    var_47 = var_38.is_skipped(var_45)
    assert var_47 is True
    var_48 = 'real.py'
    var_49 = var_40 / var_48
    var_50 = 'link.py'
    var_51 = var_42 / var_50
    var_52 = module_0.Config()
    var_53 = var_52.is_skipped(var_51)
    assert var_53 is False



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = {}
    var_4 = module_0.Config()
    var_5 = 'skip_dir'
    var_6 = {var_5}
    assert var_6 is True
    var_7 = module_0.Config()
    var_8 = var_3 / var_5
    var_9 = 'test.py'
    var_10 = var_8 / var_9
    var_11 = var_7.is_skipped(var_10)
    assert var_11 is True
    var_12 = '*.pyc'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = '**/test/*.py'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = 'test'
    var_19 = var_5 / var_18
    var_20 = 'file.py'
    var_21 = var_19 / var_20
    var_22 = var_17.is_skipped(var_21)
    assert var_22 is True
    var_23 = module_0.Config()
    var_24 = '/non/existent/file.py'
    var_25 = 'skip_dir'
    var_26 = {var_25}
    assert var_26 is True
    var_27 = module_0.Config()
    var_28 = var_3 / var_25
    var_29 = var_27.is_skipped(var_28)
    assert var_29 is True
    var_30 = True
    var_31 = 'not_in_git.py'
    var_32 = var_26 / var_31
    var_33 = var_27.is_skipped(var_32)
    assert var_33 is True
    var_34 = True
    var_35 = module_0.Config()
    var_36 = '.git'
    var_37 = var_26 / var_36
    var_38 = var_35.is_skipped(var_37)
    assert var_38 is True
    var_39 = 'specific.py'
    var_40 = {var_39}
    var_41 = '*.tmp'
    var_42 = {var_41}
    var_43 = module_0.Config()
    var_44 = 'rel/path.py'
    var_45 = {var_44}
    assert var_45 is False
    assert var_45 is True
    var_46 = 'rel'
    var_47 = var_36 / var_46
    var_48 = 'path.py'
    var_49 = var_47 / var_48
    var_50 = var_43.is_skipped(var_49)
    assert var_50 is True
    var_51 = {}
    var_52 = {}
    var_53 = False
    var_54 = module_0.Config()
    var_55 = 'base.py'
    var_56 = {var_55}
    var_57 = 'extra.py'
    var_58 = {var_57}
    var_59 = module_0.Config()
    var_60 = 'extra.py'
    var_61 = var_36 / var_60
    var_62 = var_59.is_skipped(var_61)
    assert var_62 is True
    var_63 = '*.log'
    var_64 = {var_63}
    var_65 = {var_41}
    var_66 = module_0.Config()
    var_67 = 'real_file.py'
    var_68 = {var_67}
    var_69 = module_0.Config()
    var_70 = var_36 / var_67
    var_71 = 'link.py'
    var_72 = var_38 / var_71
    var_73 = var_69.is_skipped(var_72)
    assert var_73 is True



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    assert var_0 is True
    assert var_0 is False
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_dir'
    assert var_6 is True
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = var_3 / var_6
    var_10 = 'file.py'
    var_11 = var_9 / var_10
    var_12 = True
    var_13 = var_8.is_skipped(var_11)
    assert var_13 is True
    var_14 = '*.pyc'
    var_15 = {var_14}
    var_16 = module_0.Config()
    var_17 = module_0.Config()
    var_18 = '/non/existent/path/file.py'
    var_19 = 'git'
    assert var_19 is True
    var_20 = 'init'
    var_21 = [var_19, var_20]
    var_22 = True
    var_23 = 'config'
    var_24 = 'user.email'
    var_25 = 'test@test.com'
    var_26 = [var_19, var_23, var_24, var_25]
    var_27 = 'user.name'
    var_28 = 'Test User'
    var_29 = [var_19, var_23, var_27, var_28]
    var_30 = 'tracked.py'
    var_31 = 'add'
    var_32 = [var_19, var_31, var_30]
    var_33 = 'commit'
    var_34 = '-m'
    var_35 = 'Add tracked'
    var_36 = [var_19, var_33, var_34, var_35]
    var_37 = 'untracked.py'
    var_38 = module_0.Config()
    var_39 = 'file1.py'
    var_40 = {var_39}
    var_41 = 'file2.py'
    var_42 = {var_41}
    var_43 = module_0.Config()
    var_44 = {var_14}
    var_45 = '*.pyo'
    var_46 = {var_45}
    var_47 = module_0.Config()
    var_48 = 'skip_dir'
    var_49 = {var_48}
    var_50 = module_0.Config()
    var_51 = var_21 / var_48
    var_52 = var_50.is_skipped(var_51)
    assert var_52 is True
    var_53 = 'relative/path.py'
    var_54 = {var_53}
    var_55 = 'relative'
    var_56 = var_21 / var_55
    var_57 = 'path.py'
    var_58 = var_56 / var_57
    var_59 = True
    var_60 = var_50.is_skipped(var_58)
    assert var_60 is True



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'skip_this'
    var_2 = {var_0, var_1}
    var_3 = module_0.Config()
    var_4 = 'skip_this/test.py'
    var_5 = 'other.py'
    var_6 = '/project'
    var_7 = {var_0}
    var_8 = module_0.Config()
    var_9 = '/project/test.py'
    var_10 = '/other/test.py'
    var_11 = '*.tmp'
    var_12 = 'temp/*'
    var_13 = {var_11, var_12}
    var_14 = module_0.Config()
    var_15 = 'file.tmp'
    var_16 = 'temp/file.py'
    var_17 = 'file.py'
    var_18 = 'a.py'
    var_19 = {var_18}
    var_20 = 'b.py'
    var_21 = {var_20}
    var_22 = {var_11}
    var_23 = '*.bak'
    var_24 = {var_23}
    var_25 = module_0.Config()
    var_26 = 'file.bak'
    var_27 = 'c.py'
    var_28 = True
    var_29 = module_0.Config()
    var_30 = '/project/tracked.py'
    var_31 = {var_30}
    var_32 = '/project/untracked.py'
    var_33 = '/project/.git'
    var_34 = 'folder/file.py'
    var_35 = {var_34}
    var_36 = module_0.Config()
    var_37 = module_0.Config()
    var_38 = 'nonexistent.txt'
    var_39 = '/absolute/path.py'
    var_40 = {var_39}
    var_41 = module_0.Config()
    var_42 = 'skipdir'
    var_43 = {var_42}
    var_44 = module_0.Config()
    var_45 = 'skipdir/subdir/file.py'
    var_46 = 'otherdir/file.py'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = 'test_file.py'
    var_5 = '*.tmp'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = module_0.Config()
    var_9 = module_0.Config()
    var_10 = 'test.py'
    var_11 = var_2 / var_10
    var_12 = var_9.is_skipped(var_11)
    var_13 = module_0.Config()
    var_14 = 'non_existent_file.py'
    var_15 = False
    var_16 = module_0.Config()
    var_17 = var_16.is_skipped(var_11)
    var_18 = 'file1.py'
    var_19 = {var_18}
    var_20 = 'file2.py'
    var_21 = {var_20}
    var_22 = module_0.Config()
    var_23 = {var_5}
    var_24 = '*.temp'
    var_25 = {var_24}
    var_26 = module_0.Config()
    var_27 = 'test/file.py'
    var_28 = {var_27}
    var_29 = module_0.Config()
    var_30 = 'test'
    var_31 = 'file.py'
    var_32 = True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyx'
    var_3 = [var_1, var_2]
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyx'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'txt'
    var_9 = 'md'
    var_10 = [var_8, var_9]
    var_11 = 'test.txt'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.md'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.py~'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'test'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = 'test.PY'
    var_20 = var_0.is_supported_filetype(var_19)
    assert var_20 is True
    var_21 = 'test.TXT'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'test.utils.py'
    var_24 = var_0.is_supported_filetype(var_23)
    assert var_24 is True
    var_25 = 'test.utils.txt'
    var_26 = var_0.is_supported_filetype(var_25)
    assert var_26 is False
    var_27 = 'test..py'
    var_28 = var_0.is_supported_filetype(var_27)
    assert var_28 is False
    var_29 = ''
    var_30 = var_0.is_supported_filetype(var_29)
    assert var_30 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = 'test_file.py'
    var_5 = 'test_dir'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = 'test.py'
    var_10 = '*.txt'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = '**/test/*.py'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'test'
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = '/non/existent/file.py'
    var_20 = {var_5}
    var_21 = module_0.Config()
    var_22 = 'test_dir'
    var_23 = 'other.py'
    var_24 = {var_23}
    var_25 = {var_10}
    var_26 = module_0.Config()
    var_27 = 'base.py'
    var_28 = {var_27}
    var_29 = 'extended.py'
    var_30 = {var_29}
    var_31 = module_0.Config()
    var_32 = 'base.py'
    var_33 = 'extended.py'
    var_34 = 'other.py'
    var_35 = {var_10}
    var_36 = '*.log'
    var_37 = {var_36}
    var_38 = module_0.Config()
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'test.py'
    var_42 = {var_41}
    var_43 = module_0.Config()
    var_44 = 'real.py'
    var_45 = 'link.py'
    var_46 = var_33 / var_45
    var_47 = var_43.is_skipped(var_46)
    var_48 = '/exact.py'
    var_49 = {var_47}
    var_50 = module_0.Config()
    var_51 = 'exact.py'
    var_52 = var_33 / var_51
    var_53 = var_50.is_skipped(var_52)
    var_54 = 'exact.py.bak'
    var_55 = var_11 / var_54
    var_56 = var_50.is_skipped(var_55)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '\n[tool.isort]\nprofile = "black"\nline_length = 88\n'
    var_2 = 0
    var_3 = 'subdir'
    var_4 = '\n[tool.isort]\nprofile = "hug"\nline_length = 100\n'
    var_5 = 'subsubdir'
    var_6 = 'setup.cfg'
    var_7 = '\n[isort]\nforce_sort_within_sections = true\n'
    var_8 = '.isort.cfg'
    var_9 = 'invalid content'
    var_10 = 'another'
    var_11 = 'tox.ini'
    var_12 = '\n[isort]\nmulti_line_output = 3\n'
    var_13 = '\n[tool.isort]\nprofile = "django"\n'
    var_14 = 0
    var_15 = list(var_0)[var_14]
    var_16 = 'empty'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = 'test_file.py'
    var_5 = 'test_dir'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = var_1 / var_8
    var_10 = 'file.py'
    var_11 = var_9 / var_10
    var_12 = var_7.is_skipped(var_11)
    var_13 = '*.txt'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = {var_13}
    var_17 = module_0.Config()
    var_18 = '/test/*.py'
    var_19 = {var_18}
    var_20 = module_0.Config()
    var_21 = 'test'
    var_22 = var_1 / var_21
    var_23 = 'file.py'
    var_24 = var_22 / var_23
    var_25 = var_20.is_skipped(var_24)
    var_26 = module_0.Config()
    var_27 = '/non/existent/file.py'
    var_28 = {var_22}
    var_29 = module_0.Config()
    var_30 = 'test_dir'
    var_31 = var_1 / var_30
    var_32 = var_29.is_skipped(var_31)
    var_33 = module_0.Config()
    var_34 = 'target.py'
    var_35 = var_1 / var_34
    var_36 = 'link.py'
    var_37 = var_32 / var_36
    var_38 = var_33.is_skipped(var_37)
    var_39 = 'skip1.py'
    var_40 = {var_39}
    var_41 = 'skip2.py'
    var_42 = {var_41}
    var_43 = module_0.Config()
    var_44 = {var_36}
    var_45 = '*.md'
    var_46 = {var_45}
    var_47 = module_0.Config()
    var_48 = True
    var_49 = module_0.Config()
    var_50 = '.git'
    var_51 = var_1 / var_50
    var_52 = var_49.is_skipped(var_51)
    var_53 = '/test/dir'
    var_54 = module_0.Config()
    var_55 = 'test\\file.py'
    var_56 = {var_55}
    var_57 = module_0.Config()
    var_58 = 'test/file.py'
    var_59 = '/absolute/path/file.py'
    var_60 = {var_59}
    var_61 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_dir'
    assert var_6 is False
    var_7 = {var_6}
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = var_3 / var_6
    var_10 = 'file.py'
    var_11 = var_9 / var_10
    var_12 = True
    var_13 = var_8.is_skipped(var_11)
    assert var_13 is True
    var_14 = '*.pyc'
    var_15 = {var_14}
    var_16 = module_0.Config()
    var_17 = True
    var_18 = module_0.Config()
    var_19 = module_0.Config()
    var_20 = '/non/existent/file.py'
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = '.git'
    var_24 = var_6 / var_23
    var_25 = var_22.is_skipped(var_24)
    assert var_25 is True
    var_26 = 'subdir/file.py'
    var_27 = {var_26}
    assert var_27 is True
    var_28 = 'subdir'
    var_29 = var_3 / var_28
    var_30 = 'file.py'
    var_31 = var_29 / var_30
    var_32 = True
    var_33 = var_22.is_skipped(var_31)
    assert var_33 is True
    var_34 = 'file1.py'
    var_35 = {var_34}
    var_36 = 'file2.py'
    var_37 = {var_36}
    var_38 = module_0.Config()
    var_39 = 'file2.py'
    var_40 = var_27 / var_39
    var_41 = var_38.is_skipped(var_40)
    assert var_41 is True
    var_42 = {var_41}
    var_43 = '*.pyo'
    var_44 = {var_43}
    var_45 = module_0.Config()
    var_46 = module_0.Config()
    var_47 = 'real.py'
    var_48 = var_26 / var_47
    var_49 = 'link.py'
    var_50 = var_28 / var_49
    var_51 = var_46.is_skipped(var_50)
    assert var_51 is False
    var_52 = 'test\\file.py'
    var_53 = {var_52}
    var_54 = module_0.Config()
    var_55 = 'test/file.py'
    var_56 = {var_55}
    var_57 = module_0.Config()
    var_58 = 'test'
    var_59 = 'file.py'
    var_60 = var_49 / var_59
    var_61 = var_57.is_skipped(var_60)
    assert var_61 is True



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = {}
    var_4 = module_0.Config()
    var_5 = 'skip_dir'
    var_6 = {var_5}
    assert var_6 is True
    var_7 = module_0.Config()
    var_8 = var_3 / var_5
    var_9 = 'test.py'
    var_10 = var_8 / var_9
    var_11 = var_7.is_skipped(var_10)
    assert var_11 is True
    var_12 = '*.pyc'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = '/test/*.py'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = 'test'
    assert var_18 is False
    assert var_18 is True
    var_19 = var_5 / var_18
    var_20 = 'file.py'
    var_21 = var_19 / var_20
    var_22 = var_17.is_skipped(var_21)
    assert var_22 is True
    var_23 = module_0.Config()
    var_24 = '/non/existent/file.py'
    var_25 = {}
    var_26 = module_0.Config()
    var_27 = module_0.Config()
    var_28 = 'relative/path.py'
    var_29 = {var_28}
    var_30 = 'relative'
    var_31 = var_3 / var_30
    var_32 = 'path.py'
    var_33 = var_31 / var_32
    var_34 = True
    var_35 = var_27.is_skipped(var_33)
    assert var_35 is True
    var_36 = 'git'
    var_37 = 'init'
    var_38 = True
    var_39 = 'untracked.py'
    var_40 = var_32 / var_39
    var_41 = module_0.Config()
    var_42 = var_41.is_skipped(var_40)
    assert var_42 is True
    var_43 = module_0.Config()
    var_44 = 'target.py'
    var_45 = var_36 / var_44
    var_46 = 'link.py'
    var_47 = var_38 / var_46
    var_48 = var_43.is_skipped(var_47)
    assert var_48 is False
    var_49 = 'skip1.py'
    var_50 = {var_49}
    var_51 = 'skip2.py'
    var_52 = {var_51}
    var_53 = module_0.Config()
    var_54 = var_53.is_skipped(var_48)
    assert var_54 is True
    var_55 = var_53.is_skipped(var_46)
    assert var_55 is True
    var_56 = {var_38}
    var_57 = '*.pyo'
    var_58 = {var_57}
    var_59 = module_0.Config()
    var_60 = var_59.is_skipped(var_48)
    assert var_60 is True
    var_61 = var_59.is_skipped(var_46)
    assert var_61 is True
    var_62 = '.git'
    var_63 = var_36 / var_62
    var_64 = True
    var_65 = module_0.Config()
    var_66 = var_65.is_skipped(var_63)
    assert var_66 is True
    var_67 = 'test\\file.py'
    var_68 = {var_67}
    var_69 = module_0.Config()
    var_70 = '\\'
    var_71 = '/'
    var_72 = 2



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 100
    var_2 = 80
    var_3 = module_0._Config(line_length=var_1, wrap_length=var_2)
    var_4 = module_0.Config(config=var_3)
    var_5 = 120
    var_6 = module_0.Config(config=var_3)
    var_7 = 50
    var_8 = 60
    var_9 = module_0._Config(line_length=var_7, wrap_length=var_8)
    var_10 = '[tool.isort]\nline_length = 88\n'
    var_11 = 'black'
    var_12 = module_0.Config()
    var_13 = 'always'
    var_14 = True
    var_15 = False
    var_16 = module_0.Config()
    var_17 = 'value'
    var_18 = module_0.Config()
    var_19 = 4
    var_20 = module_0.Config()
    var_21 = '2'
    var_22 = module_0.Config()
    var_23 = "'\\t'"
    var_24 = module_0.Config()
    var_25 = 'mypy_module'
    var_26 = {var_25}
    var_27 = 'STDLIB'
    var_28 = 'MYPY'
    var_29 = (var_27, var_28)
    var_30 = module_0.Config()
    var_31 = 'Standard Library'
    var_32 = 'End Standard Library'
    var_33 = module_0.Config()
    var_34 = 'src'
    var_35 = var_33.src_paths
    var_36 = 'non_existent_formatter'
    var_37 = module_0.Config()
    var_38 = 'natural'
    var_39 = module_0.Config()
    var_40 = 'native'
    var_41 = module_0.Config()
    var_42 = True
    var_43 = module_0.Config()
    var_44 = 'py310'
    var_45 = module_0._Config(var_44)
    var_46 = module_0.Config(config=var_45)
    var_47 = module_0.Config()
    var_48 = '[isort]\nline_length = 99\n'
    var_49 = 'skip1'
    var_50 = {var_49}
    var_51 = 'skip2'
    var_52 = {var_51}
    var_53 = module_0.Config()
    var_54 = '*.pyc'
    var_55 = {var_54}
    var_56 = '__pycache__'
    var_57 = {var_56}
    var_58 = module_0.Config()



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyx'
    assert var_2 is False
    var_3 = [var_1, var_2]
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyx'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'txt'
    var_9 = 'md'
    var_10 = [var_8, var_9]
    var_11 = 'test.txt'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.md'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.py~'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'test.pyx~'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = '#!/usr/bin/env python\n'
    assert var_19 is True
    var_20 = 'This is not a shebang\n'
    assert var_20 is False
    var_21 = 'non_existent_file.xyz'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = '#!python\n'
    assert var_23 is True



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = 3
    var_3 = 8
    var_4 = 'auto'
    var_5 = module_0._Config(var_4)
    var_6 = 'invalid'
    var_7 = module_0._Config(var_6)
    var_8 = 'all'
    var_9 = module_0._Config(var_8)
    var_10 = frozenset()
    var_11 = module_0._Config(var_6, known_standard_library=var_10)
    var_12 = 'custom_module'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = module_0._Config(var_6, known_standard_library=var_14)
    var_16 = True
    var_17 = module_0._Config(force_alphabetical_sort=var_16)
    var_18 = 79
    var_19 = module_0._Config(line_length=var_18, wrap_length=var_18)
    var_20 = 50
    var_21 = module_0._Config(line_length=var_18, wrap_length=var_20)
    var_22 = 79
    var_23 = 100
    var_24 = module_0._Config(line_length=var_22, wrap_length=var_23)
    var_25 = 88
    var_26 = '  '
    var_27 = 'module1'
    var_28 = [var_27]
    var_29 = frozenset(var_28)
    var_30 = '.git'
    var_31 = [var_30]
    var_32 = frozenset(var_31)
    var_33 = 'myapp'
    var_34 = [var_33]
    var_35 = frozenset(var_34)
    var_36 = module_0._Config(var_22, var_29, var_32, line_length=var_25, known_first_party=var_35, indent=var_26)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '[settings]\nline_length = 88'
    var_3 = '.isort.cfg'
    var_4 = []
    var_5 = '.isort.cfg'
    var_6 = [var_5]
    var_7 = 'line_length'
    var_8 = '88'
    var_9 = len(var_6)
    assert var_9 == 1
    var_10 = 'subdir'
    var_11 = 'subdir'
    var_12 = [var_11]
    var_13 = '.isort.cfg'
    var_14 = [var_13]
    var_15 = []
    var_16 = 'pyproject.toml'
    var_17 = [var_16]
    var_18 = len(var_11)
    assert var_18 == 2
    var_19 = []
    var_20 = '.isort.cfg'
    var_21 = [var_20]
    var_22 = []
    var_23 = '.isort.cfg'
    var_24 = [var_23]



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'migrations'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = 'test.py'
    var_5 = 'test_*.py'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_file.py'
    var_9 = 'other.py'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = 'skip1.py'
    var_14 = {var_13}
    var_15 = 'skip2.py'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = var_4 / var_13
    var_19 = var_17.is_skipped(var_18)
    assert var_19 is True
    var_20 = 'test_*.py'
    var_21 = {var_20}
    var_22 = 'temp_*.py'
    var_23 = {var_22}
    var_24 = module_0.Config()
    var_25 = 'test_file.py'
    var_26 = var_4 / var_25
    var_27 = 'temp_file.py'
    var_28 = var_24.is_skipped(var_26)
    assert var_28 is True
    var_29 = module_0.Config()
    var_30 = 'nonexistent.py'
    var_31 = var_20 / var_30
    var_32 = var_29.is_skipped(var_31)
    assert var_32 is True
    var_33 = True
    var_34 = module_0.Config()
    var_35 = 'test.py'
    var_36 = var_30 / var_35
    var_37 = set()
    var_38 = var_34.is_skipped(var_36)
    assert var_38 is True
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'test.py'
    var_42 = var_30 / var_41
    var_43 = str(var_37)
    var_44 = {var_43}
    var_45 = var_40.is_skipped(var_42)
    assert var_45 is False
    var_46 = True
    var_47 = module_0.Config()
    var_48 = '.git'
    var_49 = var_30 / var_48
    var_50 = var_47.is_skipped(var_49)
    assert var_50 is True
    var_51 = 'test.py'
    var_52 = {var_30}
    var_53 = module_0.Config()
    var_54 = var_23 / var_51
    var_55 = '\\test.py'
    var_56 = module_0.Config()
    var_57 = var_56.is_skipped(var_54)
    assert var_57 is True
    var_58 = 'subdir'
    var_59 = {var_58}
    var_60 = var_57 / var_58
    var_61 = 'nested'
    var_62 = var_60 / var_61
    var_63 = 'file.py'
    var_64 = var_62 / var_63
    var_65 = var_56.is_skipped(var_64)
    assert var_65 is True
    var_66 = module_0.Config()
    var_67 = 'real.py'
    var_68 = var_58 / var_67
    var_69 = 'link.py'
    var_70 = var_23 / var_69
    var_71 = var_66.is_skipped(var_70)
    assert var_71 is False



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Config()
    var_3 = {}
    var_4 = module_0.Config()
    var_5 = 'test_dir'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    assert var_8 is True
    var_9 = var_0 / var_8
    var_10 = 'test.py'
    var_11 = var_9 / var_10
    var_12 = var_7.is_skipped(var_11)
    assert var_12 is True
    var_13 = '*.pyc'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = '**/test/*.py'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = 'test'
    assert var_19 is True
    var_20 = var_0 / var_19
    var_21 = 'test.py'
    var_22 = var_20 / var_21
    var_23 = var_18.is_skipped(var_22)
    assert var_23 is True
    var_24 = module_0.Config()
    var_25 = '/non/existent/path'
    var_26 = True
    var_27 = module_0.Config()
    var_28 = {var_21}
    var_29 = module_0.Config()
    var_30 = 'test_dir'
    assert var_30 is False
    var_31 = var_0 / var_30
    var_32 = var_29.is_skipped(var_31)
    assert var_32 is True
    var_33 = '/some/dir'
    var_34 = 'subdir/file.py'
    var_35 = {var_34}
    var_36 = module_0.Config()
    var_37 = 'subdir'
    var_38 = 'file.py'
    var_39 = var_36.is_skipped(var_22)
    assert var_39 is True
    var_40 = 'C:/test/file.py'
    var_41 = {var_40}
    var_42 = module_0.Config()



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = 'file1.py'
    var_2 = 'file2.txt'
    var_3 = [var_1, var_2]
    var_4 = '[tool.isort]\nprofile = "black"\nline_length = 88'
    var_5 = 'pyproject.toml'
    var_6 = []
    var_7 = 'pyproject.toml'
    var_8 = [var_7]
    var_9 = 'profile'
    var_10 = 'line_length'
    var_11 = 'black'
    var_12 = 88
    var_13 = 'subdir1'
    var_14 = 'subdir2'
    var_15 = '.isort.cfg'
    var_16 = 'setup.cfg'
    var_17 = 'subdir1'
    var_18 = 'subdir2'
    var_19 = [var_17, var_18]
    var_20 = []
    var_21 = []
    var_22 = '.isort.cfg'
    var_23 = [var_22]
    var_24 = []
    var_25 = 'setup.cfg'
    var_26 = [var_25]
    var_27 = 'line_length'
    var_28 = 79
    var_29 = {var_27: var_28}
    var_30 = 'profile'
    var_31 = 'django'
    var_32 = {var_30: var_31}
    var_33 = len(var_22)
    assert var_33 == 2
    var_34 = '.isort.cfg'
    var_35 = []
    var_36 = '.isort.cfg'
    var_37 = [var_36]
    var_38 = 'Parse error'
    var_39 = 'pyproject.toml'
    var_40 = []
    var_41 = 'pyproject.toml'
    var_42 = [var_41]
    var_43 = len(var_40)
    assert var_43 == 0
    var_44 = []
    var_45 = '.isort.cfg'
    var_46 = 'setup.cfg'
    var_47 = 'pyproject.toml'
    var_48 = [var_45, var_46, var_47]
    var_49 = 0



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = b'import os'
    var_4 = 'test_file.py'
    var_5 = 'import sys'
    var_6 = 'skip_dir'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'skip_dir'
    var_10 = var_0 / var_9
    var_11 = 'test.py'
    assert var_11 is True
    var_12 = var_10 / var_11
    var_13 = 'import os'
    var_14 = var_8.is_skipped(var_12)
    assert var_14 is True
    var_15 = '*.pyc'
    assert var_15 is True
    assert var_15 is False
    var_16 = {var_15}
    assert var_16 is False
    var_17 = module_0.Config()
    var_18 = b'bytecode'
    var_19 = module_0.Config()
    var_20 = b'import os'
    var_21 = 'file1.py'
    var_22 = {var_21}
    var_23 = 'file2.py'
    var_24 = {var_23}
    var_25 = module_0.Config()
    var_26 = 'file1.py'
    var_27 = 'file2.py'
    var_28 = 'file3.py'
    var_29 = 'import os'
    var_30 = '*.tmp'
    var_31 = {var_30}
    var_32 = '*.bak'
    var_33 = {var_32}
    var_34 = module_0.Config()
    var_35 = 'test.tmp'
    var_36 = 'test.bak'
    var_37 = 'test.py'
    var_38 = 'content'
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'non_existent.py'
    var_42 = var_38 / var_41
    var_43 = var_40.is_skipped(var_42)
    assert var_43 is True
    var_44 = module_0.Config()
    var_45 = '.git'
    var_46 = var_38 / var_45
    var_47 = var_44.is_skipped(var_46)
    assert var_47 is True
    var_48 = '/some/dir'
    var_49 = 'skipped.py'
    var_50 = {var_49}
    var_51 = module_0.Config()
    var_52 = 'skipped.py'
    var_53 = 'import os'
    var_54 = module_0.Config()
    var_55 = 'real.py'
    var_56 = 'import os'
    assert var_56 is False
    var_57 = 'link.py'
    var_58 = 'test/file.py'
    var_59 = {var_58}
    var_60 = module_0.Config()
    var_61 = 'test'
    var_62 = 'file.py'
    var_63 = 'import os'
    var_64 = var_60.is_skipped(var_12)
    assert var_64 is True



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = 'test_file.py'
    var_5 = 'test_dir'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_dir'
    var_9 = 'test.py'
    var_10 = '*.txt'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = {var_10}
    var_14 = module_0.Config()
    var_15 = {var_9}
    var_16 = module_0.Config()
    var_17 = 'test_dir'
    var_18 = module_0.Config()
    var_19 = '/non/existent/file.py'
    var_20 = 'subdir/file.py'
    var_21 = {var_20}
    var_22 = module_0.Config()
    var_23 = 'subdir'
    var_24 = 'file.py'
    var_25 = 'skip1'
    var_26 = {var_25}
    var_27 = 'skip2'
    var_28 = {var_27}
    var_29 = module_0.Config()
    var_30 = '*.py'
    var_31 = {var_30}
    var_32 = {var_10}
    var_33 = module_0.Config()
    var_34 = False
    var_35 = module_0.Config()
    var_36 = True
    var_37 = module_0.Config()
    var_38 = '.git'
    var_39 = module_0.Config()
    var_40 = 'target.py'
    var_41 = 'link.py'
    var_42 = var_24 / var_41
    var_43 = var_39.is_skipped(var_42)
    var_44 = '/some/dir'
    var_45 = 'relative/path.py'
    var_46 = {var_45}
    var_47 = module_0.Config()
    var_48 = 'path\\to\\file.py'
    var_49 = {var_48}
    var_50 = module_0.Config()
    var_51 = 'path'
    var_52 = 'to'
    var_53 = var_4 / var_52
    var_54 = 'file.py'
    var_55 = var_53 / var_54
    var_56 = True
    var_57 = var_50.is_skipped(var_55)



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyx'
    var_3 = [var_1, var_2]
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyx'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'txt'
    var_9 = 'md'
    var_10 = [var_8, var_9]
    var_11 = 'test.txt'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.md'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.py~'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'backup~'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = b"#!/usr/bin/env python\nprint('hello')"
    var_20 = 'script_without_extension'
    var_21 = var_0.is_supported_filetype(var_20)
    assert var_21 is True
    var_22 = b"print('hello')"
    var_23 = 'no_shebang_file'
    var_24 = var_0.is_supported_filetype(var_23)
    assert var_24 is False
    var_25 = 'nonexistent_file'
    var_26 = var_0.is_supported_filetype(var_25)
    assert var_26 is False



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '[settings]\nline_length = 100\n'
    var_1 = '.isort.cfg'
    var_2 = 'subdir'
    var_3 = "[tool.isort]\nprofile = 'black'\n"
    var_4 = 'pyproject.toml'
    var_5 = 'nested'
    var_6 = 'bad.isort.cfg'
    var_7 = 'invalid content'
    var_8 = 'setup.cfg'
    var_9 = '[isort]\nforce_grid_wrap = true\n'



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyx'
    assert var_2 is False
    var_3 = [var_1, var_2]
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyx'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'txt'
    var_9 = 'md'
    var_10 = [var_8, var_9]
    var_11 = 'test.txt'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.md'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.py~'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = '#!/usr/bin/env python\n'
    assert var_17 is True
    var_18 = "print('Hello')"
    var_19 = "print('Hello')"
    var_20 = b'#!/usr/bin/env python\n'
    var_21 = b"print('Hello')"
    assert var_21 is True
    var_22 = 'non_existent_file.xyz'
    var_23 = var_0.is_supported_filetype(var_22)
    assert var_23 is False
    var_24 = [var_20]
    var_25 = 'test'
    var_26 = var_0.is_supported_filetype(var_25)
    assert var_26 is False
    var_27 = 'test.'
    var_28 = var_0.is_supported_filetype(var_27)
    assert var_28 is False
    var_29 = 'test.module.py'
    var_30 = var_0.is_supported_filetype(var_29)
    assert var_30 is True
    var_31 = 'PY'
    var_32 = 'PYX'
    var_33 = [var_31, var_32]
    var_34 = var_0.is_supported_filetype(var_4)
    assert var_34 is False
    var_35 = 'test.PY'
    var_36 = var_0.is_supported_filetype(var_35)
    assert var_36 is True



