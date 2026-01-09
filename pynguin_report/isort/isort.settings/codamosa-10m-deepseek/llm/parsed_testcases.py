####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0


def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'txt'
    var_3 = 'test.py'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is True
    var_5 = 'test.txt'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is True
    var_7 = 'log'
    var_8 = 'tmp'
    var_9 = 'test.log'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.tmp'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'MockFile'
    var_14 = ()
    var_15 = 'readline'
    var_16 = b'#!/usr/bin/env python\n'
    var_17 = lambda self, size: var_16
    var_18 = {var_15: var_17}
    var_19 = 'test.sh'
    var_20 = var_0.is_supported_filetype(var_19)
    assert var_20 is True
    var_21 = ()
    var_22 = b'no shebang\n'
    var_23 = lambda self, size: var_22
    var_24 = {var_15: var_23}
    var_25 = var_0.is_supported_filetype(var_19)
    assert var_25 is False
    var_26 = 'test.py~'
    var_27 = var_0.is_supported_filetype(var_26)
    assert var_27 is False
    var_28 = 'MockStat'
    var_29 = ()
    var_30 = 'st_mode'
    var_31 = 'test.fifo'
    var_32 = var_0.is_supported_filetype(var_31)
    assert var_32 is False
    var_33 = ()
    var_34 = 'test.error'
    var_35 = var_0.is_supported_filetype(var_34)
    assert var_35 is False
    var_36 = ()
    var_37 = var_0.is_supported_filetype(var_34)
    assert var_37 is False
    var_38 = 'All tests passed!'
    var_39 = print(var_38)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = frozenset()
    var_7 = frozenset()
    var_8 = frozenset()
    var_9 = frozenset()
    var_10 = frozenset()
    var_11 = frozenset()
    var_12 = frozenset()
    var_13 = frozenset()
    var_14 = frozenset()
    var_15 = frozenset()
    var_16 = frozenset()
    var_17 = frozenset()
    var_18 = frozenset()
    var_19 = frozenset()
    var_20 = frozenset()
    var_21 = frozenset()
    var_22 = frozenset()
    var_23 = frozenset()
    var_24 = frozenset()
    var_25 = frozenset()
    var_26 = frozenset()
    var_27 = frozenset()
    var_28 = frozenset()
    var_29 = frozenset()
    var_30 = frozenset()
    var_31 = frozenset()
    var_32 = frozenset()



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length = 100'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = '.isort.cfg'
    var_5 = 'pyproject.toml'
    var_6 = '[settings]\nline_length = 80'
    var_7 = '[tool.isort]\nline_length = 120'
    var_8 = '.isort.cfg'
    var_9 = 'invalid content'
    var_10 = '.isort.cfg'
    var_11 = 'pyproject.toml'
    var_12 = '[settings]\nline_length = 90'
    var_13 = 'invalid toml'
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'invalid_version'
    var_3 = module_0._Config(var_2)
    var_4 = 'all'
    var_5 = module_0._Config(var_4)
    var_6 = '3'
    var_7 = frozenset()
    var_8 = module_0._Config(var_6, known_standard_library=var_7)
    var_9 = 'py3'
    var_10 = True
    var_11 = module_0._Config(force_alphabetical_sort=var_10)
    var_12 = 100
    var_13 = 80
    var_14 = module_0._Config(line_length=var_13, wrap_length=var_12)
    var_15 = 80
    var_16 = 100
    var_17 = module_0._Config(line_length=var_16, wrap_length=var_15)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'txt'
    var_4 = 'test.txt'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = 'test.unknown'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.py~'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = var_0.is_supported_filetype(var_3)
    assert var_10 is False
    var_11 = '/nonexistent/file.py'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = '#!/usr/bin/env python\n'
    assert var_13 is True



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/path/to/skip'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = [var_1]
    var_5 = '/path/to/not_skip'
    var_6 = module_0.Config()
    var_7 = '*.txt'
    var_8 = [var_7]
    var_9 = '/path/to/file.txt'
    var_10 = module_0.Config()
    var_11 = [var_7]
    var_12 = '/path/to/file.py'
    var_13 = module_0.Config()
    var_14 = '/path/to/.git'
    var_15 = module_0.Config()
    var_16 = module_0.Config()
    var_17 = [var_1]
    var_18 = module_0.Config()
    var_19 = [var_1]
    var_20 = module_0.Config()
    var_21 = '/path/to/nonexistent'
    var_22 = module_0.Config()
    var_23 = 'skip'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_23]
    var_27 = 'not_skip'
    var_28 = module_0.Config()
    var_29 = [var_7]
    var_30 = 'file.txt'
    var_31 = module_0.Config()
    var_32 = [var_7]
    var_33 = 'file.py'
    var_34 = module_0.Config()
    var_35 = '.git'
    var_36 = module_0.Config()
    var_37 = module_0.Config()
    var_38 = [var_23]
    var_39 = module_0.Config()
    var_40 = [var_23]
    var_41 = module_0.Config()
    var_42 = 'nonexistent'
    var_43 = module_0.Config()
    var_44 = [var_1]
    var_45 = module_0.Config()
    var_46 = [var_1]
    var_47 = module_0.Config()
    var_48 = [var_7]
    var_49 = module_0.Config()
    var_50 = [var_7]
    var_51 = module_0.Config()
    var_52 = module_0.Config()
    var_53 = module_0.Config()
    var_54 = [var_1]
    var_55 = module_0.Config()
    var_56 = [var_1]
    var_57 = module_0.Config()
    var_58 = module_0.Config()
    var_59 = [var_1]
    var_60 = '\\path\\to\\skip'
    var_61 = module_0.Config()
    var_62 = [var_1]
    var_63 = '\\path\\to\\not_skip'
    var_64 = module_0.Config()
    var_65 = [var_7]
    var_66 = '\\path\\to\\file.txt'
    var_67 = module_0.Config()
    var_68 = [var_7]
    var_69 = '\\path\\to\\file.py'
    var_70 = module_0.Config()
    var_71 = '\\path\\to\\.git'
    var_72 = module_0.Config()
    var_73 = module_0.Config()
    var_74 = [var_1]
    var_75 = module_0.Config()
    var_76 = [var_1]



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/path/to/directory'
    var_2 = module_0.Config()
    var_3 = '/path/to/file.txt'
    var_4 = module_0.Config()
    var_5 = '/path/to/symlink'
    var_6 = module_0.Config()
    var_7 = '/path/to/nonexistent'
    var_8 = [var_3]
    var_9 = module_0.Config()
    var_10 = '*.txt'
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = True
    var_14 = module_0.Config()
    var_15 = False
    var_16 = module_0.Config()
    var_17 = module_0.Config()
    var_18 = '/path/to'
    var_19 = {var_3}
    var_20 = module_0.Config()
    var_21 = '/path/to/other.txt'
    var_22 = {var_21}
    var_23 = [var_1]
    var_24 = module_0.Config()
    var_25 = '*/directory'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = module_0.Config()
    var_29 = module_0.Config()
    var_30 = module_0.Config()
    var_31 = {var_1}
    var_32 = module_0.Config()
    var_33 = '/path/to/other'
    var_34 = {var_33}
    var_35 = [var_5]
    var_36 = module_0.Config()
    var_37 = '*/symlink'
    var_38 = [var_37]
    var_39 = module_0.Config()
    var_40 = module_0.Config()
    var_41 = module_0.Config()
    var_42 = module_0.Config()
    var_43 = {var_5}
    var_44 = module_0.Config()
    var_45 = {var_33}
    var_46 = [var_7]
    var_47 = module_0.Config()
    var_48 = '*/nonexistent'
    var_49 = [var_48]
    var_50 = module_0.Config()
    var_51 = module_0.Config()
    var_52 = module_0.Config()
    var_53 = module_0.Config()
    var_54 = {var_7}
    var_55 = module_0.Config()
    var_56 = {var_33}
    var_57 = [var_3]
    var_58 = [var_10]
    var_59 = module_0.Config()
    var_60 = [var_3]
    var_61 = '*.py'
    var_62 = [var_61]
    var_63 = module_0.Config()
    var_64 = [var_21]
    var_65 = [var_10]
    var_66 = module_0.Config()
    var_67 = [var_21]
    var_68 = [var_61]
    var_69 = module_0.Config()
    var_70 = [var_1]
    var_71 = [var_25]
    var_72 = module_0.Config()



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = {var_1}
    var_3 = module_0.Config()
    var_4 = '*.py'
    var_5 = {var_4}
    var_6 = module_0.Config()
    var_7 = 'test_dir'
    var_8 = {var_7}
    var_9 = module_0.Config()
    var_10 = {var_1}
    var_11 = module_0.Config()
    var_12 = {var_4}
    var_13 = module_0.Config()
    var_14 = 'other_file.py'
    var_15 = {var_14}
    var_16 = '*.txt'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = 'test_file.py~'
    var_20 = module_0.Config()
    var_21 = 'skipped_dir/test_file.py'
    var_22 = 'skipped_dir'
    var_23 = {var_22}
    var_24 = module_0.Config()
    var_25 = 'skipped_dir/*'
    var_26 = {var_25}
    var_27 = module_0.Config()
    var_28 = '*skipped*'
    var_29 = {var_28}
    var_30 = module_0.Config()
    var_31 = '*other*'
    var_32 = {var_31}
    var_33 = module_0.Config()
    var_34 = {var_1}
    var_35 = {var_4}
    var_36 = module_0.Config()
    var_37 = {var_1}
    var_38 = {var_4}
    var_39 = module_0.Config()
    var_40 = {var_14}
    var_41 = {var_1}
    var_42 = module_0.Config()
    var_43 = '.git'
    var_44 = True
    var_45 = module_0.Config()
    var_46 = 'non_existent_file.py'
    var_47 = module_0.Config()
    var_48 = 'test-file[1].py'
    var_49 = module_0.Config()
    var_50 = 'test*'
    var_51 = '*file*'
    var_52 = {var_50, var_4, var_51}
    var_53 = module_0.Config()
    var_54 = 'a/b/c/test_file.py'
    var_55 = 'a/b/c'
    var_56 = {var_55}
    var_57 = module_0.Config()
    var_58 = 'b/c'
    var_59 = {var_58}
    var_60 = module_0.Config()
    var_61 = '/absolute/path/test_file.py'
    var_62 = {var_61}
    var_63 = module_0.Config()
    var_64 = 'relative/test_file.py'
    var_65 = {var_64}
    var_66 = module_0.Config()
    var_67 = 'relative'
    var_68 = {var_67}
    var_69 = module_0.Config()
    var_70 = 'sibling'
    var_71 = {var_70}
    var_72 = module_0.Config()
    var_73 = '**/test_*.py'
    var_74 = {var_73}
    var_75 = module_0.Config()
    var_76 = 'relative/*.py'
    var_77 = {var_76}
    var_78 = module_0.Config()
    var_79 = 'other/*.py'
    var_80 = {var_79}
    var_81 = module_0.Config()
    var_82 = 'specific_file.py'
    var_83 = {var_82}
    var_84 = '*.pyc'
    var_85 = {var_84}
    var_86 = module_0.Config()
    var_87 = 'test_*'
    var_88 = {var_87}
    var_89 = module_0.Config()
    var_90 = 'test_file'
    var_91 = {var_90}
    var_92 = {var_4}
    var_93 = module_0.Config()
    var_94 = 'nested_file.py'



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '*.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_0.Config()
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_0.Config()
    var_11 = 'test_link'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = module_0.Config()
    var_15 = module_0.Config()
    var_16 = 'non_existent_file'
    var_17 = True
    var_18 = module_0.Config()
    var_19 = module_0.Config()
    var_20 = '/'
    var_21 = False
    var_22 = module_0.Config()
    var_23 = module_0.Config()
    var_24 = module_0.Config()
    var_25 = module_0.Config()
    var_26 = module_0.Config()
    var_27 = module_0.Config()
    var_28 = module_0.Config()
    var_29 = module_0.Config()
    var_30 = [var_0]
    var_31 = module_0.Config()
    var_32 = [var_0]
    var_33 = module_0.Config()
    var_34 = [var_3]
    var_35 = module_0.Config()
    var_36 = [var_3]
    var_37 = module_0.Config()
    var_38 = module_0.Config()
    var_39 = module_0.Config()
    var_40 = [var_7]
    var_41 = module_0.Config()
    var_42 = 'test_*'
    var_43 = [var_42]
    var_44 = module_0.Config()
    var_45 = module_0.Config()
    var_46 = [var_11]
    var_47 = module_0.Config()
    var_48 = [var_42]
    var_49 = module_0.Config()
    var_50 = module_0.Config()
    var_51 = [var_16]
    var_52 = module_0.Config()
    var_53 = 'non_*'
    var_54 = [var_53]
    var_55 = module_0.Config()
    var_56 = module_0.Config()
    var_57 = [var_0]
    var_58 = module_0.Config()



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/tmp/test_dir'
    var_2 = module_0.Config()
    var_3 = '/tmp/test_file.txt'
    var_4 = module_0.Config()
    var_5 = '/tmp/test_symlink'
    var_6 = module_0.Config()
    var_7 = '/tmp/test_nonexistent'
    var_8 = [var_3]
    var_9 = module_0.Config()
    var_10 = [var_3]
    var_11 = module_0.Config()
    var_12 = '/tmp/test_file2.txt'
    var_13 = '/tmp/*.txt'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = [var_13]
    var_17 = module_0.Config()
    var_18 = '/tmp/test_file2.py'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = module_0.Config()
    var_22 = [var_1]
    var_23 = module_0.Config()
    var_24 = [var_1]
    var_25 = module_0.Config()
    var_26 = '/tmp/test_dir2'
    var_27 = '/tmp/*'
    var_28 = [var_27]
    var_29 = module_0.Config()
    var_30 = [var_27]
    var_31 = module_0.Config()
    var_32 = module_0.Config()
    var_33 = module_0.Config()
    var_34 = [var_5]
    var_35 = module_0.Config()
    var_36 = [var_5]
    var_37 = module_0.Config()
    var_38 = '/tmp/test_symlink2'
    var_39 = [var_27]
    var_40 = module_0.Config()
    var_41 = [var_27]
    var_42 = module_0.Config()
    var_43 = module_0.Config()
    var_44 = module_0.Config()
    var_45 = [var_3]
    var_46 = [var_13]
    var_47 = module_0.Config()
    var_48 = [var_3]
    var_49 = '/tmp/*.py'
    var_50 = [var_49]
    var_51 = module_0.Config()
    var_52 = [var_12]
    var_53 = [var_13]
    var_54 = module_0.Config()
    var_55 = [var_12]
    var_56 = [var_49]
    var_57 = module_0.Config()
    var_58 = [var_3]
    var_59 = module_0.Config()
    var_60 = [var_12]
    var_61 = module_0.Config()
    var_62 = [var_3]
    var_63 = module_0.Config()
    var_64 = [var_12]
    var_65 = module_0.Config()
    var_66 = [var_13]
    var_67 = module_0.Config()
    var_68 = [var_49]
    var_69 = module_0.Config()
    var_70 = [var_13]
    var_71 = module_0.Config()
    var_72 = [var_49]
    var_73 = module_0.Config()



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nprofile = black'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nprofile = black'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile = black'
    var_7 = 'subdir'
    var_8 = '[settings]\nline_length = 100'
    var_9 = '.isort.cfg'
    var_10 = 'invalid content'
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = var_0.is_skipped(var_1)
    var_3 = module_0.Config()
    var_4 = 'test_file.py'
    var_5 = var_3.is_skipped(var_1)
    var_6 = module_0.Config()
    var_7 = 'test_dir/test_file.py'
    var_8 = var_6.is_skipped(var_1)
    var_9 = module_0.Config()
    var_10 = var_9.is_skipped(var_1)
    var_11 = module_0.Config()
    var_12 = var_11.is_skipped(var_1)
    var_13 = module_0.Config()
    var_14 = var_13.is_skipped(var_1)
    var_15 = module_0.Config()
    var_16 = var_15.is_skipped(var_1)
    var_17 = module_0.Config()
    var_18 = var_17.is_skipped(var_1)
    var_19 = module_0.Config()
    var_20 = var_19.is_skipped(var_1)
    var_21 = module_0.Config()
    var_22 = var_21.is_skipped(var_1)
    var_23 = module_0.Config()
    var_24 = var_23.is_skipped(var_1)
    var_25 = module_0.Config()
    var_26 = var_25.is_skipped(var_1)
    var_27 = module_0.Config()
    var_28 = var_27.is_skipped(var_1)
    var_29 = module_0.Config()
    var_30 = var_29.is_skipped(var_1)
    var_31 = module_0.Config()
    var_32 = var_31.is_skipped(var_1)
    var_33 = module_0.Config()
    var_34 = var_33.is_skipped(var_1)
    var_35 = module_0.Config()
    var_36 = var_35.is_skipped(var_1)
    var_37 = module_0.Config()
    var_38 = var_37.is_skipped(var_1)
    var_39 = module_0.Config()
    var_40 = var_39.is_skipped(var_1)
    var_41 = module_0.Config()
    var_42 = var_41.is_skipped(var_1)
    var_43 = module_0.Config()
    var_44 = var_43.is_skipped(var_1)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/path/to/skip'
    var_2 = '/path/to/not_skip'
    var_3 = '*.txt'
    var_4 = '/path/to/file.txt'
    var_5 = '/path/to/file.py'
    var_6 = '/path/to/skip_dir'
    var_7 = '/path/to/not_skip_dir'
    var_8 = '/path/to/skip_link'
    var_9 = '/path/to/not_skip_link'
    var_10 = '/path/to/nonexistent'
    var_11 = '/path/to/git_folder'
    var_12 = '/path/to/git_folder/file.py'
    var_13 = {var_12}
    var_14 = '/path/to/git_folder/other_file.py'
    var_15 = {var_12}
    var_16 = '/path/to/.git'
    var_17 = '*/skip_dir/*'
    var_18 = '/path/to/skip_dir/file.py'
    var_19 = '/path/to/not_skip_dir/file.py'
    var_20 = '*/skip_link/*'
    var_21 = '/path/to/skip_link/file.py'
    var_22 = '/path/to/not_skip_link/file.py'
    var_23 = '/path/to/skip_file.txt'
    var_24 = '/path/to/not_skip_file.txt'
    var_25 = '/path/to/not_skip_file.py'
    var_26 = {var_12}



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'txt'
    var_3 = 'test.py'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is True
    var_5 = 'test.txt'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is True
    var_7 = 'exe'
    var_8 = 'dll'
    var_9 = 'test.exe'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.dll'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is True
    var_15 = 'test'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'test'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = 'test~'
    var_20 = var_0.is_supported_filetype(var_19)
    assert var_20 is False
    var_21 = 'test'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nprofile = black'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    assert var_3 == 1
    var_4 = '[settings]\nprofile = black'
    var_5 = '[settings]\nline_length = 100'
    var_6 = '.isort.cfg'
    var_7 = 'invalid content'
    var_8 = 'always'
    var_9 = 0
    var_10 = 'All tests passed!'
    var_11 = print(var_10)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = frozenset()
    var_7 = 100
    var_8 = 'os'
    var_9 = 'sys'
    var_10 = {var_8, var_9}
    var_11 = module_0.Config()
    var_12 = {var_8, var_9}
    var_13 = frozenset(var_12)
    var_14 = 'black'
    var_15 = module_0.Config()
    var_16 = 'test_settings.ini'
    var_17 = '[isort]\nline_length = 120\nknown_standard_library = os,sys\n'
    var_18 = module_0.Config(var_16)
    var_19 = {var_8, var_9}
    var_20 = frozenset(var_19)
    var_21 = '.'
    var_22 = module_0.Config(settings_path=var_21)
    var_23 = module_0.Config()
    var_24 = module_0.Config(config=var_23)
    var_25 = {var_8, var_9}
    var_26 = 120
    var_27 = module_0.Config()
    var_28 = 'invalid_settings.ini'
    var_29 = '[settings]\nline_length = 120\n'
    var_30 = True
    var_31 = module_0.Config(var_28)
    var_32 = 'invalid_profile'
    var_33 = module_0.Config()
    var_34 = module_0.Config()
    var_35 = True
    var_36 = module_0.Config()
    var_37 = 'colorama'
    var_38 = module_0.Config()
    var_39 = 'natural'
    var_40 = module_0.Config()
    var_41 = 'invalid_sort'
    var_42 = module_0.Config()
    var_43 = 'django'
    var_44 = {var_43}
    var_45 = 'FUTURE'
    var_46 = 'STDLIB'
    var_47 = 'DJANGO'
    var_48 = 'THIRDPARTY'
    var_49 = (var_45, var_46, var_47, var_48)
    var_50 = module_0.Config()
    var_51 = {var_43}
    var_52 = frozenset(var_51)
    var_53 = {var_43: var_52}
    var_54 = 'Standard Library'
    var_55 = 'End Stdlib'
    var_56 = module_0.Config()
    var_57 = module_0.Config()
    var_58 = 'py'
    var_59 = 'pyx'
    var_60 = (var_58, var_59)
    var_61 = module_0.Config()
    var_62 = 'txt'
    var_63 = 'md'
    var_64 = (var_62, var_63)
    var_65 = module_0.Config()
    var_66 = 60
    var_67 = 80
    var_68 = module_0.Config()
    var_69 = 100
    var_70 = 80
    var_71 = module_0.Config()
    var_72 = 'All tests passed!'
    var_73 = print(var_72)



