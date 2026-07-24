####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = {}
    var_2 = module_0.Trie(var_0, var_1)
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=88'
    var_5 = 'subdir1'
    var_6 = 'setup.cfg'
    var_7 = '[isort]\nprofile=black'
    var_8 = 'subdir2'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nknown_first_party=myapp'
    var_11 = 'nested'
    var_12 = "[settings]\nindent='    '"
    var_13 = 'invalid.cfg'
    var_14 = 'invalid content'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'other_*'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = module_0.Config()
    var_13 = 'test_dir'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = 'symlink.py'
    var_17 = module_0.Config()
    var_18 = module_0.Config()
    var_19 = '.git'
    var_20 = 'parent_dir'
    var_21 = {var_20}
    var_22 = module_0.Config()
    var_23 = 'parent_dir/test_file.py'
    var_24 = module_0.Config()
    var_25 = 'non_existent_file.py'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'
    var_2 = 'subdir'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nline_length=88\n'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_me.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'dont_skip.py'
    var_4 = 'skip_dir'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'skip_dir/file.py'
    var_8 = 'other_dir/file.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test.tmp'
    var_13 = 'test.py'
    var_14 = [var_0]
    var_15 = [var_9]
    var_16 = module_0.Config()
    var_17 = 'normal.py'
    var_18 = True
    var_19 = module_0.Config()
    var_20 = '/test'
    var_21 = '/test/committed.py'
    var_22 = '/test/uncommitted.py'
    var_23 = module_0.Config()
    var_24 = 'nonexistent.py'
    var_25 = module_0.Config()
    var_26 = 'file.py~'
    var_27 = '/project'
    var_28 = module_0.Config()
    var_29 = '/project/skip_me.py'
    var_30 = '/other/skip_me.py'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    assert var_3 is True
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.tmp'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.tmp'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = 'test_dir'
    var_15 = True
    var_16 = module_0.Config()
    var_17 = 'nonexistent_file.py'
    var_18 = module_0.Config()
    var_19 = '/test'
    var_20 = '/test/file1.py'
    var_21 = '/test/file2.py'
    var_22 = module_0.Config()
    var_23 = '/test'
    var_24 = '/test/file1.py'
    var_25 = var_22.is_skipped(var_21)
    assert var_25 is False
    var_26 = module_0.Config()
    var_27 = '.git'
    var_28 = 'dir1'
    var_29 = {var_28}
    var_30 = module_0.Config()
    var_31 = 'dir1/file.py'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'nonexistent.py'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = '#!/usr/bin/env python\n'
    assert var_9 is True
    var_10 = "print('hello')"
    var_11 = "print('hello')"
    assert var_11 is False



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/repo'
    var_17 = '/repo/file1.py'
    var_18 = '/repo/file2.py'
    var_19 = '/repo/file3.py'
    var_20 = module_0.Config()
    var_21 = 'nonexistent.py'
    var_22 = module_0.Config()
    var_23 = 'file.py~'
    var_24 = '/project'
    var_25 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'subdir'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nline_length=88'
    var_5 = '.isort.cfg'
    var_6 = '[isort]\nmulti_line_output=3'
    var_7 = 'invalid.cfg'
    var_8 = 'invalid content'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = True



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_configs'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = 'pyproject.toml'
    var_4 = 'tox.ini'
    var_5 = '.isort.cfg'
    var_6 = '[isort]\nprofile=black'
    var_7 = '[tool.isort]\nprofile="black"'
    var_8 = 'profile=black'
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_6, var_5: var_8}
    var_10 = module_0.find_all_configs(var_0)
    var_11 = var_10.search(var_1)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'test*'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = {var_5}
    var_9 = module_0.Config()
    var_10 = 'other.py'
    var_11 = {var_10}
    var_12 = 'other*'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = module_0.Config()
    var_16 = 'test_dir'
    var_17 = module_0.Config()
    var_18 = 'test_link'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = '.git'
    var_22 = False
    var_23 = module_0.Config()
    var_24 = module_0.Config()
    var_25 = '/test'
    var_26 = '/test/test.py'
    var_27 = {var_26}
    var_28 = '/test/other.py'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyi'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyi'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'test.txt'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'test'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is True
    var_14 = 'nonexistent'
    var_15 = var_0.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = 'fifo'
    var_17 = var_0.is_supported_filetype(var_16)
    assert var_17 is False



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.py'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.py'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is True
    var_11 = 'test.py'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'nonexistent.py'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=88\n'
    var_5 = 'empty'
    var_6 = 'non_existent'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = 'other_*.py'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = module_0.Config()
    var_14 = 'some_directory'
    var_15 = module_0.Config()
    var_16 = 'some_symlink'
    var_17 = True
    var_18 = module_0.Config()
    var_19 = '.git'
    var_20 = module_0.Config()
    var_21 = '/some/path/file1.py'
    var_22 = '/some/path/file2.py'
    var_23 = '/some/path/file3.py'
    var_24 = module_0.Config()
    var_25 = module_0.Config()
    var_26 = 'test.py~'
    var_27 = module_0.Config()
    var_28 = 'test.py'
    var_29 = var_27.is_skipped(var_1)
    var_30 = 'txt'
    var_31 = {var_30}
    var_32 = module_0.Config()
    var_33 = 'test.txt'
    var_34 = 'py'
    var_35 = {var_34}
    var_36 = module_0.Config()
    var_37 = {var_28}
    var_38 = module_0.Config()
    var_39 = {var_6}
    var_40 = module_0.Config()



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'other_*'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = module_0.Config()
    var_13 = 'test_directory'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/test'
    var_17 = '/test/file1.py'
    var_18 = '/test/file2.py'
    var_19 = {var_17, var_18}
    var_20 = '/test/file3.py'
    var_21 = module_0.Config()
    var_22 = {var_17, var_18}
    var_23 = module_0.Config()
    var_24 = '.git'
    var_25 = module_0.Config()
    var_26 = 'test_file.py~'
    var_27 = module_0.Config()
    var_28 = '/dev/zero'
    var_29 = var_27.is_skipped(var_1)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'subdir1'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nline_length=88\n'
    var_5 = 'subdir2'
    var_6 = '.isort.cfg'
    var_7 = 'profile=black\n'
    var_8 = 'invalid.cfg'
    var_9 = 'invalid content'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 0
    var_3 = 'subdir'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.isort]\nprofile="black"'
    var_6 = '.isort.cfg'
    var_7 = 'profile=black'
    var_8 = 'invalid.cfg'
    var_9 = 'invalid content'
    var_10 = 'nested'



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = '*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = '*.txt'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = module_0.Config()
    var_13 = 'test_dir'
    var_14 = module_0.Config()
    var_15 = 'test_link'
    var_16 = module_0.Config()
    var_17 = 'non_existent_file'
    var_18 = True
    var_19 = module_0.Config()
    var_20 = '.git'
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = '/test/test.py'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=88\n'
    var_3 = 'dir2'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nprofile=black\n'
    var_6 = 'subdir'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nknown_first_party=myapp\n'
    var_9 = 'nonexistent.cfg'
    var_10 = 'empty'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = 'other_*.py'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = 'src/'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'src/test.py'
    var_17 = 'other/'
    var_18 = {var_17}
    var_19 = module_0.Config()
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '/repo'
    var_23 = '/repo/tracked.py'
    var_24 = {var_23}
    var_25 = '/repo/untracked.py'
    var_26 = module_0.Config()
    var_27 = {var_23}
    var_28 = module_0.Config()
    var_29 = '/some/directory'
    var_30 = module_0.Config()
    var_31 = '/nonexistent/file.py'



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file'
    var_2 = module_0.Config(var_1)
    var_3 = 'nonexistent_path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = 100
    var_6 = module_0.Config()
    var_7 = 'black'
    var_8 = module_0.Config()
    var_9 = var_8.source
    var_10 = str(var_9)
    var_11 = 'invalid_profile'
    var_12 = module_0.Config()
    var_13 = module_0._Config()
    var_14 = module_0.Config(config=var_13)
    var_15 = module_0._Config()
    var_16 = module_0.Config(config=var_15)
    var_17 = 100
    var_18 = 80
    var_19 = module_0.Config()
    var_20 = True
    var_21 = module_0.Config()
    var_22 = 'value'
    var_23 = module_0.Config()
    var_24 = 'custom'
    var_25 = 'custom_module'
    var_26 = {var_25}
    var_27 = {var_24: var_26}
    var_28 = module_0.Config()
    var_29 = 'Custom Heading'
    var_30 = module_0.Config()
    var_31 = 'Custom Footer'
    var_32 = module_0.Config()
    var_33 = 'src'
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = var_35.src_paths
    var_37 = len(var_36)
    var_38 = module_0.Config()
    var_39 = 'natural'
    var_40 = module_0.Config()
    var_41 = 'invalid'
    var_42 = module_0.Config()



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = '*.test.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'example.test.py'
    var_7 = 'other_file.py'
    var_8 = {var_7}
    var_9 = module_0.Config()
    var_10 = True
    var_11 = module_0.Config()
    var_12 = '/test'
    var_13 = '/test/file1.py'
    var_14 = '/test/file2.py'
    var_15 = {var_13, var_14}
    var_16 = '/test/file3.py'
    var_17 = module_0.Config()
    var_18 = {var_13, var_14}
    var_19 = 'test_dir'
    var_20 = {var_19}
    var_21 = module_0.Config()
    var_22 = 'test_dir/file.py'
    var_23 = 'other_dir'
    var_24 = {var_23}
    var_25 = module_0.Config()
    var_26 = 'test_*'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = {var_26}
    var_30 = module_0.Config()
    var_31 = 'example_file.py'
    var_32 = module_0.Config()



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'other_file.py'
    var_13 = 'tests'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'tests/test.py'
    var_17 = {var_13}
    var_18 = module_0.Config()
    var_19 = 'src/test.py'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '.'
    var_23 = 'src/file.py'
    var_24 = {var_23}
    var_25 = 'other/file.py'
    var_26 = module_0.Config()
    var_27 = {var_23, var_25}
    var_28 = {var_0}
    var_29 = module_0.Config()
    var_30 = 'test_dir'
    var_31 = {var_0}
    var_32 = module_0.Config()
    var_33 = 'test.py'
    var_34 = var_0 / var_33
    var_35 = 'link.py'



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 120
    var_3 = module_0.Config()
    var_4 = 'nonexistent_file.py'
    var_5 = module_0.Config(var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.Config(settings_path=var_6)
    var_8 = 100
    var_9 = 5
    var_10 = module_0._Config(line_length=var_8, wrap_length=var_9)
    var_11 = module_0.Config(config=var_10)
    var_12 = 'black'
    var_13 = module_0.Config()
    var_14 = 'nonexistent_profile'
    var_15 = module_0.Config()
    var_16 = 'bar'
    var_17 = 'baz'
    var_18 = [var_16, var_17]
    var_19 = 'FOO'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = 'Foo Imports'
    var_23 = module_0.Config()
    var_24 = 'End of Foo Imports'
    var_25 = module_0.Config()
    var_26 = False
    var_27 = True
    var_28 = module_0.Config()
    var_29 = 'value'
    var_30 = module_0.Config()
    var_31 = module_0.Config()
    var_32 = 'nonexistent_formatter'
    var_33 = module_0.Config()
    var_34 = 'natural'
    var_35 = module_0.Config()
    var_36 = 'nonexistent_sort'
    var_37 = module_0.Config()



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests/'
    var_5 = {var_4}
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.txt'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = 'test.txt'
    var_13 = True
    var_14 = module_0.Config()
    var_15 = '/test'
    var_16 = '/test/file1.py'
    var_17 = '/test/file2.py'
    var_18 = '/test/file3.py'
    var_19 = module_0.Config()
    var_20 = 'nonexistent.py'
    var_21 = 'test_dir'
    var_22 = {var_21}
    var_23 = module_0.Config()
    var_24 = 'test_dir/file.py'
    var_25 = '/project'
    var_26 = 'skip_me.py'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = '/project/skip_me.py'
    var_30 = '/other/skip_me.py'
    var_31 = 'tests/*'
    var_32 = {var_31}
    var_33 = module_0.Config()
    var_34 = 'src/tests/test.py'
    var_35 = module_0.Config()
    var_36 = '.git'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = '.isort.cfg'
    var_2 = '[isort]\nprofile=black\n'
    var_3 = 'dir2'
    var_4 = 'setup.cfg'
    var_5 = '[tool.isort]\nprofile=black\n'
    var_6 = 'empty_dir'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 120
    var_3 = module_0.Config()
    var_4 = 100
    var_5 = 80
    var_6 = module_0.Config()
    var_7 = 'nonexistent_file.py'
    var_8 = module_0.Config(var_7)
    var_9 = '/nonexistent/path'
    var_10 = module_0.Config(settings_path=var_9)
    var_11 = '\t'
    var_12 = 100
    var_13 = module_0._Config(line_length=var_12, indent=var_11)
    var_14 = module_0.Config(config=var_13)
    var_15 = 'black'
    var_16 = module_0.Config()
    var_17 = 'nonexistent_profile'
    var_18 = module_0.Config()
    var_19 = 'bar'
    var_20 = 'baz'
    var_21 = [var_19, var_20]
    var_22 = 'FOO'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = 'Foo Imports'
    var_26 = module_0.Config()
    var_27 = 'End of Foo Imports'
    var_28 = module_0.Config()
    var_29 = 'value'
    var_30 = module_0.Config()
    var_31 = True
    var_32 = module_0.Config()
    var_33 = module_0.Config()
    var_34 = 'nonexistent_formatter'
    var_35 = module_0.Config()
    var_36 = 'natural'
    var_37 = module_0.Config()
    var_38 = 'nonexistent_sort'
    var_39 = module_0.Config()



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'txt'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = 'test.txt'
    var_7 = var_5.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.py~'
    var_9 = var_5.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py'
    var_11 = var_5.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'test.py'
    var_13 = var_5.is_supported_filetype(var_12)
    assert var_13 is True
    var_14 = 'test.py'
    var_15 = var_5.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = 'test.py'
    var_17 = var_5.is_supported_filetype(var_16)
    assert var_17 is False



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.isort]\nprofile=black\n'
    var_4 = 'subdir'
    var_5 = '.isort.cfg'
    var_6 = set()
    var_7 = 'empty'
    var_8 = 'non_existent'



# Parsed testcases at query #28
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = '/nonexistent/path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = module_0._Config()
    var_6 = module_0.Config(config=var_5)
    var_7 = True
    var_8 = module_0.Config()
    var_9 = 'nonexistent_profile'
    var_10 = module_0.Config()
    var_11 = 100
    var_12 = 50
    var_13 = module_0.Config()
    var_14 = 50
    var_15 = 100
    var_16 = module_0.Config()
    var_17 = '4'
    var_18 = module_0.Config()
    var_19 = 'tab'
    var_20 = module_0.Config()
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = False
    var_24 = 'value'
    var_25 = module_0.Config()



# Parsed testcases at query #29
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = '*.tmp'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'file.tmp'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'file.py'
    var_13 = module_0.Config()
    var_14 = 'directory'
    var_15 = True
    var_16 = module_0.Config()
    var_17 = '/repo/file.py'
    var_18 = '/repo/ignored.py'
    var_19 = module_0.Config()
    var_20 = 'skip_dir'
    var_21 = {var_20}
    var_22 = module_0.Config()
    var_23 = 'skip_dir/file.py'
    var_24 = 'other_dir'
    var_25 = {var_24}
    var_26 = module_0.Config()
    var_27 = module_0.Config()
    var_28 = 'file.py~'



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_directory'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\n'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.isort]\nprofile=black\n'
    var_6 = 'subdir'
    var_7 = '.isort.cfg'
    var_8 = '[isort]\nprofile=black\n'
    var_9 = module_0.find_all_configs(var_0)
    var_10 = var_9.children
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = []
    var_13 = lambda node: found_configs.append(node.value)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'subdir'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nline_length=88'
    var_5 = '.isort.cfg'
    var_6 = '[isort]\nmulti_line_output=3'
    var_7 = 'invalid.cfg'
    var_8 = 'invalid content'
    var_9 = 'nested'
    var_10 = "[isort]\nindent='    '"



# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = '/nonexistent/path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = module_0._Config()
    var_6 = module_0.Config(config=var_5)
    var_7 = 'nonexistent_profile'
    var_8 = module_0.Config()
    var_9 = 'nonexistent_formatter'
    var_10 = module_0.Config()
    var_11 = 'nonexistent_sort'
    var_12 = module_0.Config()
    var_13 = 'value'
    var_14 = module_0.Config()
    var_15 = 100
    var_16 = '    '
    var_17 = module_0.Config()
    var_18 = True
    var_19 = module_0.Config()
    var_20 = 'src'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'numpy'
    var_24 = 'pandas'
    var_25 = [var_23, var_24]
    var_26 = module_0.Config()
    var_27 = 'Standard Library'
    var_28 = module_0.Config()
    var_29 = 'End Standard Library'
    var_30 = module_0.Config()
    var_31 = 'FUTURE'
    var_32 = 'STDLIB'
    var_33 = 'THIRDPARTY'
    var_34 = 'FIRSTPARTY'
    var_35 = 'LOCALFOLDER'
    var_36 = [var_31, var_32, var_33, var_34, var_35]
    var_37 = module_0.Config()
    var_38 = True
    var_39 = module_0.Config()



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.pyi'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is True
    var_5 = 'test.jpg'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.png'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.py~'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.py'
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



# Parsed testcases at query #34
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'dir/'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = 'dir/test.py'
    var_16 = 'other_dir/'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = True
    var_20 = module_0.Config()
    var_21 = '/repo'
    var_22 = '/repo/committed.py'
    var_23 = {var_22}
    var_24 = '/repo/ignored.py'
    var_25 = module_0.Config()
    var_26 = {var_22}
    var_27 = {var_0}
    var_28 = module_0.Config()
    var_29 = 'test_dir'
    var_30 = module_0.Config()
    var_31 = 'nonexistent.py'



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    var_3 = 'test.pyi'
    var_4 = var_0.is_supported_filetype(var_3)
    var_5 = 'test.c'
    var_6 = var_0.is_supported_filetype(var_5)
    var_7 = 'test.h'
    var_8 = var_0.is_supported_filetype(var_7)
    var_9 = 'test.jpg'
    var_10 = var_0.is_supported_filetype(var_9)
    var_11 = 'test.png'
    var_12 = var_0.is_supported_filetype(var_11)
    var_13 = 'test.py~'
    var_14 = var_0.is_supported_filetype(var_13)
    var_15 = 'nonexistent.py'
    var_16 = var_0.is_supported_filetype(var_15)
    var_17 = '#!/usr/bin/env python\n'
    var_18 = "print('hello')"
    var_19 = var_0.is_supported_filetype(var_6)
    var_20 = "print('hello')"
    var_21 = var_0.is_supported_filetype(var_4)
    var_22 = '#!/bin/bash\necho hello'
    var_23 = var_0.is_supported_filetype(var_4)
    var_24 = 'echo hello'
    var_25 = var_0.is_supported_filetype(var_4)



# Parsed testcases at query #36
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'txt'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = 'test.txt'
    var_7 = var_5.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.py~'
    var_9 = var_5.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py'
    var_11 = var_5.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'test.py'
    var_13 = var_5.is_supported_filetype(var_12)
    assert var_13 is True
    var_14 = 'test.py'
    var_15 = var_5.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = 'nonexistent.py'
    var_17 = var_5.is_supported_filetype(var_16)
    assert var_17 is False



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = '.isort.cfg'
    var_2 = '[isort]\nprofile=black\n'
    var_3 = 'dir2'
    var_4 = 'subdir'
    var_5 = True
    var_6 = 'setup.cfg'
    var_7 = '[isort]\nline_length=120\n'
    var_8 = 'empty_dir'
    var_9 = 'non_existent'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'pyproject.toml'
    var_2 = "[tool.isort]\nprofile = 'black'\n"
    var_3 = 'dir2'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nline_length = 88\n'
    var_6 = 'dir3'
    var_7 = 'subdir'
    var_8 = True
    var_9 = 'setup.cfg'
    var_10 = '[isort]\nmulti_line_output = 3\n'
    var_11 = 'nonexistent.toml'
    var_12 = 'empty'
    var_13 = 'nonexistent.cfg'



# Parsed testcases at query #39
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = 'dir1'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'dir1/file.py'
    var_17 = {var_13}
    var_18 = module_0.Config()
    var_19 = 'dir2/file.py'
    var_20 = {var_13}
    var_21 = module_0.Config()
    var_22 = {var_13}
    var_23 = module_0.Config()
    var_24 = 'dir2'
    var_25 = 'link1'
    var_26 = {var_25}
    var_27 = module_0.Config()
    var_28 = {var_25}
    var_29 = module_0.Config()
    var_30 = 'link2'
    var_31 = module_0.Config()
    var_32 = 'fifo'
    var_33 = module_0.Config()
    var_34 = 'file.py'
    var_35 = module_0.Config()
    var_36 = 'file.py~'
    var_37 = module_0.Config()
    var_38 = True
    var_39 = module_0.Config()
    var_40 = '.git'
    var_41 = False
    var_42 = module_0.Config()



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile=black'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nprofile=black'
    var_5 = 'subdir'
    var_6 = '.isort.cfg'
    var_7 = 'profile'
    var_8 = 'empty_dir'
    var_9 = 'invalid_dir'
    var_10 = 'invalid.cfg'
    var_11 = 'invalid config data'



# Parsed testcases at query #41
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 120
    var_2 = 10
    var_3 = module_0.Config()
    var_4 = 100
    var_5 = 80
    var_6 = module_0.Config()
    var_7 = '[isort]\nline_length = 100\n'
    var_8 = 'pyproject.toml'
    var_9 = '[tool.isort]\nline_length = 110\n'
    var_10 = 'black'
    var_11 = module_0.Config()
    var_12 = 'invalid_profile'
    var_13 = module_0.Config()
    var_14 = 90
    var_15 = module_0._Config(line_length=var_14)
    var_16 = 95
    var_17 = module_0.Config(config=var_15)
    var_18 = 'src'
    var_19 = module_0.Config()
    var_20 = module_0.Config()
    var_21 = 'invalid_formatter'
    var_22 = module_0.Config()
    var_23 = True
    var_24 = module_0.Config()
    var_25 = 'value'
    var_26 = module_0.Config()
    var_27 = 'bar'
    var_28 = [var_27]
    var_29 = module_0.Config()
    var_30 = module_0.Config()
    var_31 = module_0.Config()



# Parsed testcases at query #42
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 120
    var_2 = 10
    var_3 = '\t'
    var_4 = module_0.Config()
    var_5 = 'nonexistent_file.py'
    var_6 = module_0.Config(var_5)
    var_7 = '/nonexistent/path'
    var_8 = module_0.Config(settings_path=var_7)
    var_9 = 'black'
    var_10 = module_0.Config()
    var_11 = 'nonexistent_profile'
    var_12 = module_0.Config()
    var_13 = 100
    var_14 = 5
    var_15 = module_0._Config(line_length=var_13, wrap_length=var_14)
    var_16 = module_0.Config(config=var_15)
    var_17 = 10
    var_18 = 5
    var_19 = module_0.Config()
    var_20 = False
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = 'value'
    var_24 = module_0.Config()
    var_25 = 'First Party'
    var_26 = 'Third Party'
    var_27 = module_0.Config()
    var_28 = 'custom'
    var_29 = 'custom_module'
    var_30 = {var_29}
    var_31 = {var_28: var_30}
    var_32 = module_0.Config()
    var_33 = {var_29}
    var_34 = frozenset(var_33)
    var_35 = {var_28: var_34}
    var_36 = 'src'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = var_38.src_paths
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = module_0.Config()
    var_42 = 'nonexistent_formatter'
    var_43 = module_0.Config()
    var_44 = 'natural'
    var_45 = module_0.Config()
    var_46 = 'nonexistent_sort'
    var_47 = module_0.Config()



# Parsed testcases at query #43
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    assert var_6 is False
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = 'other_*.py'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = module_0.Config()
    var_14 = var_13.is_skipped(var_0)
    assert var_14 is False
    var_15 = module_0.Config()
    var_16 = '_link'
    var_17 = var_0 + var_16
    assert var_17 is False
    var_18 = True
    var_19 = module_0.Config()
    var_20 = '.git'
    var_21 = var_0 / var_20
    var_22 = 'test.py'
    var_23 = False
    var_24 = module_0.Config()
    var_25 = '.git'
    var_26 = var_0 / var_25
    var_27 = 'test.py'
    var_28 = module_0.Config()
    var_29 = '.git'
    var_30 = var_0 / var_29
    var_31 = 'test.py'
    var_32 = module_0.Config()
    var_33 = '.git'
    var_34 = var_0 / var_33
    var_35 = 'test.py'
    var_36 = module_0.Config()
    var_37 = var_36.is_skipped(var_33)
    assert var_37 is True
    var_38 = module_0.Config()
    var_39 = 'test.fifo'
    var_40 = var_0 / var_39
    var_41 = var_38.is_skipped(var_40)
    assert var_41 is True
    var_42 = module_0.Config()
    var_43 = '/nonexistent/path'
    var_44 = {var_0}
    var_45 = module_0.Config()
    var_46 = {var_6}
    var_47 = module_0.Config()



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\n'
    var_4 = '.isort.cfg'
    var_5 = '[isort]\nline_length=100\n'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nmulti_line_output=3\n'
    var_8 = 'profile'
    var_9 = 'line_length'
    var_10 = 'multi_line_output'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nprofile=black'
    var_2 = 0
    var_3 = 'subdir'
    var_4 = 'setup.cfg'
    var_5 = '[tool.isort]\nprofile=black'
    var_6 = 'pyproject.toml'
    var_7 = 'invalid.cfg'
    var_8 = 'invalid content'
    var_9 = 'nested'



# Parsed testcases at query #46
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/repo'
    var_17 = '/repo/file1.py'
    var_18 = '/repo/file2.py'
    var_19 = '/repo/file3.py'
    var_20 = module_0.Config()
    var_21 = 'nonexistent.py'
    var_22 = module_0.Config()
    var_23 = 'file.py~'
    var_24 = '/project'
    var_25 = module_0.Config()
    var_26 = '/project/skip_me.py'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nprofile=black'
    var_6 = '.isort.cfg'
    var_7 = '[settings]\nindent=4'
    var_8 = 'pyproject.toml'
    var_9 = '[tool.isort]\nknown_first_party=mypackage'
    var_10 = 'setup.cfg'
    var_11 = 'invalid content'
    var_12 = len(var_3)
    assert var_12 == 0
    var_13 = '.isort.cfg'
    var_14 = '[settings]\nline_length=88'
    var_15 = 'setup.cfg'
    var_16 = '[isort]\nprofile=black'
    var_17 = len(var_12)
    assert var_17 == 1



# Parsed testcases at query #48
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'test_*'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = {var_5}
    var_9 = module_0.Config()
    var_10 = module_0.Config()
    var_11 = 'test_dir'
    var_12 = {var_11}
    var_13 = module_0.Config()
    var_14 = 'test_dir/test_file.py'
    var_15 = 'parent_dir'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = 'parent_dir/sub_dir/test_file.py'
    var_19 = '*/test_*'
    var_20 = {var_19}
    var_21 = module_0.Config()
    var_22 = 'dir/test_file.py'
    var_23 = True
    var_24 = module_0.Config()
    var_25 = '/test'
    var_26 = '/test/file1.py'
    var_27 = {var_26}
    var_28 = '/test/file2.py'
    var_29 = module_0.Config()
    var_30 = {var_26}
    var_31 = module_0.Config()
    var_32 = {var_26}
    var_33 = '/test/subdir'
    var_34 = module_0.Config()
    var_35 = '.git'
    var_36 = module_0.Config()
    var_37 = 'nonexistent_file.py'
    var_38 = module_0.Config()
    var_39 = 'fifo_file'
    var_40 = var_38.is_skipped(var_1)
    assert var_40 is True
    var_41 = module_0.Config()
    var_42 = 'test_file.py~'



# Parsed testcases at query #49
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'other_*'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = True
    var_13 = module_0.Config()
    var_14 = '/repo'
    var_15 = '/repo/committed_file.py'
    var_16 = {var_15}
    var_17 = '/repo/ignored_file.py'
    var_18 = module_0.Config()
    var_19 = {var_15}
    var_20 = module_0.Config()
    var_21 = '/some/directory'
    var_22 = module_0.Config()
    var_23 = '/nonexistent/file.py'
    var_24 = module_0.Config()
    var_25 = '/repo/.git'
    var_26 = {var_0}
    var_27 = 'another_file.py'
    var_28 = {var_27}
    var_29 = module_0.Config()



# Parsed testcases at query #50
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 120
    var_2 = 10
    var_3 = '\t'
    var_4 = module_0.Config()
    var_5 = 'nonexistent_file.py'
    var_6 = module_0.Config(var_5)
    var_7 = '/nonexistent/path'
    var_8 = module_0.Config(settings_path=var_7)
    var_9 = 100
    var_10 = 5
    var_11 = module_0._Config(line_length=var_9, wrap_length=var_10)
    var_12 = module_0.Config(config=var_11)
    var_13 = 'black'
    var_14 = module_0.Config()
    var_15 = 'nonexistent_profile'
    var_16 = module_0.Config()
    var_17 = True
    var_18 = module_0.Config()
    var_19 = 'value'
    var_20 = module_0.Config()
    var_21 = 'First Party'
    var_22 = 'Third Party'
    var_23 = module_0.Config()
    var_24 = 'custom_module'
    var_25 = 'CUSTOM'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = 'src'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = var_30.src_paths
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = 0
    var_34 = var_30.src_paths[var_33]
    var_35 = str(var_34)
    var_36 = module_0.Config()
    var_37 = 'nonexistent_formatter'
    var_38 = module_0.Config()
    var_39 = 'natural'
    var_40 = module_0.Config()
    var_41 = 'nonexistent_sort'
    var_42 = module_0.Config()



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'pyi'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyi'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is True
    var_8 = 'test.txt'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'nonexistent.py'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = b'#!/usr/bin/env python\n'
    var_15 = 'test_shebang'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is True
    var_17 = b"print('hello')\n"
    var_18 = 'test_no_shebang'
    var_19 = var_0.is_supported_filetype(var_18)
    assert var_19 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/repo'
    var_17 = '/repo/file1.py'
    var_18 = '/repo/file2.py'
    var_19 = '/repo/file3.py'
    var_20 = var_15.is_skipped(var_3)
    var_21 = module_0.Config()
    var_22 = 'nonexistent.py'
    var_23 = module_0.Config()
    var_24 = 'file.py~'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'setup.cfg'
    assert var_4 == 1
    var_5 = '[isort]\nprofile=black'
    var_6 = '.isort.cfg'
    var_7 = '[settings]\nindent=4'
    var_8 = 'pyproject.toml'
    var_9 = "[tool.isort]\nknown_third_party=['django']"
    var_10 = 'invalid.cfg'
    var_11 = 'invalid content'
    var_12 = 'setup.cfg'
    var_13 = '[isort]\nline_length=120'
    var_14 = 'invalid.cfg'
    var_15 = 'invalid content'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    assert var_5 is False
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = 'dir1'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'dir1/file.py'
    var_17 = {var_13}
    var_18 = module_0.Config()
    var_19 = 'dir2/file.py'
    var_20 = module_0.Config()
    var_21 = 'nonexistent_file.py'
    var_22 = module_0.Config()
    var_23 = '.'
    var_24 = module_0.Config()
    var_25 = '.link'
    var_26 = var_0 + var_25
    var_27 = True
    var_28 = module_0.Config()
    var_29 = '.git'
    var_30 = var_0 / var_29
    var_31 = 'info'
    var_32 = var_30 / var_31
    var_33 = var_30 / var_31
    var_34 = 'exclude'
    var_35 = var_33 / var_34
    var_36 = '*.pyc\n'
    var_37 = 'test.pyc'
    var_38 = ''
    var_39 = module_0.Config()
    var_40 = '.git'
    var_41 = var_0 / var_40
    var_42 = 'info'
    var_43 = var_41 / var_42
    var_44 = var_41 / var_42
    var_45 = 'exclude'
    var_46 = var_44 / var_45
    var_47 = '*.pyc\n'
    var_48 = 'test.py'
    var_49 = ''
    var_50 = {var_0}
    var_51 = module_0.Config()
    var_52 = {var_46}
    var_53 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.pyw'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is True
    var_5 = 'test.c'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is True
    var_7 = 'test.h'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is True
    var_9 = 'test.sopel'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.min.py'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.py~'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'nonexistent.py'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = '#!/usr/bin/env python3\n'
    var_18 = 'test_shebang.py'
    var_19 = var_0.is_supported_filetype(var_18)
    assert var_19 is True
    var_20 = "print('hello')\n"
    var_21 = 'test_no_shebang.py'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is True
    var_23 = 'hello\n'
    var_24 = 'test.txt'
    var_25 = var_0.is_supported_filetype(var_24)
    assert var_25 is False



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'other_file.py'
    var_13 = module_0.Config()
    var_14 = 'some_directory'
    var_15 = module_0.Config()
    var_16 = 'symlink_to_file'
    var_17 = True
    var_18 = module_0.Config()
    var_19 = '/some/path/file.py'
    var_20 = '/some/path/other_file.py'
    var_21 = module_0.Config()
    var_22 = {var_14}
    var_23 = module_0.Config()
    var_24 = 'some_directory/file.py'
    var_25 = 'other_directory'
    var_26 = {var_25}
    var_27 = module_0.Config()



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = b'#!/usr/bin/env python\n'
    var_8 = 'test_file'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is True
    var_10 = b"print('hello')\n"
    var_11 = var_0.is_supported_filetype(var_8)
    assert var_11 is False
    var_12 = 'non_existent_file.py'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = '/dev/null'
    var_15 = var_0.is_supported_filetype(var_14)
    assert var_15 is True



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/repo'
    var_17 = '/repo/tracked.py'
    var_18 = '/repo/untracked.py'
    var_19 = module_0.Config()
    var_20 = 'nonexistent.py'
    var_21 = 'txt'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = 'file.txt'
    var_25 = 'py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = module_0.Config()
    var_29 = 'file.py~'
    var_30 = '/absolute/path'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = '/absolute/path/file.py'
    var_34 = '/project'
    var_35 = module_0.Config()
    var_36 = '/project/skipped.py'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = 'nonexistent_path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = 100
    var_6 = module_0.Config()
    var_7 = 'black'
    var_8 = module_0.Config()
    var_9 = 'invalid_profile'
    var_10 = module_0.Config()
    var_11 = module_0.Config()
    var_12 = 120
    var_13 = module_0.Config(config=var_11)
    var_14 = 'value'
    var_15 = module_0.Config()
    var_16 = False
    var_17 = 'value'
    var_18 = module_0.Config()
    var_19 = 4
    var_20 = module_0.Config()
    var_21 = 'tab'
    var_22 = module_0.Config()
    var_23 = 'custom'
    var_24 = 'custom_module'
    var_25 = {var_24}
    var_26 = {var_23: var_25}
    var_27 = module_0.Config()
    var_28 = 'Custom Heading'
    var_29 = module_0.Config()
    var_30 = 'Custom Footer'
    var_31 = module_0.Config()
    var_32 = module_0.Config()
    var_33 = 'invalid_formatter'
    var_34 = module_0.Config()
    var_35 = 'natural'
    var_36 = module_0.Config()
    var_37 = 'invalid_sort'
    var_38 = module_0.Config()



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    assert var_3 is True
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = 'dir1'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'dir1/file.py'
    var_17 = {var_13}
    var_18 = module_0.Config()
    var_19 = 'dir2/file.py'
    var_20 = {var_13}
    var_21 = module_0.Config()
    var_22 = {var_13}
    var_23 = module_0.Config()
    var_24 = 'dir2'
    var_25 = True
    var_26 = module_0.Config()
    var_27 = '/test'
    var_28 = '/test/file1.py'
    var_29 = '/test/file2.py'
    var_30 = module_0.Config()
    var_31 = '/test'
    var_32 = '/test/file1.py'
    var_33 = var_30.is_skipped(var_29)
    assert var_33 is False
    var_34 = module_0.Config()
    var_35 = '.git'
    var_36 = module_0.Config()
    var_37 = 'file.py'
    var_38 = module_0.Config()
    var_39 = 'file.py~'
    var_40 = module_0.Config()
    var_41 = module_0.Config()
    var_42 = 'file.py'
    var_43 = var_41.is_skipped(var_32)
    assert var_43 is True
    var_44 = module_0.Config()
    var_45 = 'file.py'
    var_46 = var_44.is_skipped(var_32)
    assert var_46 is False
    var_47 = module_0.Config()
    var_48 = 'file.py'
    var_49 = var_47.is_skipped(var_32)
    assert var_49 is False
    var_50 = module_0.Config()
    var_51 = 'file.py'
    var_52 = var_50.is_skipped(var_32)
    assert var_52 is True



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[isort]\nline_length=120\n'
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = 'setup.cfg'
    var_7 = 'invalid config content'
    var_8 = 'setup.cfg'
    var_9 = '[isort]\nprofile=black\n'
    var_10 = '.isort.cfg'
    var_11 = '[isort]\nline_length=120\n'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile=black\n'
    var_3 = 'dir2'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nline_length=120\n'
    var_6 = 'subdir'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nmulti_line_output=3\n'
    var_9 = 'profile'
    var_10 = 'line_length'
    var_11 = 'multi_line_output'
    var_12 = 'empty'
    var_13 = 'invalid'
    var_14 = 'invalid.cfg'
    var_15 = 'invalid content'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=120\n'
    var_3 = 'dir2'
    var_4 = 'subdir'
    var_5 = True
    var_6 = 'setup.cfg'
    var_7 = '[tool.isort]\nprofile=black\n'
    var_8 = 'not_a_config.txt'
    var_9 = 'line_length=88\n'
    var_10 = 'line_length'
    var_11 = 'profile'
    var_12 = 'empty'
    var_13 = 'does_not_exist'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 120
    var_3 = module_0.Config()
    var_4 = 'nonexistent_file.py'
    var_5 = module_0.Config(var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.Config(settings_path=var_6)
    var_8 = 100
    var_9 = 50
    var_10 = module_0._Config(line_length=var_8, wrap_length=var_9)
    var_11 = module_0.Config(config=var_10)
    var_12 = 'black'
    var_13 = module_0.Config()
    var_14 = 'nonexistent_profile'
    var_15 = module_0.Config()
    var_16 = 'value'
    var_17 = module_0.Config()
    var_18 = False
    var_19 = 'value'
    var_20 = module_0.Config()
    var_21 = '4'
    var_22 = module_0.Config()
    var_23 = 'tab'
    var_24 = module_0.Config()
    var_25 = 'custom'
    var_26 = 'custom_module'
    var_27 = {var_26}
    var_28 = {var_25: var_27}
    var_29 = module_0.Config()
    var_30 = {var_26}
    var_31 = frozenset(var_30)
    var_32 = {var_25: var_31}
    var_33 = 'Custom Heading'
    var_34 = module_0.Config()
    var_35 = 'Custom Footer'
    var_36 = module_0.Config()
    var_37 = module_0.Config()
    var_38 = 'nonexistent_formatter'
    var_39 = module_0.Config()
    var_40 = 'natural'
    var_41 = module_0.Config()
    var_42 = 'nonexistent_sort_order'
    var_43 = module_0.Config()
    var_44 = 100
    var_45 = 50
    var_46 = module_0.Config()



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'config1.py'
    var_1 = "isort_config = {'profile': 'black'}"
    var_2 = 'subdir'
    var_3 = 'config2.py'
    var_4 = "isort_config = {'line_length': 120}"
    var_5 = 'not_config.txt'
    var_6 = 'some content'
    var_7 = 'empty'
    var_8 = 'invalid.py'
    var_9 = 'invalid python syntax'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = 'path\\to\\file.py'
    var_5 = {var_4}
    var_6 = module_0.Config()
    var_7 = 'path/to/file.py'
    var_8 = '*.tmp'
    var_9 = {var_8}
    var_10 = module_0.Config()
    var_11 = 'test.tmp'
    var_12 = 'test.py'
    var_13 = 'test_dir'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'test_dir/file.py'
    var_17 = 'other_dir/file.py'
    var_18 = module_0.Config()
    var_19 = 'nonexistent_file.py'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '.git'
    var_23 = 'git_root\n'
    var_24 = 'file1.py\x00file2.py\x00'
    var_25 = 'file3.py\x00'
    var_26 = True
    var_27 = module_0.Config()
    var_28 = 'git_root/file1.py'
    var_29 = 'git_root/file3.py'
    var_30 = var_27.is_skipped(var_5)
    var_31 = 'file1.py'
    var_32 = {var_31}
    var_33 = 'file2.py'
    var_34 = {var_33}
    var_35 = {var_8}
    var_36 = '*.log'
    var_37 = {var_36}
    var_38 = module_0.Config()
    var_39 = 'test.log'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.py'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is True
    var_11 = 'test'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = '/nonexistent/path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = module_0._Config()
    var_6 = module_0.Config(config=var_5)
    var_7 = True
    var_8 = 120
    var_9 = module_0.Config()
    var_10 = 'nonexistent_profile'
    var_11 = module_0.Config()
    var_12 = 100
    var_13 = 80
    var_14 = module_0._Config(line_length=var_13, wrap_length=var_12)
    var_15 = module_0.Config()
    var_16 = module_0.Config()
    var_17 = hash(var_15)
    var_18 = id(var_15)
    var_19 = hash(var_16)
    var_20 = id(var_16)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'subdir1'
    var_3 = '.isort.cfg'
    var_4 = '[isort]\nline_length=88'
    var_5 = 'subdir2'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nmulti_line_output=3'
    var_8 = 'invalid.cfg'
    var_9 = 'invalid content'
    var_10 = 'nested'
    var_11 = "[isort]\nindent='    '"



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '\t'
    var_2 = 120
    var_3 = module_0.Config()
    var_4 = 100
    var_5 = 80
    var_6 = module_0.Config()
    var_7 = 'nonexistent_file.py'
    var_8 = module_0.Config(var_7)
    var_9 = '/nonexistent/path'
    var_10 = module_0.Config(settings_path=var_9)
    var_11 = 'nonexistent_profile'
    var_12 = module_0.Config()
    var_13 = module_0._Config()
    var_14 = module_0.Config(config=var_13)
    var_15 = True
    var_16 = module_0.Config()
    var_17 = 'value'
    var_18 = module_0.Config()
    var_19 = 'Future Imports'
    var_20 = 'End Future'
    var_21 = module_0.Config()
    var_22 = 'custom'
    var_23 = [var_22]
    var_24 = 'CUSTOM'
    var_25 = [var_24]
    var_26 = module_0.Config()



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = 'dir1'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'dir1/file.py'
    var_17 = {var_13}
    var_18 = module_0.Config()
    var_19 = 'dir2/file.py'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '/test/file1.py'
    var_23 = '/test/file2.py'
    var_24 = var_21.is_skipped(var_3)
    assert var_24 is False
    var_25 = False
    var_26 = module_0.Config()
    var_27 = '/test/file1.py'
    var_28 = '/test/file2.py'
    var_29 = module_0.Config()
    var_30 = 'dir1'
    var_31 = var_29.is_skipped(var_28)
    assert var_31 is False
    var_32 = module_0.Config()
    var_33 = 'link.py'
    var_34 = var_32.is_skipped(var_28)
    assert var_34 is False
    var_35 = module_0.Config()
    var_36 = 'nonexistent.py'
    var_37 = var_35.is_skipped(var_28)
    assert var_37 is True
    var_38 = module_0.Config()
    var_39 = 'file.py~'
    var_40 = module_0.Config()
    var_41 = 'fifo'
    var_42 = var_40.is_skipped(var_28)
    assert var_42 is True
    var_43 = module_0.Config()
    var_44 = 'file.unsupported'
    var_45 = var_43.is_skipped(var_28)
    assert var_45 is False



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = 'py3'
    var_2 = 'auto'
    var_3 = module_0._Config(var_2)
    var_4 = 'invalid'
    var_5 = module_0._Config(var_4)
    var_6 = True
    var_7 = module_0._Config(force_alphabetical_sort=var_6)
    var_8 = 80
    var_9 = 79
    var_10 = module_0._Config(line_length=var_9, wrap_length=var_8)
    var_11 = frozenset()
    var_12 = module_0._Config(known_standard_library=var_11)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=120'
    var_2 = 'subdir1'
    var_3 = 'setup.cfg'
    var_4 = '[isort]\nprofile=black'
    var_5 = 'subdir2'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nmulti_line_output=3'
    var_8 = 'nested'
    var_9 = "[settings]\nindent='    '"
    var_10 = 'invalid.cfg'
    var_11 = 'invalid content'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 100
    var_3 = module_0.Config()
    var_4 = 120
    var_5 = 100
    var_6 = module_0.Config()
    var_7 = 'nonexistent_file.py'
    var_8 = module_0.Config(var_7)
    var_9 = '/nonexistent/path'
    var_10 = module_0.Config(settings_path=var_9)
    var_11 = 'nonexistent_profile'
    var_12 = module_0.Config()
    var_13 = module_0._Config()
    var_14 = module_0.Config(config=var_13)
    var_15 = 'value'
    var_16 = module_0.Config()
    var_17 = True
    var_18 = module_0.Config()
    var_19 = 'custom_module'
    var_20 = module_0.Config()
    var_21 = 'Custom Heading'
    var_22 = module_0.Config()
    var_23 = 'Custom Footer'
    var_24 = module_0.Config()
    var_25 = 'nonexistent_formatter'
    var_26 = module_0.Config()
    var_27 = 'nonexistent_sort'
    var_28 = module_0.Config()



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'nonexistent.py'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = '#!/usr/bin/env python\n'
    var_10 = 'test_with_shebang'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is True
    var_12 = "print('hello')\n"
    var_13 = 'test_no_shebang'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = module_0.Config()
    var_6 = '*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = {var_6}
    var_10 = module_0.Config()
    var_11 = module_0.Config()
    var_12 = 'test.txt'
    var_13 = module_0.Config()
    var_14 = 'test_dir'
    var_15 = module_0.Config()
    var_16 = 'test_link'
    var_17 = module_0.Config()
    var_18 = 'test_nonexistent'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = '.git'
    var_22 = module_0.Config()



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'config1.py'
    var_1 = "setting1 = 'value1'"
    var_2 = 'subdir'
    var_3 = 'config2.py'
    var_4 = "setting2 = 'value2'"
    var_5 = 'non_config.txt'
    var_6 = 'not a config'

def test_case_0():
    var_0 = 'invalid.py'
    var_1 = 'invalid python syntax @#$%'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'config_dir'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile = black'
    var_3 = 'sub_dir'
    var_4 = '.isort.cfg'
    var_5 = 'profile = black'
    var_6 = 'empty_dir'
    var_7 = 'no_config'
    var_8 = 'test.py'
    var_9 = "print('hello')"



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 120
    var_3 = module_0.Config()
    var_4 = 100
    var_5 = 50
    var_6 = module_0.Config()
    var_7 = 'nonexistent_file.cfg'
    var_8 = module_0.Config(var_7)
    var_9 = '/nonexistent/path'
    var_10 = module_0.Config(settings_path=var_9)
    var_11 = module_0._Config()
    var_12 = 100
    var_13 = module_0.Config(config=var_11)
    var_14 = 'black'
    var_15 = module_0.Config()
    var_16 = 'nonexistent_profile'
    var_17 = module_0.Config()
    var_18 = 'value'
    var_19 = module_0.Config()
    var_20 = True
    var_21 = module_0.Config()



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'pyproject.toml'
    var_2 = "[tool.isort]\nprofile = 'black'"
    var_3 = 'dir2'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nline_length = 88'
    var_6 = 'empty_dir'

def test_case_0():
    var_0 = 'empty'

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'pyproject.toml'
    var_2 = 'invalid toml content'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = True
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nindent = 4'



# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = 'nonexistent_path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = module_0._Config()
    var_6 = module_0.Config(config=var_5)
    var_7 = True
    var_8 = module_0.Config()
    var_9 = 'nonexistent_profile'
    var_10 = module_0.Config()
    var_11 = 'value'
    var_12 = module_0.Config()
    var_13 = 'value'
    var_14 = module_0.Config()
    var_15 = '4'
    var_16 = module_0.Config()
    var_17 = 'tab'
    var_18 = module_0.Config()
    var_19 = 'value'
    var_20 = module_0.Config()
    var_21 = 'First Party'
    var_22 = module_0.Config()
    var_23 = 'First Party Footer'
    var_24 = module_0.Config()
    var_25 = 'nonexistent_formatter'
    var_26 = module_0.Config()
    var_27 = 'natural'
    var_28 = module_0.Config()
    var_29 = 'nonexistent_sort_order'
    var_30 = module_0.Config()



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 0
    var_3 = 'subdir'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.isort]\nprofile="black"'
    var_6 = '.isort.cfg'
    var_7 = 'profile=black'
    var_8 = 'invalid.cfg'
    var_9 = 'invalid content'
    var_10 = 'tox.ini'



# Parsed testcases at query #34
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.isort]\nprofile=black'
    var_4 = 'subdir'
    var_5 = '.isort.cfg'
    var_6 = 'isort.settings._get_config_data'
    var_7 = 'profile'
    var_8 = 'black'
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = lambda path, _: var_9 if path.exists() else var_10
    var_12 = 'isort.settings.Trie.insert'
    var_13 = {var_7: var_8}
    var_14 = {var_7: var_8}
    var_15 = {var_7: var_8}
    var_16 = 'empty'
    var_17 = 'invalid.cfg'
    var_18 = 'invalid content'
    var_19 = 'Invalid config'
    var_20 = module_0.find_all_configs(var_0)



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = module_0.Config(var_1)
    var_3 = '/nonexistent/path'
    var_4 = module_0.Config(settings_path=var_3)
    var_5 = module_0._Config()
    var_6 = module_0.Config(config=var_5)
    var_7 = True
    var_8 = 100
    var_9 = module_0.Config()
    var_10 = 'nonexistent_profile'
    var_11 = module_0.Config()
    var_12 = 'value'
    var_13 = module_0.Config()
    var_14 = False
    var_15 = 'value'
    var_16 = module_0.Config()
    var_17 = 'custom_module'
    var_18 = [var_17]
    var_19 = module_0.Config()
    var_20 = 'custom_section'
    var_21 = ()
    var_22 = 'Custom Heading'
    var_23 = module_0.Config()
    var_24 = 'Custom Footer'
    var_25 = module_0.Config()
    var_26 = 'nonexistent_formatter'
    var_27 = module_0.Config()
    var_28 = 'nonexistent_sort_order'
    var_29 = module_0.Config()



# Parsed testcases at query #36
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = '*.tmp'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'file.tmp'
    var_7 = module_0.Config()
    var_8 = 'normal_file.py'
    var_9 = True
    var_10 = module_0.Config()
    var_11 = '/repo'
    var_12 = '/repo/allowed.py'
    var_13 = {var_12}
    var_14 = '/repo/ignored.py'
    var_15 = 'skip_dir'
    var_16 = {var_15}
    var_17 = module_0.Config()
    var_18 = 'skip_dir/file.py'
    var_19 = 'other_dir'
    var_20 = {var_19}
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = '/repo/tracked.py'
    var_24 = {var_23}
    var_25 = '/repo/untracked.py'
    var_26 = False
    var_27 = module_0.Config()
    var_28 = {var_23}
    var_29 = module_0.Config()
    var_30 = 'directory'
    var_31 = module_0.Config()
    var_32 = 'symlink'
    var_33 = var_1 / var_32
    var_34 = var_31.is_skipped(var_33)
    assert var_34 is True



# Parsed testcases at query #37
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'txt'
    var_4 = 'test.txt'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = 'test.py~'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.py'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is True
    var_12 = 'test.py'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'test.py'
    var_15 = var_0.is_supported_filetype(var_14)
    assert var_15 is False



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile=black\n'
    var_3 = 'dir2'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nline_length=120\n'
    var_6 = 'empty_dir'
    var_7 = 'nonexistent.cfg'
    var_8 = 'no_configs'



# Parsed testcases at query #39
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    assert var_3 is True
    assert var_3 is False
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = 'other_*.py'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = 'test_dir'
    var_14 = {var_13}
    var_15 = module_0.Config()
    var_16 = 'test_dir/file.py'
    var_17 = 'other_dir'
    var_18 = {var_17}
    var_19 = module_0.Config()
    var_20 = {var_13}
    var_21 = module_0.Config()
    var_22 = {var_17}
    var_23 = module_0.Config()
    var_24 = 'test_link'
    var_25 = {var_24}
    var_26 = module_0.Config()
    var_27 = 'test_link'
    var_28 = 'other_link'
    var_29 = {var_28}
    var_30 = module_0.Config()
    var_31 = 'test_link'
    var_32 = module_0.Config()
    var_33 = module_0.Config()
    var_34 = var_33.is_skipped(var_31)
    assert var_34 is False
    var_35 = module_0.Config()
    var_36 = 'test.py~'
    var_37 = module_0.Config()
    var_38 = True
    var_39 = module_0.Config()
    var_40 = '.git'
    var_41 = var_0 / var_40
    var_42 = 'info'
    var_43 = var_41 / var_42
    var_44 = 'exclude'
    var_45 = var_43 / var_44
    var_46 = var_41 / var_42
    var_47 = var_46 / var_44
    var_48 = 'test.py\n'
    var_49 = 'test.py'
    var_50 = '# test'
    var_51 = module_0.Config()
    var_52 = '.git'
    var_53 = var_0 / var_52
    var_54 = 'info'
    var_55 = var_53 / var_54
    var_56 = 'exclude'
    var_57 = var_55 / var_56
    var_58 = var_53 / var_54
    var_59 = var_58 / var_56
    var_60 = 'other.py\n'
    var_61 = 'test.py'
    var_62 = '# test'



# Parsed testcases at query #40
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = 'dir/'
    var_15 = module_0.Config()
    var_16 = 'test.py'
    var_17 = var_0 / var_16
    var_18 = 'link.py'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = 'test.py'
    var_22 = var_0 / var_21
    var_23 = var_20.is_skipped(var_22)
    assert var_23 is True
    var_24 = False
    var_25 = module_0.Config()
    var_26 = 'test.py'
    var_27 = var_0 / var_26
    var_28 = var_25.is_skipped(var_27)
    assert var_28 is False
    var_29 = 'dir1'
    var_30 = {var_29}
    var_31 = module_0.Config()
    var_32 = 'dir1/file.py'
    var_33 = {var_29}
    var_34 = module_0.Config()
    var_35 = 'dir2/file.py'



# Parsed testcases at query #41
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'tests/test.py'
    var_8 = 'src/test.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = '/project'
    var_15 = 'subdir'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = '/project/subdir/file.py'
    var_19 = '/other/subdir/file.py'
    var_20 = module_0.Config()
    var_21 = 'nonexistent.py'
    var_22 = True
    var_23 = module_0.Config()
    var_24 = '/git_root'
    var_25 = '/git_root/tracked.py'
    var_26 = '/git_root/untracked.py'
    var_27 = module_0.Config()
    var_28 = '.git'
    var_29 = module_0.Config()
    var_30 = 'file.py~'
    var_31 = '.blocked'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = 'file.blocked'
    var_35 = '.custom'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = 'file.custom'
    var_39 = 'skip.py'
    var_40 = [var_39]
    var_41 = '*.glob'
    var_42 = [var_41]
    var_43 = module_0.Config()
    var_44 = 'test.glob'



# Parsed testcases at query #42
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = 'test_dir'
    var_5 = {var_4}
    var_6 = module_0.Config()
    var_7 = 'test_dir/file.py'
    var_8 = 'other_dir/file.py'
    var_9 = '*.tmp'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/repo/tracked.py'
    var_17 = '/repo/untracked.py'
    var_18 = module_0.Config()
    var_19 = 'nonexistent_file.py'
    var_20 = module_0.Config()
    var_21 = 'file.py~'
    var_22 = 'txt'
    var_23 = {var_22}
    var_24 = module_0.Config()
    var_25 = 'file.txt'
    var_26 = var_24.is_supported_filetype(var_25)
    assert var_26 is False
    var_27 = 'py'
    var_28 = 'js'
    var_29 = {var_27, var_28}
    var_30 = module_0.Config()
    var_31 = 'file.js'
    var_32 = 'test_dir/subdir/file.py'
    var_33 = '/project'
    var_34 = 'subdir'
    var_35 = {var_34}
    var_36 = module_0.Config()
    var_37 = '/project/subdir/file.py'
    var_38 = '/other/subdir/file.py'
    var_39 = 'file1.py'
    var_40 = {var_39}
    var_41 = 'file2.py'
    var_42 = {var_41}
    var_43 = module_0.Config()
    var_44 = 'file3.py'
    var_45 = {var_9}
    var_46 = '*.bak'
    var_47 = {var_46}
    var_48 = module_0.Config()
    var_49 = 'file.bak'



# Parsed testcases at query #43
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'txt'
    var_4 = 'test.txt'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = 'test.py~'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'nonexistent.py'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = '#!/usr/bin/env python\n'
    var_11 = var_0.is_supported_filetype(var_4)
    assert var_11 is True
    var_12 = "print('hello')\n"
    var_13 = var_0.is_supported_filetype(var_4)
    assert var_13 is True
    var_14 = '#!/usr/bin/env python\n'
    var_15 = var_0.is_supported_filetype(var_4)
    assert var_15 is True
    var_16 = "print('hello')\n"
    var_17 = var_0.is_supported_filetype(var_4)
    assert var_17 is False



# Parsed testcases at query #44
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'other_*'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = module_0.Config()
    var_13 = 'some_directory'
    var_14 = module_0.Config()
    var_15 = 'symlink_file.py'
    var_16 = var_14.is_skipped(var_1)
    assert var_16 is True
    var_17 = module_0.Config()
    var_18 = 'non_existent_file.py'
    var_19 = var_17.is_skipped(var_1)
    assert var_19 is True
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '/git_folder'
    var_23 = '/git_folder/committed_file.py'
    var_24 = {var_23}
    var_25 = '/git_folder/unstaged_file.py'
    var_26 = module_0.Config()
    var_27 = {var_23}
    var_28 = module_0.Config()
    var_29 = '.git'



# Parsed testcases at query #45
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other_file.py'
    var_4 = {var_3}
    var_5 = module_0.Config()
    var_6 = 'test_*'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'other_*'
    var_10 = {var_9}
    var_11 = module_0.Config()
    var_12 = 'test_dir'
    var_13 = {var_12}
    var_14 = module_0.Config()
    var_15 = 'test_dir/test_file.py'
    var_16 = 'other_dir'
    var_17 = {var_16}
    var_18 = module_0.Config()
    var_19 = module_0.Config()
    var_20 = True
    var_21 = module_0.Config()
    var_22 = 'non_existent_file.py'
    var_23 = module_0.Config()
    var_24 = 'symlink.py'
    var_25 = module_0.Config()
    var_26 = 'fifo_file.py'
    var_27 = module_0.Config()
    var_28 = 'test_file.py~'
    var_29 = module_0.Config()
    var_30 = '.git'
    var_31 = module_0.Config()
    var_32 = False
    var_33 = module_0.Config()



# Parsed testcases at query #46
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = {var_0}
    assert var_3 is True
    var_4 = module_0.Config()
    var_5 = 'file2.py'
    var_6 = '*.txt'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = 'test.py'
    var_13 = module_0.Config()
    var_14 = 'some_directory'
    var_15 = True
    var_16 = module_0.Config()
    var_17 = '/some/git/folder'
    var_18 = '/some/git/folder/file1.py'
    var_19 = '/some/git/folder/file2.py'
    var_20 = module_0.Config()
    var_21 = '/some/git/folder'
    var_22 = '/some/git/folder/file1.py'
    var_23 = var_20.is_skipped(var_19)
    assert var_23 is False
    var_24 = 'dir1'
    var_25 = {var_24}
    var_26 = module_0.Config()
    var_27 = 'dir1/file.py'
    var_28 = {var_24}
    var_29 = module_0.Config()
    var_30 = 'dir2/file.py'
    var_31 = module_0.Config()
    var_32 = 'file.py~'



# Parsed testcases at query #47
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 120
    var_2 = 10
    var_3 = '\t'
    var_4 = module_0.Config()
    var_5 = 'nonexistent_file.py'
    var_6 = module_0.Config(var_5)
    var_7 = '/nonexistent/path'
    var_8 = module_0.Config(settings_path=var_7)
    var_9 = 100
    var_10 = 8
    var_11 = module_0._Config(line_length=var_9, wrap_length=var_10)
    var_12 = module_0.Config(config=var_11)
    var_13 = module_0._Config(line_length=var_9, wrap_length=var_10)
    var_14 = module_0.Config(config=var_13)
    var_15 = 'black'
    var_16 = module_0.Config()
    var_17 = 'invalid_profile'
    var_18 = module_0.Config()
    var_19 = 10
    var_20 = 5
    var_21 = module_0.Config()
    var_22 = 'value'
    var_23 = module_0.Config()
    var_24 = True
    var_25 = module_0.Config()



# Parsed testcases at query #48
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = {var_3}
    assert var_4 is False
    var_5 = module_0.Config()
    var_6 = 'test_*.py'
    var_7 = {var_6}
    var_8 = module_0.Config()
    var_9 = 'test_file.py'
    var_10 = 'other_*.py'
    var_11 = {var_10}
    var_12 = module_0.Config()
    var_13 = module_0.Config()
    var_14 = 'some_directory'
    var_15 = module_0.Config()
    var_16 = '_link'
    var_17 = var_0 + var_16
    var_18 = True
    var_19 = module_0.Config()
    var_20 = '.git'
    var_21 = var_0 / var_20
    var_22 = 'test.py'
    var_23 = False
    var_24 = module_0.Config()
    var_25 = '.git'
    var_26 = var_0 / var_25
    var_27 = 'test.py'
    var_28 = 'skipped_dir'
    var_29 = {var_28}
    var_30 = module_0.Config()
    var_31 = 'skipped_dir'
    var_32 = var_0 / var_31
    var_33 = 'test.py'
    var_34 = var_32 / var_33
    var_35 = var_30.is_skipped(var_34)
    assert var_35 is True
    var_36 = 'other_dir'
    var_37 = {var_36}
    var_38 = module_0.Config()
    var_39 = 'test_dir'
    var_40 = var_0 / var_39
    var_41 = 'test.py'
    var_42 = var_40 / var_41
    var_43 = var_38.is_skipped(var_42)
    assert var_43 is False



# Parsed testcases at query #49
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = "[isort]\nline_length = 100\nindent = '\\t'\n"
    var_2 = 'pyproject.toml'
    var_3 = '[tool.isort]\nline_length = 120\n'
    var_4 = 80
    var_5 = '  '
    var_6 = module_0.Config()
    var_7 = '/nonexistent/path'
    var_8 = module_0.Config(settings_path=var_7)
    var_9 = 'nonexistent_profile'
    var_10 = module_0.Config()
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'force_single_line'
    var_14 = hasattr(var_12, var_13)
    var_15 = 'value'
    var_16 = module_0.Config()
    var_17 = 'custom_module'
    var_18 = [var_17]
    var_19 = module_0.Config()
    var_20 = 'Custom Heading'
    var_21 = module_0.Config()
    var_22 = 'Custom Footer'
    var_23 = module_0.Config()
    var_24 = 'natural'
    var_25 = module_0.Config()
    var_26 = 'invalid'
    var_27 = module_0.Config()
    var_28 = 'black'
    var_29 = module_0.Config()
    var_30 = 'invalid_formatter'
    var_31 = module_0.Config()
    var_32 = module_0.Config()
    var_33 = 10
    var_34 = 5
    var_35 = module_0.Config()



# Parsed testcases at query #50
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = 'test_dir'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'test_dir/file.py'
    var_8 = 'other_dir/file.py'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'file.tmp'
    var_13 = 'file.py'
    var_14 = True
    var_15 = module_0.Config()
    var_16 = '/git_root'
    var_17 = '/git_root/tracked.py'
    var_18 = '/git_root/untracked.py'
    var_19 = module_0.Config()
    var_20 = 'nonexistent.py'
    var_21 = module_0.Config()
    var_22 = 'file.py~'
    var_23 = '/project'
    var_24 = module_0.Config()
    var_25 = '/project/skip_me.py'



