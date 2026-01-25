####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_15 = 'nonexistent.py'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = 100
    var_3 = '\t'
    var_4 = module_0.Config()
    var_5 = 100
    var_6 = 50
    var_7 = module_0.Config()
    var_8 = 'nonexistent_file.py'
    var_9 = module_0.Config(var_8)
    var_10 = '/nonexistent/path'
    var_11 = module_0.Config(settings_path=var_10)
    var_12 = 'nonexistent_profile'
    var_13 = module_0.Config()
    var_14 = module_0._Config()
    var_15 = module_0.Config(config=var_14)
    var_16 = 'value'
    var_17 = module_0.Config()
    var_18 = False
    var_19 = 'value'
    var_20 = module_0.Config()
    var_21 = 'custom'
    var_22 = 'custom_module'
    var_23 = {var_22}
    var_24 = {var_21: var_23}
    var_25 = module_0.Config()
    var_26 = 'Custom Heading'
    var_27 = module_0.Config()
    var_28 = 'Custom Footer'
    var_29 = module_0.Config()
    var_30 = 'src'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = var_32.src_paths
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 0
    var_36 = var_32.src_paths[var_35]
    var_37 = str(var_36)
    var_38 = 'nonexistent_formatter'
    var_39 = module_0.Config()
    var_40 = 'natural'
    var_41 = module_0.Config()
    var_42 = 'nonexistent_sort_order'
    var_43 = module_0.Config()



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black'
    var_2 = 'subdir'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nprofile="black"'
    var_5 = '.isort.cfg'
    var_6 = 'profile=black'
    var_7 = 'invalid.cfg'
    var_8 = 'invalid content'
    var_9 = 'nested'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'config1.py'
    var_1 = "setting1 = 'value1'"
    var_2 = 'subdir'
    var_3 = 'config2.py'
    var_4 = "setting2 = 'value2'"
    var_5 = 'nested'
    var_6 = 'config3.py'
    var_7 = True
    var_8 = "setting3 = 'value3'"
    var_9 = 'not_config.txt'
    var_10 = 'not a config'
    var_11 = 'empty'
    var_12 = 'no_config'
    var_13 = 'file.txt'
    var_14 = 'content'



# Parsed testcases at query #5
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
    var_9 = 'black'
    var_10 = module_0.Config()
    var_11 = 'invalid_profile'
    var_12 = module_0.Config()
    var_13 = 'value'
    var_14 = module_0.Config()
    var_15 = 'deprecated_option'
    var_16 = module_0.Config()
    var_17 = 'custom_section'
    var_18 = 'custom_module'
    var_19 = {var_18}
    var_20 = {var_17: var_19}
    var_21 = module_0.Config()
    var_22 = {var_18}
    var_23 = 'CUSTOM_SECTION'
    var_24 = (var_23,)
    var_25 = module_0.Config()
    var_26 = 'Custom Section'
    var_27 = 'Custom Footer'
    var_28 = module_0.Config()
    var_29 = module_0.Config()
    var_30 = 'invalid_formatter'
    var_31 = module_0.Config()
    var_32 = 'natural'
    var_33 = module_0.Config()
    var_34 = 'invalid_sort_order'
    var_35 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = '*.txt'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = 'test.txt'
    var_10 = [var_6]
    var_11 = module_0.Config()
    var_12 = 'skip_dir'
    var_13 = [var_12]
    var_14 = module_0.Config()
    var_15 = 'skip_dir/test.py'
    var_16 = [var_12]
    var_17 = module_0.Config()
    var_18 = 'other_dir/test.py'
    var_19 = module_0.Config()
    var_20 = 'test_dir'
    var_21 = True
    var_22 = module_0.Config()
    var_23 = 'nonexistent.py'
    var_24 = module_0.Config()
    var_25 = 'test_file.py'
    var_26 = 'symlink.py'
    var_27 = module_0.Config()
    var_28 = '.git'
    var_29 = False
    var_30 = module_0.Config()



# Parsed testcases at query #7
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
    var_10 = var_0.is_supported_filetype(var_4)
    assert var_10 is True
    var_11 = "print('hello')\n"
    var_12 = var_0.is_supported_filetype(var_4)
    assert var_12 is True
    var_13 = 'test.txt'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False



# Parsed testcases at query #8
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
    var_8 = 'profile'
    var_9 = 'line_length'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'test.py'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'txt'
    var_5 = 'test.txt'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.py~'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'nonexistent.py'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = '#!/usr/bin/env python\n'
    var_12 = 'test_script'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is True
    var_14 = "print('hello')\n"
    var_15 = 'test_script_no_shebang'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '38'
    var_1 = module_0._Config(var_0)
    var_2 = 'auto'
    var_3 = module_0._Config(var_2)
    var_4 = 'invalid'
    var_5 = module_0._Config(var_4)
    var_6 = True
    var_7 = module_0._Config(force_alphabetical_sort=var_6)
    var_8 = 100
    var_9 = 79
    var_10 = module_0._Config(line_length=var_9, wrap_length=var_8)
    var_11 = module_0._Config(var_8)
    var_12 = 'py38'
    var_13 = module_0._Config()
    var_14 = module_0._Config()
    var_15 = hash(var_13)
    var_16 = id(var_13)
    var_17 = hash(var_14)
    var_18 = id(var_14)



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'
    var_2 = '[isort]\nline_length=120\n'
    var_3 = module_0.Config(var_1)
    var_4 = 'pyproject.toml'
    var_5 = '[tool.isort]\nline_length=120\n'
    var_6 = 100
    var_7 = module_0._Config(line_length=var_6)
    var_8 = module_0.Config(config=var_7)
    var_9 = module_0.Config()
    var_10 = 100
    var_11 = 80
    var_12 = module_0.Config()
    var_13 = 'black'
    var_14 = module_0.Config()
    var_15 = 'nonexistent'
    var_16 = module_0.Config()
    var_17 = '/nonexistent/path'
    var_18 = module_0.Config(settings_path=var_17)
    var_19 = 2
    var_20 = module_0.Config()
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = module_0.Config()
    var_24 = 'nonexistent'
    var_25 = module_0.Config()
    var_26 = 'natural'
    var_27 = module_0.Config()
    var_28 = 'nonexistent'
    var_29 = module_0.Config()



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 100
    var_2 = 10
    var_3 = '\t'
    var_4 = module_0.Config()
    var_5 = 'test.ini'
    var_6 = module_0.Config(var_5)
    var_7 = '/test/path'
    var_8 = module_0.Config(settings_path=var_7)
    var_9 = 90
    var_10 = 5
    var_11 = module_0._Config(line_length=var_9, wrap_length=var_10)
    var_12 = 95
    var_13 = module_0.Config(config=var_11)
    var_14 = 'black'
    var_15 = module_0.Config()
    var_16 = 'nonexistent'
    var_17 = module_0.Config()
    var_18 = '/invalid/path'
    var_19 = module_0.Config(settings_path=var_18)
    var_20 = 'value'
    var_21 = module_0.Config()
    var_22 = True
    var_23 = module_0.Config()
    var_24 = 10
    var_25 = 5
    var_26 = module_0.Config()
    var_27 = '4'
    var_28 = module_0.Config()
    var_29 = 'tab'
    var_30 = module_0.Config()
    var_31 = '    '
    var_32 = module_0.Config()
    var_33 = 'bar'
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = 'qux'
    var_37 = module_0.Config()
    var_38 = 'src'
    var_39 = 'tests'
    var_40 = [var_38, var_39]
    var_41 = module_0.Config()
    var_42 = var_41.src_paths
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = var_41.src_paths
    var_45 = all(var_27)
    var_46 = 'test_formatter'
    var_47 = module_0.Config()
    var_48 = 'nonexistent'
    var_49 = module_0.Config()
    var_50 = 'natural'
    var_51 = module_0.Config()
    var_52 = 'native'
    var_53 = module_0.Config()
    var_54 = 'nonexistent'
    var_55 = module_0.Config()



# Parsed testcases at query #14
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
    var_13 = True
    var_14 = module_0.Config()
    var_15 = '/test/file1.py'
    var_16 = '/test/file2.py'
    var_17 = module_0.Config()
    var_18 = module_0.Config()
    var_19 = '/test/directory'
    var_20 = module_0.Config()
    var_21 = '/test/symlink'
    var_22 = var_20.is_skipped(var_1)
    var_23 = module_0.Config()
    var_24 = '/test/nonexistent.py'
    var_25 = var_23.is_skipped(var_1)
    var_26 = module_0.Config()
    var_27 = '/test/fifo'
    var_28 = var_26.is_skipped(var_1)
    var_29 = module_0.Config()
    var_30 = 'test.py~'
    var_31 = module_0.Config()
    var_32 = 'test.unsupported'
    var_33 = var_31.is_skipped(var_1)
    var_34 = module_0.Config()
    var_35 = 'test.py'
    var_36 = var_34.is_skipped(var_1)
    assert var_36 is False



# Parsed testcases at query #15
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
    var_10 = 'test_shebang.py'
    var_11 = var_0.is_supported_filetype(var_10)
    assert var_11 is True
    var_12 = "print('hello')\n"
    var_13 = 'test_no_shebang.py'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is True
    var_15 = 'hello\n'
    var_16 = 'test.txt'
    var_17 = var_0.is_supported_filetype(var_16)
    assert var_17 is False



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'other.py'
    assert var_3 is False
    var_4 = 'tests/'
    assert var_4 is True
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
    var_16 = '/git/root'
    var_17 = '/git/root/file1.py'
    var_18 = '/git/root/file2.py'
    var_19 = '/git/root/file3.py'
    var_20 = module_0.Config()
    var_21 = 'nonexistent.py'
    var_22 = 'test_dir'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = 'test_dir/file.py'
    var_26 = 'skipme.py'
    var_27 = [var_26]
    var_28 = [var_9]
    var_29 = module_0.Config()



# Parsed testcases at query #2
#--------------------------


import isort.utils as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = {}
    var_2 = module_0.Trie(var_0, var_1)
    var_3 = 'subdir'
    var_4 = 'default'
    var_5 = {}
    var_6 = module_0.Trie(var_4, var_5)
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length=88\n'
    var_9 = '.isort.cfg'
    var_10 = '[settings]\nline_length=88\n'
    var_11 = 'setup.cfg'
    var_12 = '[isort]\nline_length=120\n'
    var_13 = '.isort.cfg'
    var_14 = '[settings]\nline_length=88\n'
    var_15 = 'subdir'
    var_16 = 'setup.cfg'
    var_17 = '[isort]\nline_length=120\n'
    var_18 = '.isort.cfg'
    var_19 = 'invalid config content'
    var_20 = 'default'
    var_21 = {}
    var_22 = module_0.Trie(var_20, var_21)



# Parsed testcases at query #3
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
    var_16 = 'symlink_to_test'
    var_17 = module_0.Config()
    var_18 = 'non_existent_file.py'
    var_19 = module_0.Config()
    var_20 = 'test_git_folder'
    var_21 = 'git'
    var_22 = 'init'
    var_23 = [var_21, var_22]
    var_24 = 'config'
    var_25 = 'user.email'
    var_26 = 'test@test.com'
    var_27 = [var_21, var_24, var_25, var_26]
    var_28 = 'user.name'
    var_29 = 'Test User'
    var_30 = [var_21, var_24, var_28, var_29]
    var_31 = 'add'
    var_32 = '.'
    var_33 = [var_21, var_31, var_32]
    var_34 = 'commit'
    var_35 = '-m'
    var_36 = 'Initial commit'
    var_37 = [var_21, var_34, var_35, var_36]
    var_38 = [var_0]



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'
    var_2 = True
    var_3 = 100
    var_4 = 90
    var_5 = module_0.Config()
    var_6 = 120
    var_7 = 100
    var_8 = module_0.Config()
    var_9 = '[isort]\nline_length=88\n'
    var_10 = module_0.Config(var_1)
    var_11 = '/nonexistent/path'
    var_12 = module_0.Config(settings_path=var_11)
    var_13 = 'black'
    var_14 = module_0.Config()
    var_15 = var_14.source
    var_16 = str(var_15)
    var_17 = 'nonexistent'
    var_18 = module_0.Config()
    var_19 = module_0._Config(line_length=var_3, wrap_length=var_4)
    var_20 = module_0.Config(config=var_19)
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = False
    var_24 = 'test'
    var_25 = module_0.Config()
    var_26 = 'test'
    var_27 = 'testmodule'
    var_28 = {var_27}
    var_29 = {var_26: var_28}
    var_30 = module_0.Config()
    var_31 = 'Test Heading'
    var_32 = module_0.Config()
    var_33 = 'Test Footer'
    var_34 = module_0.Config()
    var_35 = module_0.Config()
    var_36 = 'nonexistent'
    var_37 = module_0.Config()
    var_38 = 'natural'
    var_39 = module_0.Config()
    var_40 = 'nonexistent'
    var_41 = module_0.Config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88\n'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'setup.cfg'
    var_5 = '[settings]\nline_length=100\n'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nprofile=black\n'
    var_8 = '.isort.cfg'
    var_9 = 'invalid config content'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'config_dir'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile = black\n'
    var_3 = 'pyproject.toml'
    var_4 = 'invalid toml content'
    var_5 = 'subdir'
    var_6 = '.isort.cfg'
    var_7 = '[isort]\nline_length = 100\n'
    var_8 = 'empty_dir'



# Parsed testcases at query #7
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
    var_15 = 'invalid_profile'
    var_16 = module_0.Config()
    var_17 = 10
    var_18 = 5
    var_19 = module_0.Config()
    var_20 = True
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = 'value'
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
    var_34 = {var_25: var_33}
    var_35 = module_0.Config()
    var_36 = 'Custom Footer'
    var_37 = {var_25: var_36}
    var_38 = module_0.Config()
    var_39 = 'src'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = var_41.src_paths
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = 0
    var_45 = var_41.src_paths[var_44]
    var_46 = str(var_45)
    var_47 = module_0.Config()
    var_48 = 'invalid_formatter'
    var_49 = module_0.Config()
    var_50 = 'natural'
    var_51 = module_0.Config()
    var_52 = 'invalid_sort_order'
    var_53 = module_0.Config()



# Parsed testcases at query #8
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
    var_17 = '/repo/src/file.py'
    var_18 = '/repo/ignored.py'
    var_19 = 'nonexistent.py'
    var_20 = '.git'
    var_21 = 'file~'
    var_22 = '/project'
    var_23 = 'skip_me'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = '/project/skip_me'
    var_27 = '/other/skip_me'



# Parsed testcases at query #9
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
    var_7 = 'tests/file.py'
    var_8 = 'src/file.py'
    var_9 = '*.tmp'
    var_10 = {var_9}
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
    var_24 = 'dir_to_skip'
    var_25 = {var_24}
    var_26 = module_0.Config()



# Parsed testcases at query #10
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
    var_9 = 'test.tmp'
    var_10 = {var_6}
    var_11 = module_0.Config()
    var_12 = module_0.Config()
    var_13 = var_12.is_skipped(var_0)
    assert var_13 is False
    var_14 = module_0.Config()
    var_15 = var_14.is_skipped(var_3)
    assert var_15 is False
    var_16 = True
    var_17 = module_0.Config()
    var_18 = '.git'
    var_19 = var_0 / var_18
    var_20 = 'test.py'
    var_21 = False
    var_22 = module_0.Config()
    var_23 = '.git'
    var_24 = var_0 / var_23
    var_25 = 'test.py'
    var_26 = 'dir_to_skip'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = 'dir_to_skip'
    var_30 = var_0 / var_29
    var_31 = 'test.py'
    var_32 = var_30 / var_31
    var_33 = var_28.is_skipped(var_32)
    assert var_33 is True
    var_34 = 'other_dir'
    var_35 = {var_34}
    var_36 = module_0.Config()
    var_37 = 'test_dir'
    var_38 = var_0 / var_37
    var_39 = 'test.py'
    var_40 = var_38 / var_39
    var_41 = var_36.is_skipped(var_40)
    assert var_41 is False



