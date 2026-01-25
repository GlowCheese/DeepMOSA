####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_supported_filetype method of Config class.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'module.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyc'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.pyo'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_1.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'backup~'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'test.txt'
    var_15 = var_1.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = 'test.md'
    var_17 = var_1.is_supported_filetype(var_16)
    assert var_17 is False
    var_18 = b'#!/usr/bin/env python\n'
    assert var_18 is True
    var_19 = b"print('hello')\n"
    var_20 = b'just some text\n'
    assert var_20 is False
    var_21 = '/nonexistent/path/file.py'
    var_22 = var_1.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'py'
    var_24 = 'pyi'
    var_25 = 'txt'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.Config()
    var_28 = var_27.is_supported_filetype(var_14)
    assert var_28 is True
    var_29 = [var_23]
    var_30 = module_0.Config()
    var_31 = var_30.is_supported_filetype(var_2)
    assert var_31 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config constructor with various parameter combinations.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 'tab'
    var_6 = module_0.Config()
    var_7 = "'    '"
    var_8 = module_0.Config()
    var_9 = 120
    var_10 = module_0.Config()
    var_11 = 2
    var_12 = module_0.Config(config=var_10)
    var_13 = True
    var_14 = module_0.Config()
    var_15 = 'black'
    var_16 = module_0.Config()
    var_17 = 'django'
    var_18 = [var_17]
    var_19 = 'flask'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = set()
    var_23 = set()
    var_24 = 'Future imports'
    var_25 = 'End stdlib'
    var_26 = module_0.Config()
    var_27 = 'FUTURE'
    var_28 = 'STDLIB'
    var_29 = 'THIRDPARTY'
    var_30 = 'FIRSTPARTY'
    var_31 = 'LOCALFOLDER'
    var_32 = [var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.Config()
    var_34 = '/path/to/src'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = '/tmp'
    var_38 = module_0.Config()
    var_39 = 'migrations'
    var_40 = [var_39]
    var_41 = 'node_modules'
    var_42 = [var_41]
    var_43 = module_0.Config()
    var_44 = '*.egg-info'
    var_45 = [var_44]
    var_46 = 'build/*'
    var_47 = [var_46]
    var_48 = module_0.Config()
    var_49 = 'natural'
    var_50 = module_0.Config()
    var_51 = 150
    var_52 = 100
    var_53 = module_0.Config()
    var_54 = 8
    var_55 = module_0.Config()
    var_56 = 'py'
    var_57 = 'pyi'
    var_58 = [var_56, var_57]
    var_59 = module_0.Config()
    var_60 = 'pyc'
    var_61 = [var_60]
    var_62 = module_0.Config()
    var_63 = module_0.Config()
    var_64 = module_0.Config()
    var_65 = hash(var_63)
    var_66 = hash(var_64)
    var_67 = hash(var_63)
    var_68 = id(var_63)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function with various config file scenarios.'
    var_1 = 'project'
    var_2 = 'src'
    var_3 = 'tests'
    var_4 = 'nested'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nline_length=100\n'
    var_9 = '.isort.cfg'
    var_10 = '[settings]\nmulti_line_mode=3\n'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty_project'
    var_2 = 'src'

def test_case_0():
    var_0 = 'Test find_all_configs with malformed config file.'
    var_1 = 'project'
    var_2 = 'setup.cfg'
    var_3 = '[invalid config content {{{'

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directories.'
    var_1 = 'project'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=88\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nmulti_line_mode=3\n'

def test_case_0():
    var_0 = 'Test find_all_configs with empty directory.'
    var_1 = 'empty'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to verify it correctly finds and parses config files.'
    var_1 = 'subdir1'
    var_2 = 'subdir2'
    var_3 = 'nested'
    var_4 = '[isort]\nprofile=black\n'
    var_5 = "[tool.isort]\nprofile='django'\n"
    var_6 = 'setup.cfg'
    var_7 = '.isort.cfg'
    var_8 = 'pyproject.toml'

def test_case_0():
    var_0 = 'Test find_all_configs with directory containing no config files.'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'setup.cfg'
    var_2 = '[invalid content that cannot be parsed'

def test_case_0():
    var_0 = 'Test find_all_configs with nested directory structure.'
    var_1 = 'level1'
    var_2 = 'level2'
    var_3 = 'level3'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nprofile=black\n'
    var_6 = '.isort.cfg'
    var_7 = '[isort]\nprofile=django\n'



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization methods.'
    var_1 = module_0.Config()
    var_2 = 'known_other'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'import_headings'
    var_5 = hasattr(var_1, var_4)
    var_6 = 'import_footers'
    var_7 = hasattr(var_1, var_6)
    var_8 = 100
    var_9 = 4
    var_10 = module_0.Config()
    var_11 = 'black'
    var_12 = module_0.Config()
    var_13 = 88
    var_14 = module_0.Config()
    var_15 = module_0.Config(config=var_14)
    var_16 = 'tab'
    var_17 = module_0.Config()
    var_18 = "'    '"
    var_19 = module_0.Config()
    var_20 = 'mymodule'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'from __future__ imports'
    var_24 = module_0.Config()
    var_25 = 'End of stdlib'
    var_26 = module_0.Config()
    var_27 = 'migrations'
    var_28 = [var_27]
    var_29 = 'node_modules'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = '*.egg-info'
    var_33 = [var_32]
    var_34 = 'venv/**'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = 'FUTURE'
    var_38 = 'STDLIB'
    var_39 = 'THIRDPARTY'
    var_40 = 'FIRSTPARTY'
    var_41 = 'LOCALFOLDER'
    var_42 = [var_37, var_38, var_39, var_40, var_41]
    var_43 = module_0.Config()
    var_44 = 'django'
    var_45 = [var_44]
    var_46 = module_0.Config()
    var_47 = []
    var_48 = '/tmp'
    var_49 = module_0.Config()
    var_50 = 120
    var_51 = 3
    var_52 = True
    var_53 = False
    var_54 = module_0.Config()
    var_55 = 'nonexistent_profile_that_should_not_error'
    var_56 = module_0.Config()
    var_57 = 'invalid_profile_xyz'
    var_58 = module_0.Config()
    var_59 = module_0.Config()
    var_60 = var_59.src_paths
    var_61 = len(var_60)
    var_62 = 'py'
    var_63 = 'pyi'
    var_64 = [var_62, var_63]
    var_65 = module_0.Config()
    var_66 = 'pyc'
    var_67 = [var_66]
    var_68 = module_0.Config()
    var_69 = 'natural'
    var_70 = module_0.Config()
    var_71 = module_0.Config()
    var_72 = 79
    var_73 = module_0.Config()
    var_74 = 100
    var_75 = 88
    var_76 = module_0.Config()
    var_77 = module_0.Config()
    var_78 = module_0.Config(config=var_77)



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_skipped method with various file paths and skip configurations.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'skip_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'skip_dir/test_file.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'test_file.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = module_0.Config()
    var_18 = 'test_file.py~'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = '.git'
    var_22 = module_0.Config()
    var_23 = '/nonexistent/path/to/file.py'
    var_24 = 'dir1'
    var_25 = [var_24]
    var_26 = 'dir2'
    var_27 = [var_26]
    var_28 = module_0.Config()
    var_29 = 'dir2/test_file.py'
    var_30 = '*.log'
    var_31 = [var_30]
    var_32 = '*.tmp'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = 'test_file.tmp'
    var_36 = 'nested'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = 'nested/deep/test_file.py'
    var_40 = 'test_file.py'
    var_41 = var_0 / var_40
    var_42 = var_38.is_skipped(var_41)
    assert var_42 is False



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_supported_filetype method.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'module.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyc'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.pyo'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_1.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'backup~'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'README'
    var_15 = var_1.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = '/nonexistent/path/file.py'
    var_17 = var_1.is_supported_filetype(var_16)
    assert var_17 is False
    var_18 = 'py'
    var_19 = 'pyi'
    var_20 = 'txt'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.Config()
    var_23 = 'test.txt'
    var_24 = var_22.is_supported_filetype(var_23)
    assert var_24 is True
    var_25 = [var_18]
    var_26 = module_0.Config()
    var_27 = var_26.is_supported_filetype(var_2)
    assert var_27 is False
    var_28 = 'test.PY'
    var_29 = var_1.is_supported_filetype(var_28)
    assert var_29 is True
    var_30 = 'test.PYI'
    var_31 = var_1.is_supported_filetype(var_30)
    assert var_31 is True



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 'tab'
    var_6 = module_0.Config()
    var_7 = '2'
    var_8 = module_0.Config()
    var_9 = "'    '"
    var_10 = module_0.Config()
    var_11 = module_0._Config()
    var_12 = 120
    var_13 = module_0.Config(config=var_11)
    var_14 = 80
    var_15 = 100
    var_16 = module_0.Config()
    var_17 = '/nonexistent/path'
    var_18 = module_0.Config(settings_path=var_17)
    var_19 = 'nonexistent_profile'
    var_20 = module_0.Config()
    var_21 = 'value'
    var_22 = module_0.Config()
    var_23 = True
    var_24 = 80
    var_25 = module_0.Config()
    var_26 = 'mypackage'
    var_27 = [var_26]
    var_28 = frozenset(var_27)
    var_29 = 'FUTURE'
    var_30 = 'STDLIB'
    var_31 = 'THIRDPARTY'
    var_32 = 'CUSTOM'
    var_33 = 'FIRSTPARTY'
    var_34 = 'LOCALFOLDER'
    var_35 = [var_29, var_30, var_31, var_32, var_33, var_34]
    var_36 = module_0.Config()
    var_37 = 'Standard Library'
    var_38 = 'Third Party'
    var_39 = module_0.Config()
    var_40 = 'End Standard Library'
    var_41 = module_0.Config()
    var_42 = module_0.Config()
    var_43 = var_42.src_paths
    var_44 = len(var_43)
    var_45 = '/tmp'
    var_46 = module_0.Config()
    var_47 = 88
    var_48 = 3
    var_49 = '__pycache__'
    var_50 = [var_49]
    var_51 = frozenset(var_50)
    var_52 = module_0.Config()
    var_53 = '100'
    var_54 = module_0.Config()
    var_55 = var_54.line_length
    var_56 = module_0.Config()
    var_57 = module_0.Config()
    var_58 = module_0.Config()
    var_59 = hash(var_57)
    var_60 = hash(var_58)
    var_61 = hash(var_57)
    var_62 = id(var_57)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function finds and parses config files in directory tree.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=88\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs with empty directory.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs when no valid config files exist.'
    var_1 = 'project'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[invalid]\nbroken=true\n'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple nested directory levels.'
    var_1 = 'project'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=88\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test that find_all_configs stops at first config file in a directory.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black\n'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nline_length=88\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles malformed config files gracefully.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = 'this is not valid config format [[['

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find_all_configs with nonexistent path.'
    var_1 = 'nonexistent'
    var_2 = module_0.find_all_configs(var_0)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs discovers and parses config files in directory tree.'
    var_1 = 'project'
    var_2 = 'src'
    var_3 = 'tests'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nline_length=88\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nprofile=black\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nline_length=100\n'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty_project'
    var_2 = 'src'

def test_case_0():
    var_0 = 'Test find_all_configs with single config file.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length=100\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = '[invalid\n'

def test_case_0():
    var_0 = 'Test find_all_configs with configs at multiple directory levels.'
    var_1 = 'multi_level'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nline_length=88\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nprofile=black\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'Test find_all_configs with empty config files.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = ''



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_skipped method with various file paths and skip configurations.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = module_0.Config()
    var_9 = 'venv'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'venv/lib/python.py'
    var_14 = '*.pyc'
    var_15 = [var_14]
    var_16 = frozenset(var_15)
    var_17 = module_0.Config()
    var_18 = 'test.pyc'
    var_19 = [var_14]
    var_20 = frozenset(var_19)
    var_21 = module_0.Config()
    var_22 = 'test.py'
    var_23 = module_0.Config()
    var_24 = '/nonexistent/path/to/file.py'
    var_25 = 'dir1'
    var_26 = [var_25]
    var_27 = frozenset(var_26)
    var_28 = 'dir2'
    var_29 = [var_28]
    var_30 = frozenset(var_29)
    var_31 = module_0.Config()
    var_32 = 'dir2/file.py'
    var_33 = [var_14]
    var_34 = frozenset(var_33)
    var_35 = '*.pyo'
    var_36 = [var_35]
    var_37 = frozenset(var_36)
    var_38 = module_0.Config()
    var_39 = 'test.pyo'
    var_40 = '.git'
    var_41 = var_0 / var_40
    var_42 = True
    var_43 = var_38.is_skipped(var_41)
    var_44 = 'test.py'
    var_45 = var_0 / var_44
    var_46 = '# test'
    var_47 = var_38.is_skipped(var_45)
    var_48 = '/absolute/path/file.py'
    var_49 = [var_48]
    var_50 = frozenset(var_49)
    var_51 = module_0.Config()
    var_52 = '**/test_*.py'
    var_53 = [var_52]
    var_54 = frozenset(var_53)
    var_55 = module_0.Config()
    var_56 = 'tests/test_config.py'
    var_57 = '__pycache__'
    var_58 = [var_57]
    var_59 = frozenset(var_58)
    var_60 = module_0.Config()
    var_61 = 'src/__pycache__/module.pyc'
    var_62 = 'real.py'
    var_63 = var_0 / var_62
    var_64 = '# test'
    var_65 = 'link.py'
    var_66 = var_47 / var_65
    var_67 = var_60.is_skipped(var_66)
    var_68 = []
    var_69 = frozenset(var_68)
    var_70 = []
    var_71 = frozenset(var_70)
    var_72 = module_0.Config()
    var_73 = 'test.py'
    var_74 = var_0 / var_73
    var_75 = '# test'
    var_76 = var_72.is_skipped(var_74)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization methods.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 'tab'
    var_6 = module_0.Config()
    var_7 = "'    '"
    var_8 = module_0.Config()
    var_9 = module_0._Config()
    var_10 = 120
    var_11 = module_0.Config(config=var_9)
    var_12 = 80
    var_13 = 100
    var_14 = module_0.Config()
    var_15 = 'black'
    var_16 = module_0.Config()
    var_17 = 'nonexistent_profile_xyz'
    var_18 = module_0.Config()
    var_19 = True
    var_20 = module_0.Config()
    var_21 = 'django'
    var_22 = [var_21]
    var_23 = 'myapp'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = 'FUTURE'
    var_27 = 'STDLIB'
    var_28 = 'THIRDPARTY'
    var_29 = 'FIRSTPARTY'
    var_30 = 'LOCALFOLDER'
    var_31 = [var_26, var_27, var_28, var_29, var_30]
    var_32 = module_0.Config()
    var_33 = '8'
    var_34 = module_0.Config()
    var_35 = 'Standard Library'
    var_36 = module_0.Config()
    var_37 = 'Third Party Footer'
    var_38 = module_0.Config()
    var_39 = 3
    var_40 = 0
    var_41 = module_0.Config()
    var_42 = module_0.Config()
    var_43 = hash(var_42)
    var_44 = id(var_42)
    var_45 = 'src'
    var_46 = 'lib'
    var_47 = [var_45, var_46]
    var_48 = module_0.Config()
    var_49 = {}
    var_50 = module_0.Config(**var_49)
    var_51 = False
    var_52 = module_0.Config()
    var_53 = '.'
    var_54 = module_0.Config()



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_skipped method with various file paths and skip configurations.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'other_file.py'
    var_7 = 'build'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'build/output.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = 'test.py'
    var_18 = '__pycache__/*'
    var_19 = [var_18]
    var_20 = module_0.Config()
    var_21 = '__pycache__/test.pyc'
    var_22 = 'skip1.py'
    var_23 = [var_22]
    var_24 = 'skip2.py'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = [var_11]
    var_28 = '*.pyo'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = 'test.pyo'
    var_32 = []
    var_33 = module_0.Config()
    var_34 = '/nonexistent/path/file.py'
    var_35 = 'test_dir'
    var_36 = '# test'
    var_37 = []
    var_38 = 'test_dir2'
    var_39 = 'exists.py'
    var_40 = []
    var_41 = module_0.Config()
    var_42 = 'src\\module'
    var_43 = [var_42]
    var_44 = module_0.Config()
    var_45 = 'src/module/file.py'
    var_46 = 'dist'
    var_47 = '*.egg-info'
    var_48 = [var_7, var_46, var_47]
    var_49 = module_0.Config()
    var_50 = 'build/file.py'
    var_51 = 'dist/file.py'
    var_52 = True
    var_53 = module_0.Config()
    var_54 = '.git'
    var_55 = 'src/generated'
    var_56 = [var_55]
    var_57 = module_0.Config()
    var_58 = 'src/generated/code.py'



# Parsed testcases at query #14
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
    var_5 = 'test.pyc'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.txt'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.py~'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = '/nonexistent/path/file.py'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = '#!/usr/bin/env python\n'
    assert var_13 is True
    var_14 = 'no shebang\n'
    assert var_14 is False
    var_15 = 'custom'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = 'test.custom'
    var_19 = var_17.is_supported_filetype(var_18)
    assert var_19 is True
    var_20 = var_17.is_supported_filetype(var_14)
    assert var_20 is False
    var_21 = 'py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = var_23.is_supported_filetype(var_14)
    assert var_24 is False



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = '_known_patterns'
    var_3 = hasattr(var_1, var_2)
    var_4 = '_section_comments'
    var_5 = hasattr(var_1, var_4)
    var_6 = '_section_comments_end'
    var_7 = hasattr(var_1, var_6)
    var_8 = '_skips'
    var_9 = hasattr(var_1, var_8)
    var_10 = '_skip_globs'
    var_11 = hasattr(var_1, var_10)
    var_12 = '_sorting_function'
    var_13 = hasattr(var_1, var_12)
    var_14 = 100
    var_15 = True
    var_16 = module_0.Config()
    var_17 = module_0._Config()
    var_18 = 88
    var_19 = module_0.Config(config=var_17)
    var_20 = '/nonexistent/path/to/config'
    var_21 = module_0.Config(settings_path=var_20)
    var_22 = 'nonexistent_profile'
    var_23 = module_0.Config()
    var_24 = module_0.Config()
    var_25 = var_24.known_patterns
    var_26 = var_24.known_patterns
    var_27 = 'future'
    var_28 = 'Future imports'
    var_29 = {var_27: var_28}
    var_30 = module_0.Config()
    var_31 = var_30.section_comments
    var_32 = 'stdlib'
    var_33 = 'Standard library'
    var_34 = {var_32: var_33}
    var_35 = module_0.Config()
    var_36 = var_35.section_comments_end
    var_37 = 'tests'
    var_38 = [var_37]
    var_39 = 'venv'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = var_41.skips
    var_43 = '*.egg-info'
    var_44 = [var_43]
    var_45 = '*.pyc'
    var_46 = [var_45]
    var_47 = module_0.Config()
    var_48 = var_47.skip_globs
    var_49 = 'natural'
    var_50 = module_0.Config()
    var_51 = var_50.sorting_function
    var_52 = callable(var_51)
    var_53 = 'native'
    var_54 = module_0.Config()
    var_55 = var_54.sorting_function
    var_56 = 'invalid_sort_order'
    var_57 = module_0.Config()
    var_58 = var_57.sorting_function
    var_59 = '4'
    var_60 = module_0.Config()
    var_61 = 'tab'
    var_62 = module_0.Config()
    var_63 = "'  '"
    var_64 = module_0.Config()
    var_65 = module_0.Config()
    var_66 = module_0.Config()
    var_67 = var_66.src_paths
    var_68 = var_66.src_paths
    var_69 = len(var_68)
    var_70 = 'invalid'
    var_71 = module_0.Config()
    var_72 = module_0.Config()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = module_0.Config()
    var_9 = 'skip_dir'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'skip_dir/test_file.py'
    var_14 = '*.pyc'
    var_15 = [var_14]
    var_16 = frozenset(var_15)
    var_17 = module_0.Config()
    var_18 = 'test_file.pyc'
    var_19 = [var_14]
    var_20 = frozenset(var_19)
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = '/non/existent/path/file.py'
    var_24 = 'file1.py'
    var_25 = [var_24]
    var_26 = frozenset(var_25)
    var_27 = 'file2.py'
    var_28 = [var_27]
    var_29 = frozenset(var_28)
    var_30 = module_0.Config()
    var_31 = [var_14]
    var_32 = frozenset(var_31)
    var_33 = '*.pyo'
    var_34 = [var_33]
    var_35 = frozenset(var_34)
    var_36 = module_0.Config()
    var_37 = 'test.pyc'
    var_38 = 'test.pyo'
    var_39 = 'skip_me'
    var_40 = [var_39]
    var_41 = frozenset(var_40)
    var_42 = var_3 / var_39
    var_43 = 'file.py'
    var_44 = var_42 / var_43
    var_45 = var_36.is_skipped(var_44)
    assert var_45 is True
    var_46 = '.git'
    var_47 = var_39 / var_46
    var_48 = True
    var_49 = module_0.Config()
    var_50 = var_49.is_skipped(var_47)
    assert var_50 is True
    var_51 = '/build/*'
    var_52 = [var_51]
    var_53 = frozenset(var_52)
    var_54 = module_0.Config()
    var_55 = 'build/test.py'
    var_56 = 'node_modules'
    var_57 = [var_56]
    var_58 = frozenset(var_57)
    var_59 = '/project'
    var_60 = module_0.Config()
    var_61 = '/project/node_modules/package/index.js'
    var_62 = var_60.is_skipped(var_44)
    assert var_62 is True



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=80\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs with directory containing no config files.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs stops at first config in directory.'
    var_1 = 'project'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nprofile=black\n'
    var_5 = '[settings]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'project'
    var_2 = 'setup.cfg'
    var_3 = '[invalid\nbroken syntax'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple config sources in same directory.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black\n'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nline_length=80\n'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    assert var_0 is False
    assert var_0 is True
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = '__pycache__'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = '*.pyc'
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test_file.pyc'
    var_14 = [var_10]
    var_15 = module_0.Config()
    var_16 = module_0.Config()
    var_17 = '/nonexistent/path/to/file.py'
    var_18 = module_0.Config()
    var_19 = 'file1.py'
    var_20 = [var_19]
    var_21 = 'file2.py'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = [var_10]
    var_25 = '*.pyo'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = 'test_file.pyo'
    var_29 = 'venv'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = 'venv/lib/python3.9/site-packages/module.py'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 'black'
    var_4 = module_0.Config()
    var_5 = True
    var_6 = module_0.Config()
    var_7 = 88
    var_8 = 3
    var_9 = module_0.Config()
    var_10 = module_0.Config(config=var_9)
    var_11 = '4'
    var_12 = module_0.Config()
    var_13 = "'    '"
    var_14 = module_0.Config()
    var_15 = 'tab'
    var_16 = module_0.Config()
    var_17 = 'django'
    var_18 = [var_17]
    var_19 = 'FUTURE'
    var_20 = 'STDLIB'
    var_21 = 'DJANGO'
    var_22 = 'THIRDPARTY'
    var_23 = 'FIRSTPARTY'
    var_24 = 'LOCALFOLDER'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24]
    var_26 = module_0.Config()
    var_27 = set()
    var_28 = module_0.Config()
    var_29 = var_28.src_paths
    var_30 = len(var_29)
    var_31 = '/nonexistent/path/that/does/not/exist'
    var_32 = module_0.Config(settings_path=var_31)
    var_33 = '/tmp'
    var_34 = module_0.Config()
    var_35 = 'py'
    var_36 = 'pyi'
    var_37 = [var_35, var_36]
    var_38 = module_0.Config()
    var_39 = 'migrations'
    var_40 = 'venv'
    var_41 = [var_39, var_40]
    var_42 = module_0.Config()
    var_43 = 80
    var_44 = module_0.Config()
    var_45 = 'mymodule'
    var_46 = [var_45]
    var_47 = 'CUSTOM'
    var_48 = [var_19, var_20, var_22, var_47, var_23, var_24]
    var_49 = module_0.Config()
    var_50 = 120
    var_51 = 2
    var_52 = module_0.Config()
    var_53 = module_0.Config()
    var_54 = module_0.Config()
    var_55 = 'pyc'
    var_56 = 'pyo'
    var_57 = [var_55, var_56]
    var_58 = module_0.Config()
    var_59 = [var_40]
    var_60 = 'build'
    var_61 = 'dist'
    var_62 = [var_60, var_61]
    var_63 = module_0.Config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it correctly discovers and parses config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length=88\n'
    var_9 = 'pyproject.toml'
    var_10 = "[tool.isort]\nprofile = 'django'\n"

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directory structure.'
    var_1 = 'root'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nline_length=80\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nprofile=black\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'root'
    var_2 = 'setup.cfg'
    var_3 = '[invalid\nbroken config file\n'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple config file types in same directory.'
    var_1 = 'root'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\n'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nline_length=88\n'

def test_case_0():
    var_0 = 'Test find_all_configs with empty config files.'
    var_1 = 'root'
    var_2 = 'setup.cfg'
    var_3 = ''



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    assert var_1 is False
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'test_dir/file.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = 'test.py~'
    var_20 = module_0.Config()
    var_21 = '/nonexistent/path/file.py'
    var_22 = 'test.py'
    var_23 = 'file1.py'
    var_24 = [var_23]
    var_25 = 'file2.py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = [var_11]
    var_29 = '*.pyo'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = 'test.pyo'



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config constructor with various initialization methods.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 3
    var_4 = module_0.Config()
    var_5 = 88
    var_6 = module_0.Config()
    var_7 = module_0.Config(config=var_6)
    var_8 = True
    var_9 = module_0.Config()
    var_10 = 'nonexistent_profile_xyz'
    var_11 = module_0.Config()
    var_12 = '/nonexistent/path/that/does/not/exist'
    var_13 = module_0.Config(settings_path=var_12)
    var_14 = 4
    var_15 = module_0.Config()
    var_16 = 'tab'
    var_17 = module_0.Config()
    var_18 = "'  '"
    var_19 = module_0.Config()
    var_20 = 'mymodule'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'Future imports'
    var_24 = module_0.Config()
    var_25 = 'End of stdlib'
    var_26 = module_0.Config()
    var_27 = module_0.Config()
    var_28 = vars(var_27)
    var_29 = var_27.src_paths
    var_30 = len(var_29)
    var_31 = True
    var_32 = module_0.Config()
    var_33 = module_0.Config()
    var_34 = vars(var_33)
    var_35 = 'black'
    var_36 = module_0.Config()
    var_37 = 'nonexistent_formatter_xyz'
    var_38 = module_0.Config()
    var_39 = 150
    var_40 = 100
    var_41 = module_0.Config()
    var_42 = 'FUTURE'
    var_43 = 'STDLIB'
    var_44 = 'THIRDPARTY'
    var_45 = 'FIRSTPARTY'
    var_46 = 'LOCALFOLDER'
    var_47 = [var_42, var_43, var_44, var_45, var_46]
    var_48 = module_0.Config()
    var_49 = 'natural'
    var_50 = module_0.Config()
    var_51 = module_0.Config()
    var_52 = module_0.Config(config=var_51)
    var_53 = module_0.Config()
    var_54 = hash(var_53)
    var_55 = hash(var_53)
    var_56 = module_0.Config()
    var_57 = module_0.Config()
    var_58 = hash(var_56)
    var_59 = hash(var_57)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it properly discovers and parses config files.'
    var_1 = 'subdir1'
    var_2 = 'subdir2'
    var_3 = 'nested'
    var_4 = '.isort.cfg'
    var_5 = '[settings]\nprofile=black\n'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nline_length=100\n'
    var_8 = 'setup.cfg'
    var_9 = '[isort]\nindent=4\n'
    var_10 = '[settings]\nskip=migrations\n'

def test_case_0():
    var_0 = 'Test find_all_configs with a directory containing no config files.'
    var_1 = 'subdir1'
    var_2 = 'subdir2'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = '.isort.cfg'
    var_2 = '[invalid\nbroken config\n'
    var_3 = 'subdir'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directory structure.'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = True
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nprofile=black\n'
    var_9 = 'setup.cfg'
    var_10 = '[isort]\nline_length=80\n'
    var_11 = 'pyproject.toml'
    var_12 = '[tool.isort]\nskip=__init__.py\n'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple config file types in same directory.'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nprofile=black\n'
    var_3 = 'setup.cfg'
    var_4 = '[isort]\nline_length=100\n'
    var_5 = 'pyproject.toml'
    var_6 = '[tool.isort]\nindent=2\n'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function with various config file scenarios.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=88\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs with empty directory.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs when config file cannot be parsed.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = "[invalid content that won't parse"

def test_case_0():
    var_0 = 'Test find_all_configs with multiple nested levels.'
    var_1 = 'root'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = True
    var_6 = '.isort.cfg'
    var_7 = '[settings]\nprofile=black\n'
    var_8 = 'setup.cfg'
    var_9 = '[isort]\nline_length=88\n'
    var_10 = 'pyproject.toml'
    var_11 = '[tool.isort]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs respects directory walking.'
    var_1 = 'project'
    var_2 = 'subdir'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nprofile=black\n'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nline_length=88\n'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function discovers and parses config files in directory tree.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile=black\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=80\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile=django\n'
    var_11 = '[settings]\nline_length=120\n'

def test_case_0():
    var_0 = 'Test find_all_configs with directory containing no config files.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs with single config file.'
    var_1 = 'single'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black\nline_length=88\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'invalid'
    var_2 = '.isort.cfg'
    var_3 = '[invalid\nbroken syntax'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to verify it correctly discovers and parses config files.'
    var_1 = 'subdir1'
    var_2 = 'subdir2'
    var_3 = 'nested'
    var_4 = 'profile'
    var_5 = 'line_length'
    var_6 = 'black'
    var_7 = 88
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'django'
    var_10 = 100
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = 'indent'
    var_13 = 2
    var_14 = {var_12: var_13}
    var_15 = '__main__._get_config_data'
    var_16 = '.isort.cfg'
    var_17 = 'setup.cfg'
    var_18 = 'pyproject.toml'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files are present.'
    var_1 = 'subdir'

def test_case_0():
    var_0 = 'Test find_all_configs handles exceptions in config parsing gracefully.'
    var_1 = 'subdir'
    var_2 = '.isort.cfg'
    var_3 = '__main__._get_config_data'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple nested directory levels.'
    var_1 = 'level1'
    var_2 = 'level2'
    var_3 = 'level3'
    var_4 = '.isort.cfg'
    var_5 = 'setup.cfg'
    var_6 = 'pyproject.toml'
    var_7 = '__main__._get_config_data'

def test_case_0():
    var_0 = 'Test find_all_configs when config files exist but return empty data.'
    var_1 = 'subdir'
    var_2 = '.isort.cfg'
    var_3 = '__main__._get_config_data'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to verify it finds and parses config files in directory tree.'
    var_1 = 'project'
    var_2 = 'src'
    var_3 = 'tests'
    var_4 = 'deep'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length=100\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile="django"\n'
    var_11 = 'insert'
    var_12 = 0

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple config file types in same directory.'
    var_1 = 'multi_config'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\n'

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directory structure.'
    var_1 = 'nested'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length=80\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'invalid'
    var_2 = 'setup.cfg'
    var_3 = 'invalid content without proper formatting'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'other_file.py'
    var_7 = 'skip_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'skip_dir/file.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = 'test.py'
    var_18 = []
    var_19 = module_0.Config()
    var_20 = '/nonexistent/path/file.py'
    var_21 = 'file1.py'
    var_22 = [var_21]
    var_23 = 'file2.py'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = [var_11]
    var_27 = '*.pyo'
    var_28 = [var_27]
    var_29 = module_0.Config()
    var_30 = 'test.pyo'
    var_31 = '.git'
    var_32 = var_0 / var_31
    var_33 = True
    var_34 = var_29.is_skipped(var_32)
    assert var_34 is True
    var_35 = 'test.py'
    var_36 = var_0 / var_35
    var_37 = [var_35]
    var_38 = var_29.is_skipped(var_36)
    assert var_38 is True
    var_39 = 'dir/file.py'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = 'dir\\file.py'
    var_43 = 'valid.py'
    var_44 = var_0 / var_43
    var_45 = []
    var_46 = var_41.is_skipped(var_44)
    assert var_46 is False



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_skipped method with various file paths and skip configurations.'
    var_1 = '__pycache__'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = [var_1]
    var_6 = frozenset(var_5)
    var_7 = module_0.Config()
    var_8 = 'myfile.py'
    var_9 = '*.pyc'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'file.pyc'
    var_14 = [var_9]
    var_15 = frozenset(var_14)
    var_16 = module_0.Config()
    var_17 = 'file.py'
    var_18 = module_0.Config()
    var_19 = '/nonexistent/path/file.py'
    var_20 = 'test_dir'
    var_21 = [var_20]
    var_22 = frozenset(var_21)
    var_23 = module_0.Config()
    var_24 = 'test_dir'
    var_25 = var_0 / var_24
    var_26 = 'file.py'
    var_27 = var_25 / var_26
    var_28 = var_23.is_skipped(var_27)
    var_29 = 'build'
    var_30 = [var_29]
    var_31 = frozenset(var_30)
    var_32 = module_0.Config()
    var_33 = 'build'
    var_34 = var_0 / var_33
    var_35 = 'lib'
    var_36 = var_34 / var_35
    var_37 = 'file.py'
    var_38 = var_36 / var_37
    var_39 = var_32.is_skipped(var_38)
    var_40 = [var_33]
    var_41 = frozenset(var_40)
    var_42 = '*.egg-info'
    var_43 = [var_42]
    var_44 = frozenset(var_43)
    var_45 = module_0.Config()
    var_46 = 'package.egg-info'
    var_47 = 'py'
    var_48 = [var_47]
    var_49 = frozenset(var_48)
    var_50 = module_0.Config()
    var_51 = 'test.py'
    var_52 = var_0 / var_51
    var_53 = var_50.is_skipped(var_52)
    var_54 = module_0.Config()
    var_55 = 'file.py~'
    var_56 = var_0 / var_55
    var_57 = var_54.is_skipped(var_56)
    var_58 = 'skip1'
    var_59 = [var_58]
    var_60 = frozenset(var_59)
    var_61 = 'skip2'
    var_62 = [var_61]
    var_63 = frozenset(var_62)
    var_64 = module_0.Config()
    var_65 = '*.tmp'
    var_66 = [var_65]
    var_67 = frozenset(var_66)
    var_68 = '*.bak'
    var_69 = [var_68]
    var_70 = frozenset(var_69)
    var_71 = module_0.Config()
    var_72 = 'file.tmp'
    var_73 = 'file.bak'
    var_74 = 'folder/file.py'
    var_75 = [var_74]
    var_76 = frozenset(var_75)
    var_77 = module_0.Config()
    var_78 = 'folder\\file.py'
    var_79 = '/test/*.py'
    var_80 = [var_79]
    var_81 = frozenset(var_80)
    var_82 = module_0.Config()
    var_83 = 'test'
    var_84 = var_0 / var_83
    var_85 = 'file.py'
    var_86 = var_84 / var_85
    var_87 = var_82.is_skipped(var_86)
    var_88 = module_0.Config()
    var_89 = 'valid_file.py'
    var_90 = var_0 / var_89
    var_91 = var_88.is_skipped(var_90)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 'tab'
    var_6 = module_0.Config()
    var_7 = "'    '"
    var_8 = module_0.Config()
    var_9 = 88
    var_10 = 'black'
    var_11 = module_0.Config()
    var_12 = module_0.Config(config=var_11)
    var_13 = 80
    var_14 = module_0.Config()
    var_15 = 80
    var_16 = 100
    var_17 = module_0.Config()
    var_18 = 'django'
    var_19 = [var_18]
    var_20 = module_0.Config()
    var_21 = set()
    var_22 = 'Future imports'
    var_23 = module_0.Config()
    var_24 = 'future'
    var_25 = 'Standard library'
    var_26 = module_0.Config()
    var_27 = 'stdlib'
    var_28 = True
    var_29 = module_0.Config()
    var_30 = 'src'
    var_31 = 'lib'
    var_32 = [var_30, var_31]
    var_33 = module_0.Config()
    var_34 = var_33.src_paths
    var_35 = len(var_34)
    var_36 = 'FUTURE'
    var_37 = 'STDLIB'
    var_38 = 'THIRDPARTY'
    var_39 = 'FIRSTPARTY'
    var_40 = 'LOCALFOLDER'
    var_41 = [var_36, var_37, var_38, var_39, var_40]
    var_42 = module_0.Config()
    var_43 = module_0.Config()
    var_44 = 120
    var_45 = 2
    var_46 = 3
    var_47 = module_0.Config()
    var_48 = module_0.Config()
    var_49 = module_0.Config()
    var_50 = hash(var_48)
    var_51 = hash(var_49)
    var_52 = hash(var_48)
    var_53 = id(var_48)
    var_54 = 'nonexistent_file.cfg'
    var_55 = module_0.Config(var_54)
    var_56 = module_0.Config()
    var_57 = module_0.Config()
    var_58 = module_0.Config()



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it correctly finds and parses config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile = black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length = 88\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nprofile = django\n'
    var_11 = 'insert'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty_project'

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directory structure.'
    var_1 = 'deep'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile = black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nline_length = 100\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'invalid_config'
    var_2 = 'setup.cfg'
    var_3 = '[invalid\nthis is not valid ini'

def test_case_0():
    var_0 = 'Test find_all_configs with empty config files.'
    var_1 = 'empty_config'
    var_2 = 'setup.cfg'
    var_3 = ''
    var_4 = '.isort.cfg'

def test_case_0():
    var_0 = 'Test find_all_configs with empty subdirectories.'
    var_1 = 'project_no_configs'
    var_2 = 'src'
    var_3 = 'tests'
    var_4 = 'package'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_supported_filetype method with various file types.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'module.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyc'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.pyo'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_1.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'test~'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = '/nonexistent/path/file.py'
    var_15 = var_1.is_supported_filetype(var_14)
    assert var_15 is False
    var_16 = '#!/usr/bin/env python\n'
    assert var_16 is True
    var_17 = "print('hello')\n"
    var_18 = 'This is a text file\n'
    assert var_18 is False
    var_19 = 'py'
    var_20 = 'pyi'
    var_21 = 'txt'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.Config()
    var_24 = 'Test content\n'
    assert var_24 is True



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_skip.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    assert var_3 is True
    var_4 = module_0.Config()
    var_5 = [var_1]
    var_6 = frozenset(var_5)
    assert var_6 is True
    var_7 = module_0.Config()
    var_8 = 'other_file.py'
    var_9 = '*.pyc'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'test.pyc'
    var_14 = 'skip_dir'
    var_15 = 'test.py'
    var_16 = [var_14]
    assert var_16 is False
    var_17 = frozenset(var_16)
    var_18 = module_0.Config()
    var_19 = '.git'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = 'test.py~'
    var_24 = 'test.py'
    var_25 = "print('hello')"
    var_26 = module_0.Config()
    var_27 = 'skip1.py'
    var_28 = [var_27]
    var_29 = frozenset(var_28)
    var_30 = 'skip2.py'
    var_31 = [var_30]
    var_32 = frozenset(var_31)
    var_33 = module_0.Config()
    var_34 = '*.tmp'
    var_35 = [var_34]
    var_36 = frozenset(var_35)
    var_37 = '*.bak'
    var_38 = [var_37]
    var_39 = frozenset(var_38)
    var_40 = module_0.Config()
    var_41 = 'file.tmp'
    var_42 = 'file.bak'
    var_43 = 'test/skip.py'
    var_44 = [var_43]
    var_45 = frozenset(var_44)
    var_46 = module_0.Config()
    var_47 = 'test\\skip.py'
    var_48 = [var_47]
    var_49 = frozenset(var_48)
    var_50 = module_0.Config()



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = 'test_dir/test_file.py'
    var_10 = '*.pyc'
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test.pyc'
    var_14 = [var_10]
    var_15 = module_0.Config()
    var_16 = 'test.py'
    var_17 = []
    var_18 = module_0.Config()
    var_19 = 'test.py~'
    var_20 = []
    var_21 = module_0.Config()
    var_22 = '/nonexistent/path/to/file.py'
    var_23 = []
    var_24 = 'test_file.py'
    var_25 = var_1 / var_24
    var_26 = var_21.is_skipped(var_25)
    var_27 = 'skip1.py'
    var_28 = [var_27]
    var_29 = 'skip2.py'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = var_31.is_skipped(var_25)
    var_33 = [var_10]
    var_34 = '*.pyo'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = 'test.pyo'
    var_38 = var_36.is_skipped(var_25)
    var_39 = 'node_modules'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = 'src/node_modules/package.js'
    var_43 = var_41.is_skipped(var_25)
    var_44 = 'test\\file.py'
    var_45 = [var_44]
    var_46 = module_0.Config()
    var_47 = 'test/file.py'
    var_48 = var_46.is_skipped(var_25)
    var_49 = var_46.is_skipped(var_25)



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    assert var_1 is False
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'skip_dir'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = 'skip_dir/file.py'
    var_10 = '*.pyc'
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test.pyc'
    var_14 = [var_10]
    var_15 = module_0.Config()
    var_16 = 'test.py'
    var_17 = module_0.Config()
    var_18 = '/nonexistent/path/to/file.py'
    var_19 = True
    var_20 = module_0.Config()
    var_21 = '.git'
    var_22 = 'dir1'
    var_23 = [var_22]
    var_24 = 'dir2'
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = 'dir2/file.py'
    var_28 = [var_10]
    var_29 = '*.pyo'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = 'test.pyo'
    var_33 = 'test.py'
    var_34 = '# test file\n'
    var_35 = [var_34]
    assert var_35 is True
    var_36 = 'test.py'
    var_37 = '# test file\n'
    var_38 = [var_37]
    var_39 = 'parent/child'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = 'parent/child/file.py'
    var_43 = 'tests/*'
    var_44 = [var_43]
    var_45 = module_0.Config()
    var_46 = 'tests/test_file.py'
    var_47 = module_0.Config()
    var_48 = '/nonexistent/symlink'



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    assert var_7 is True
    var_8 = module_0.Config()
    var_9 = 'skip_dir'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'skip_dir/test_file.py'
    var_14 = '*.pyc'
    var_15 = [var_14]
    var_16 = frozenset(var_15)
    var_17 = module_0.Config()
    var_18 = 'test_file.pyc'
    var_19 = [var_14]
    var_20 = frozenset(var_19)
    var_21 = module_0.Config()
    var_22 = True
    var_23 = module_0.Config()
    var_24 = '.git'
    var_25 = module_0.Config()
    var_26 = '/nonexistent/path/to/file.py'
    var_27 = 'skip_me'
    var_28 = 'test.py'
    var_29 = [var_27]
    var_30 = frozenset(var_29)
    var_31 = module_0.Config()
    var_32 = 'file1.py'
    var_33 = [var_32]
    var_34 = frozenset(var_33)
    var_35 = 'file2.py'
    var_36 = [var_35]
    var_37 = frozenset(var_36)
    var_38 = module_0.Config()
    var_39 = [var_14]
    var_40 = frozenset(var_39)
    var_41 = '*.pyo'
    var_42 = [var_41]
    var_43 = frozenset(var_42)
    var_44 = module_0.Config()
    var_45 = 'test.pyc'
    var_46 = 'test.pyo'



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it finds and parses config files in directories.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = '[isort]\nline_length=100\n'
    var_8 = 'pyproject.toml'
    var_9 = '[tool.isort]\nprofile=django\n'
    var_10 = '[isort]\nprofile=flask\n'
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'Test find_all_configs with a directory containing no config files.'
    var_1 = 'empty_project'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'project_invalid'
    var_2 = 'setup.cfg'
    var_3 = '[invalid content that breaks parsing'
    var_4 = 'isort.settings._get_config_data'

def test_case_0():
    var_0 = 'Test find_all_configs finds multiple types of config files.'
    var_1 = 'multi_config'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\n'
    var_4 = 'setup.py'
    var_5 = '# setup file'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nprofile=django\n'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to verify it correctly discovers and parses config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nprofile = black\n'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nline_length = 100\n'
    var_9 = 'setup.cfg'
    var_10 = '[isort]\nprofile = django\n'
    var_11 = '[settings]\nprofile = flask\n'
    var_12 = []
    var_13 = len(var_12)

def test_case_0():
    var_0 = 'Test find_all_configs with directory containing no config files.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs when config files exist but are malformed.'
    var_1 = 'project'
    var_2 = '.isort.cfg'
    var_3 = 'this is not valid config content [[['

def test_case_0():
    var_0 = 'Test find_all_configs with deeply nested directory structure.'
    var_1 = 'root'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile = black\n'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nprofile = django\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nline_length = 88\n'



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_supported_filetype method.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = 'import os'
    var_4 = 'test.pyc'
    var_5 = b'\x00\x00\x00\x00'
    var_6 = 'test.py~'
    var_7 = 'test.txt'
    var_8 = 'plain text'
    var_9 = 'test.sh'
    var_10 = '#!/usr/bin/env python\nimport os'
    var_11 = 'nonexistent.py'
    var_12 = 'test.unknown'
    var_13 = 'no shebang here'
    var_14 = 'py'
    var_15 = 'pyx'
    var_16 = [var_14, var_15]
    var_17 = module_0.Config()
    var_18 = 'test.pyx'
    var_19 = 'cython code'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to verify it finds and parses all config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '.isort.cfg'
    var_6 = '[settings]\nline_length=80\n'
    var_7 = 'setup.cfg'
    var_8 = '[isort]\nline_length=100\n'
    var_9 = 'pyproject.toml'
    var_10 = '[tool.isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'invalid'
    var_2 = '.isort.cfg'
    var_3 = '[invalid\nbroken syntax'

def test_case_0():
    var_0 = 'Test find_all_configs with pyproject.toml files.'
    var_1 = 'pyproject_test'
    var_2 = 'subdir'
    var_3 = 'pyproject.toml'
    var_4 = "[tool.isort]\nprofile='black'\n"
    var_5 = "[tool.isort]\nprofile='django'\n"

def test_case_0():
    var_0 = 'Test find_all_configs with setup.cfg files.'
    var_1 = 'setup_cfg_test'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nline_length=88\nprofile=black\n'

def test_case_0():
    var_0 = 'Test find_all_configs with tox.ini files.'
    var_1 = 'tox_test'
    var_2 = 'tox.ini'
    var_3 = '[isort]\nline_length=100\n'

def test_case_0():
    var_0 = 'Test find_all_configs with multiple config file types.'
    var_1 = 'mixed'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length=80\n'
    var_4 = 'sub1'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nline_length=88\n'
    var_7 = 'sub2'
    var_8 = 'pyproject.toml'
    var_9 = '[tool.isort]\nline_length=100\n'



# Parsed testcases at query #28
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.pyc'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = '/nonexistent/path/file.py'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'py'
    var_10 = 'pyi'
    var_11 = 'txt'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Config()
    var_14 = 'test.txt'
    var_15 = var_13.is_supported_filetype(var_14)
    assert var_15 is True
    var_16 = [var_9]
    var_17 = module_0.Config()
    var_18 = var_17.is_supported_filetype(var_1)
    assert var_18 is False
    var_19 = "#!/usr/bin/env python\nprint('hello')"
    assert var_19 is True
    var_20 = 'some text'
    var_21 = [var_9, var_11]
    var_22 = module_0.Config()
    var_23 = 'some text'
    assert var_23 is True



# Parsed testcases at query #29
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    assert var_1 is True
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = module_0.Config()
    var_9 = 'test_dir'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'test_dir/file.py'
    var_14 = '*.pyc'
    var_15 = [var_14]
    var_16 = frozenset(var_15)
    var_17 = module_0.Config()
    var_18 = 'test_file.pyc'
    var_19 = [var_14]
    var_20 = frozenset(var_19)
    var_21 = module_0.Config()
    var_22 = module_0.Config()
    var_23 = '/nonexistent/path/to/file.py'
    var_24 = module_0.Config()
    var_25 = True
    var_26 = module_0.Config()
    var_27 = '.git'
    var_28 = 'file1.py'
    var_29 = [var_28]
    var_30 = frozenset(var_29)
    var_31 = 'file2.py'
    var_32 = [var_31]
    var_33 = frozenset(var_32)
    var_34 = module_0.Config()
    var_35 = [var_14]
    var_36 = frozenset(var_35)
    var_37 = '*.pyo'
    var_38 = [var_37]
    var_39 = frozenset(var_38)
    var_40 = module_0.Config()
    var_41 = 'test.pyc'
    var_42 = 'test.pyo'
    var_43 = [var_9]
    var_44 = frozenset(var_43)
    var_45 = module_0.Config()
    var_46 = 'test_dir\\file.py'
    var_47 = '/test/*'
    var_48 = [var_47]
    var_49 = frozenset(var_48)
    var_50 = module_0.Config()
    var_51 = 'test/file.py'



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = module_0.Config()
    var_9 = 'skip_dir'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = 'skip_dir/test_file.py'
    var_13 = '*.pyc'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = module_0.Config()
    var_17 = 'test.pyc'
    var_18 = 'test.py'
    var_19 = var_0 / var_18
    var_20 = '*.pyc'
    var_21 = [var_20]
    var_22 = frozenset(var_21)
    var_23 = var_16.is_skipped(var_19)
    assert var_23 is False
    var_24 = module_0.Config()
    var_25 = 'nonexistent_file_xyz.py'
    var_26 = 'test.py~'
    var_27 = var_0 / var_26
    var_28 = var_24.is_skipped(var_27)
    assert var_28 is True
    var_29 = 'node_modules'
    var_30 = [var_29]
    var_31 = frozenset(var_30)
    var_32 = 'node_modules/package/index.js'
    var_33 = 'test.py'
    var_34 = var_0 / var_33
    var_35 = 'skip1'
    var_36 = [var_35]
    var_37 = frozenset(var_36)
    var_38 = 'skip2'
    var_39 = [var_38]
    var_40 = frozenset(var_39)
    var_41 = 'test.py'
    var_42 = var_0 / var_41
    var_43 = var_24.is_skipped(var_42)
    assert var_43 is False



# Parsed testcases at query #31
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various scenarios.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 'tab'
    var_6 = module_0.Config()
    var_7 = "'    '"
    var_8 = module_0.Config()
    var_9 = 88
    var_10 = module_0.Config()
    var_11 = module_0.Config(config=var_10)
    var_12 = True
    var_13 = module_0.Config()
    var_14 = module_0.Config()
    var_15 = var_14.src_paths
    var_16 = len(var_15)
    var_17 = 'mymodule'
    var_18 = [var_17]
    var_19 = module_0.Config()
    var_20 = 'Future imports'
    var_21 = module_0.Config()
    var_22 = 'End of stdlib'
    var_23 = module_0.Config()
    var_24 = '/tmp'
    var_25 = module_0.Config()
    var_26 = 'black'
    var_27 = module_0.Config()
    var_28 = '2'
    var_29 = module_0.Config()
    var_30 = 'FUTURE'
    var_31 = 'STDLIB'
    var_32 = 'THIRDPARTY'
    var_33 = 'FIRSTPARTY'
    var_34 = 'LOCALFOLDER'
    var_35 = [var_30, var_31, var_32, var_33, var_34]
    var_36 = module_0.Config()
    var_37 = 'venv'
    var_38 = 'build'
    var_39 = [var_37, var_38]
    var_40 = module_0.Config()
    var_41 = '*.egg-info'
    var_42 = [var_41]
    var_43 = module_0.Config()
    var_44 = 120
    var_45 = 2
    var_46 = 'myproject'
    var_47 = [var_46]
    var_48 = module_0.Config()
    var_49 = module_0.Config()
    var_50 = 'sources'
    var_51 = hasattr(var_49, var_50)
    var_52 = False
    var_53 = module_0.Config()
    var_54 = 'natural'
    var_55 = module_0.Config()



# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = '_known_patterns'
    var_3 = hasattr(var_1, var_2)
    var_4 = '_section_comments'
    var_5 = hasattr(var_1, var_4)
    var_6 = '_section_comments_end'
    var_7 = hasattr(var_1, var_6)
    var_8 = '_skips'
    var_9 = hasattr(var_1, var_8)
    var_10 = '_skip_globs'
    var_11 = hasattr(var_1, var_10)
    var_12 = '_sorting_function'
    var_13 = hasattr(var_1, var_12)
    var_14 = 100
    var_15 = 4
    var_16 = module_0.Config()
    var_17 = '2'
    var_18 = module_0.Config()
    var_19 = "'    '"
    var_20 = module_0.Config()
    var_21 = 'tab'
    var_22 = module_0.Config()
    var_23 = module_0._Config()
    var_24 = 120
    var_25 = module_0.Config(config=var_23)
    var_26 = 80
    var_27 = 100
    var_28 = module_0.Config()
    var_29 = '/nonexistent/path/config'
    var_30 = module_0.Config(settings_path=var_29)
    var_31 = 'nonexistent_profile'
    var_32 = module_0.Config()
    var_33 = 'nonexistent_formatter'
    var_34 = module_0.Config()
    var_35 = 'my_module'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = 'Standard Library'
    var_39 = module_0.Config()
    var_40 = 'End Standard Library'
    var_41 = module_0.Config()
    var_42 = 'value'
    var_43 = module_0.Config()
    var_44 = True
    var_45 = module_0.Config()
    var_46 = module_0.Config()
    var_47 = var_46.src_paths
    var_48 = len(var_47)
    var_49 = module_0.Config()



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config constructor with various initialization methods.'
    var_1 = module_0.Config()
    var_2 = 'wrap_length'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'line_length'
    var_5 = hasattr(var_1, var_4)
    var_6 = 100
    var_7 = 80
    var_8 = module_0.Config()
    var_9 = 88
    var_10 = 3
    var_11 = module_0.Config()
    var_12 = module_0.Config(config=var_11)
    var_13 = True
    var_14 = module_0.Config()
    var_15 = 'black'
    var_16 = module_0.Config()
    var_17 = 4
    var_18 = module_0.Config()
    var_19 = 'tab'
    var_20 = module_0.Config()
    var_21 = '  '
    var_22 = module_0.Config()
    var_23 = 'django'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = set()
    var_27 = 'from __future__ imports'
    var_28 = module_0.Config()
    var_29 = 'stdlib footer'
    var_30 = module_0.Config()
    var_31 = module_0.Config()
    var_32 = var_31.src_paths
    var_33 = module_0.Config()
    var_34 = '/nonexistent/path/that/does/not/exist'
    var_35 = module_0.Config(settings_path=var_34)
    var_36 = 80
    var_37 = 100
    var_38 = module_0.Config()
    var_39 = 120
    var_40 = 2
    var_41 = False
    var_42 = module_0.Config()
    var_43 = module_0.Config()
    var_44 = 'FUTURE'
    var_45 = 'STDLIB'
    var_46 = 'THIRDPARTY'
    var_47 = 'FIRSTPARTY'
    var_48 = 'LOCALFOLDER'
    var_49 = [var_44, var_45, var_46, var_47, var_48]
    var_50 = module_0.Config()
    var_51 = module_0.Config()
    var_52 = 'myapp'
    var_53 = [var_52]
    var_54 = 'requests'
    var_55 = [var_54]
    var_56 = module_0.Config()



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it discovers and parses config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nline_length=100\n'
    var_9 = '.isort.cfg'
    var_10 = '[settings]\nprofile=django\n'

def test_case_0():
    var_0 = 'Test find_all_configs when no config files exist.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs with a single config file.'
    var_1 = 'project'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black\nline_length=88\n'

def test_case_0():
    var_0 = 'Test find_all_configs with config files at multiple directory levels.'
    var_1 = 'root'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = 'pyproject.toml'
    var_8 = '[tool.isort]\nprofile=django\n'
    var_9 = '.isort.cfg'
    var_10 = '[settings]\nprofile=flask\n'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'project'
    var_2 = 'setup.cfg'
    var_3 = '[invalid content {{{'
    var_4 = 'isort.settings._get_config_data'

def test_case_0():
    var_0 = 'Test find_all_configs with empty config files.'
    var_1 = 'project'
    var_2 = 'setup.cfg'
    var_3 = ''
    var_4 = 'pyproject.toml'
    var_5 = '.isort.cfg'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test find_all_configs function to ensure it correctly finds and parses config files.'
    var_1 = 'project'
    var_2 = 'subdir1'
    var_3 = 'subdir2'
    var_4 = 'nested'
    var_5 = '\n[tool.isort]\nprofile = "black"\nline_length = 88\n'
    var_6 = 'pyproject.toml'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nprofile=black\n'

def test_case_0():
    var_0 = 'Test find_all_configs with an empty directory.'
    var_1 = 'empty'

def test_case_0():
    var_0 = 'Test find_all_configs handles invalid config files gracefully.'
    var_1 = 'test_invalid'
    var_2 = 'pyproject.toml'
    var_3 = 'invalid toml content [[['

def test_case_0():
    var_0 = 'Test find_all_configs with multiple nested directory levels.'
    var_1 = 'root'
    var_2 = 'level1'
    var_3 = 'level2'
    var_4 = 'level3'
    var_5 = '[tool.isort]\nprofile = black\n'
    var_6 = 'pyproject.toml'
    var_7 = '.isort.cfg'
    var_8 = '[settings]\nprofile=black\n'
    var_9 = 'setup.cfg'
    var_10 = '[isort]\nprofile=black\n'



# Parsed testcases at query #36
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = 'not_skipped.py'
    var_5 = var_0 / var_4
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = var_3.is_skipped(var_5)
    assert var_9 is False
    var_10 = '*.pyc'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = 'skip_me.py'
    var_16 = [var_15]
    var_17 = frozenset(var_16)
    var_18 = module_0.Config()
    var_19 = module_0.Config()
    var_20 = '/nonexistent/file/path.py'
    var_21 = 'skip_folder'
    var_22 = var_0 / var_21
    var_23 = [var_21]
    var_24 = frozenset(var_23)
    var_25 = 'file.py'
    var_26 = var_22 / var_25
    var_27 = var_19.is_skipped(var_26)
    assert var_27 is True
    var_28 = 'test.py~'
    var_29 = var_0 / var_28
    var_30 = var_19.is_skipped(var_29)
    assert var_30 is True
    var_31 = 'regular.py'
    var_32 = var_0 / var_31
    var_33 = var_19.is_skipped(var_32)
    assert var_33 is False
    var_34 = '__pycache__/*'
    var_35 = [var_34]
    var_36 = frozenset(var_35)
    var_37 = module_0.Config()
    var_38 = '__pycache__/test.pyc'
    var_39 = [var_0]
    var_40 = frozenset(var_39)
    var_41 = module_0.Config()



# Parsed testcases at query #37
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_skipped method with various skip conditions.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = 'file.py'
    var_10 = '*.pyc'
    var_11 = '__pycache__/*'
    var_12 = [var_10, var_11]
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = [var_10]
    var_16 = module_0.Config()
    var_17 = 'test.py'
    var_18 = []
    var_19 = module_0.Config()
    var_20 = 'non_existent.py'
    var_21 = []
    var_22 = module_0.Config()
    var_23 = 'target.py'
    var_24 = 'link.py'
    var_25 = 'file1.py'
    var_26 = [var_25]
    var_27 = 'file2.py'
    var_28 = [var_27]
    var_29 = module_0.Config()
    var_30 = [var_10]
    var_31 = '*.pyo'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = 'test.pyo'
    var_35 = 'nested'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = 'deep'
    var_39 = True
    var_40 = []
    var_41 = [var_1]
    var_42 = module_0.Config()



# Parsed testcases at query #38
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_supported_filetype method of Config class.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'pyc'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'test.pyc'
    var_8 = var_6.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'py'
    var_10 = 'pyi'
    var_11 = [var_9, var_10]
    var_12 = module_0.Config()
    var_13 = var_12.is_supported_filetype(var_2)
    assert var_13 is True
    var_14 = 'test.pyi'
    var_15 = var_12.is_supported_filetype(var_14)
    assert var_15 is True
    var_16 = 'test.py~'
    var_17 = var_1.is_supported_filetype(var_16)
    assert var_17 is False
    var_18 = '/nonexistent/path/file.py'
    var_19 = var_1.is_supported_filetype(var_18)
    assert var_19 is False
    var_20 = '#!/usr/bin/env python\n'
    assert var_20 is True
    var_21 = "print('hello')\n"
    var_22 = "print('hello')\n"
    assert var_22 is True
    var_23 = 'some random content\n'
    assert var_23 is False



# Parsed testcases at query #39
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'normal_file.py'
    var_7 = 'skip_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'skip_dir/file.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'test.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = '/non/existent/path/file.py'
    var_20 = 'extended_skip.py'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = '__pycache__/*'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = '__pycache__/module.pyc'
    var_27 = '.git'
    var_28 = var_0 / var_27
    var_29 = True
    var_30 = module_0.Config()
    var_31 = var_30.is_skipped(var_28)
    assert var_31 is True
    var_32 = module_0.Config()
    var_33 = 'test.py~'
    var_34 = 'test.py'
    var_35 = var_0 / var_34
    var_36 = "print('test')"
    var_37 = module_0.Config()
    var_38 = var_37.is_skipped(var_35)
    assert var_38 is False
    var_39 = 'test\\file.py'
    var_40 = [var_39]
    var_41 = module_0.Config()
    var_42 = 'test/file.py'
    var_43 = '/some/dir'
    var_44 = 'skip_me.py'
    var_45 = [var_44]
    var_46 = module_0.Config()
    var_47 = '/some/dir/skip_me.py'
    var_48 = 'file1.py'
    var_49 = 'file2.py'
    var_50 = [var_48, var_49]
    var_51 = module_0.Config()
    var_52 = 'file3.py'
    var_53 = '*.egg-info/*'
    var_54 = [var_53]
    var_55 = module_0.Config()
    var_56 = 'package.egg-info/PKG-INFO'



# Parsed testcases at query #40
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config class constructor with various initialization scenarios.'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = 4
    var_4 = module_0.Config()
    var_5 = 88
    var_6 = module_0.Config()
    var_7 = module_0.Config(config=var_6)
    var_8 = 'black'
    var_9 = module_0.Config()
    var_10 = 2
    var_11 = module_0.Config()
    var_12 = "'    '"
    var_13 = module_0.Config()
    var_14 = 'tab'
    var_15 = module_0.Config()
    var_16 = 80
    var_17 = 100
    var_18 = module_0.Config()
    var_19 = 'mypackage'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = 'Standard Library'
    var_23 = module_0.Config()
    var_24 = 'Third Party Footer'
    var_25 = module_0.Config()
    var_26 = module_0.Config()
    var_27 = var_26.src_paths
    var_28 = module_0.Config()
    var_29 = True
    var_30 = 'nonexistent_profile'
    var_31 = module_0.Config()
    var_32 = 120
    var_33 = 8
    var_34 = 'migrations'
    var_35 = [var_34]
    var_36 = 'build'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = module_0.Config()
    var_40 = module_0.Config()
    var_41 = module_0.Config()
    var_42 = module_0.Config()
    var_43 = module_0.Config()
    var_44 = 'FUTURE'
    var_45 = 'STDLIB'
    var_46 = 'THIRDPARTY'
    var_47 = 'FIRSTPARTY'
    var_48 = 'LOCALFOLDER'
    var_49 = [var_44, var_45, var_46, var_47, var_48]
    var_50 = module_0.Config()
    var_51 = module_0.Config()
    var_52 = 'py'
    var_53 = 'pyi'
    var_54 = [var_52, var_53]
    var_55 = module_0.Config()



# Parsed testcases at query #41
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_supported_filetype method of Config class.'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'module.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyc'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.pyo'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_1.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'backup~'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'py'
    var_15 = 'pyi'
    var_16 = 'txt'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Config()
    var_19 = 'test.txt'
    var_20 = var_18.is_supported_filetype(var_19)
    assert var_20 is True
    var_21 = 'test.md'
    var_22 = var_18.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = [var_14]
    var_24 = module_0.Config()
    var_25 = var_24.is_supported_filetype(var_2)
    assert var_25 is False
    var_26 = '/nonexistent/path/file.py'
    var_27 = var_1.is_supported_filetype(var_26)
    assert var_27 is False
    var_28 = b'#!/usr/bin/env python\n'
    assert var_28 is True



# Parsed testcases at query #42
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the is_skipped method of Config class.'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'skip_dir'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'skip_dir/test_file.py'
    var_11 = '*.pyc'
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'compiled.pyc'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = 'source.py'
    var_18 = module_0.Config()
    var_19 = '/nonexistent/path/to/file.py'
    var_20 = True
    var_21 = module_0.Config()
    var_22 = '.git'
    var_23 = 'file1.py'
    var_24 = [var_23]
    var_25 = 'file2.py'
    var_26 = [var_25]
    var_27 = module_0.Config()
    var_28 = [var_11]
    var_29 = '*.pyo'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = 'compiled.pyo'
    var_33 = 'test/file.py'
    var_34 = [var_33]
    var_35 = module_0.Config()
    var_36 = '**/test_*.py'
    var_37 = [var_36]
    var_38 = module_0.Config()
    var_39 = 'tests/test_example.py'
    var_40 = 'skip_me.py'
    var_41 = [var_40]
    var_42 = var_2 / var_40
    var_43 = var_38.is_skipped(var_42)
    assert var_43 is True



# Parsed testcases at query #43
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test Config.is_supported_filetype method'
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'module.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True
    var_6 = 'test.pyc'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.pyo'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'test.py~'
    var_11 = var_1.is_supported_filetype(var_10)
    assert var_11 is False
    var_12 = 'module~'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'py'
    var_15 = 'pyi'
    var_16 = 'txt'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Config()
    var_19 = 'test.txt'
    var_20 = var_18.is_supported_filetype(var_19)
    assert var_20 is True
    var_21 = 'test.md'
    var_22 = var_18.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'pyc'
    var_24 = 'pyo'
    var_25 = [var_23, var_24, var_16]
    var_26 = module_0.Config()
    var_27 = var_26.is_supported_filetype(var_19)
    assert var_27 is False
    var_28 = var_26.is_supported_filetype(var_2)
    assert var_28 is True
    var_29 = '/nonexistent/path/to/file.py'
    var_30 = var_1.is_supported_filetype(var_29)
    assert var_30 is False
    var_31 = 'Makefile'
    var_32 = var_1.is_supported_filetype(var_31)
    assert var_32 is False
    var_33 = 'README'
    var_34 = var_1.is_supported_filetype(var_33)
    assert var_34 is False



# Parsed testcases at query #44
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
    var_5 = 'test.pyc'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.py~'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = b'import os\n'
    assert var_9 is True
    var_10 = b'#!/usr/bin/env python\n'
    assert var_10 is True
    var_11 = b'This is just text\n'
    assert var_11 is False
    var_12 = '/nonexistent/path/file.py'
    var_13 = var_0.is_supported_filetype(var_12)
    assert var_13 is False
    var_14 = 'py'
    var_15 = 'pyi'
    var_16 = 'txt'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Config()
    var_19 = 'test.txt'
    var_20 = var_18.is_supported_filetype(var_19)
    assert var_20 is True
    var_21 = [var_14]
    var_22 = module_0.Config()
    var_23 = var_22.is_supported_filetype(var_11)
    assert var_23 is False



