####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_is_supported_filetype_with_python. Retrieved 2/10 statements.
# Partially parsed test_config_is_supported_filetype_with_unsupported_extension. Retrieved 2/10 statements.
# Partially parsed test_config_known_patterns_property. Retrieved 3/6 statements.
# Partially parsed test_config_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_section_comments_end_property. Retrieved 2/3 statements.
# Partially parsed test_config_skips_property. Retrieved 5/6 statements.
# Partially parsed test_config_skip_globs_property. Retrieved 5/6 statements.
# Partially parsed test_config_parse_known_pattern_with_file. Retrieved 2/3 statements.
# Partially parsed test_config_is_skipped_with_backup_file_path. Retrieved 4/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = True
    var_2 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = True
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = b'import os\n'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = b'some text\n'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.known_patterns
    var_2 = 2

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = var_3.skips

import isort.settings as module_0

def test_case_0():
    var_0 = '*.egg-info'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = var_3.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_sort_order'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'django'

import isort.settings as module_0

def test_case_0():
    var_0 = "'    '"
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = 3
    var_2 = True
    var_3 = module_0.Config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_line_43_predicate_evaluates_to_false. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = '/test/settings.cfg'
    var_4 = True
    var_5 = var_0 and var_1
    assert var_5 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py39'
    var_8 = None
    var_9 = {var_0: var_7, var_1: var_8, var_2: var_8, var_3: var_8, var_4: var_8, var_5: var_8, var_6: var_8}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_line_98_evaluates_to_true. Retrieved 13/15 statements.


def test_case_0():
    var_0 = 'known_'
    var_1 = 'known_custom_section'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 'known_standard_library'
    var_7 = 'known_future_library'
    var_8 = 'known_third_party'
    var_9 = 'known_first_party'
    var_10 = 'known_local_folder'
    var_11 = (var_6, var_7, var_8, var_9, var_10)
    var_12 = var_1 not in var_11



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = 'future'
    var_1 = 'stdlib'
    var_2 = '# Future imports'
    var_3 = '# Standard library'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyc'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.pyc'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.txt'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/nonexistent/path/to/file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = vars(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = vars(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'known_other'
    var_2 = hasattr(var_0, var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import_headings'
    var_2 = hasattr(var_0, var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import_footers'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_import_footer_prefix_condition_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'import_footer_'
    var_1 = 'import_footer_future'
    var_2 = 'import_footer_stdlib'
    var_3 = 'Future imports'
    var_4 = 'Standard library imports'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_165_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = '/nonexistent/file.txt'
    var_1 = '/nonexistent'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_skipped_with_skip_path. Retrieved 4/7 statements.
# Partially parsed test_is_skipped_with_non_existent_path. Retrieved 2/5 statements.
# Partially parsed test_is_skipped_with_skip_glob_pattern. Retrieved 5/15 statements.
# Partially parsed test_is_skipped_with_directory_in_parents. Retrieved 4/14 statements.
# Partially parsed test_is_skipped_with_skip_folder. Retrieved 5/17 statements.
# Partially parsed test_is_skipped_with_valid_file. Retrieved 2/12 statements.
# Partially parsed test_is_skipped_with_extend_skip. Retrieved 4/14 statements.
# Partially parsed test_is_skipped_with_extend_skip_glob. Retrieved 5/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/non/existent/path/file.py'

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'w'
    var_2 = '*.pyc'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'w'
    var_2 = [var_0]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'skip_me'
    var_1 = 'test_file.py'
    var_2 = 'w'
    var_3 = [var_0]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'w'

def test_case_0():
    var_0 = 'excluded.py'
    var_1 = 'w'
    var_2 = [var_0]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_file.pyc'
    var_1 = 'w'
    var_2 = '*.pyc'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_config_settings_predicate_line_76. Retrieved 15/23 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = {}
    var_5 = bool(var_4)
    assert var_5 is False
    var_6 = 'profile'
    var_7 = 'line_length'
    var_8 = 'skip'
    var_9 = 'black'
    var_10 = 88
    var_11 = 'migrations'
    var_12 = [var_11]
    var_13 = {var_6: var_9, var_7: var_10, var_8: var_12}
    var_14 = bool(var_13)
    assert var_14 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_setting'
    var_8 = 'py311'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is True
    var_2 = 'True'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is True
    var_4 = 'TRUE'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is True
    var_6 = '1'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is True
    var_8 = 'yes'
    var_9 = module_0._as_bool(var_8)
    assert var_9 is True
    var_10 = 'y'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is True
    var_12 = 'on'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is False
    var_2 = 'False'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is False
    var_4 = 'FALSE'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is False
    var_6 = '0'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is False
    var_8 = 'no'
    var_9 = module_0._as_bool(var_8)
    assert var_9 is False
    var_10 = 'n'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is False
    var_12 = 'off'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._as_bool(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_bool(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'YeS'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is True
    var_2 = 'nO'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is False
    var_4 = 'On'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is True
    var_6 = 'oFf'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/10 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_bool_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_frozenset_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_editorconfig_indent_space. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_tab. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_digit. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_multiple_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_wildcard_section. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test.toml'
    var_1 = '[tool.isort]\nline_length = 100\nskip = ["file1.py", "file2.py"]\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 120\nprofile = black\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nuse_parentheses = true\nbalanced_wrapping = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_django = django,rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = file1.py,file2.py,file3.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = 88\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 2\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\ncomment_prefix = "# "\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 100\n[other]\nvalue = test\n'
    var_2 = 'isort'
    var_3 = 'other'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[*.{py,pyi}]\nline_length = 120\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100\nprofile = "black"\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_init_creates_src_paths_default. Retrieved 1/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '311'
    var_2 = module_0.Config(config=var_0)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nprofile = "black"\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'black'
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'my_module'
    var_1 = [var_0]
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Standard Library'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of Standard Library'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'django'
    var_2 = True
    var_3 = 'migrations'
    var_4 = [var_3]
    var_5 = 2
    var_6 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = 'CUSTOM'
    var_6 = (var_0, var_1, var_2, var_3, var_4, var_5)
    var_7 = module_0.Config()



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.Config(var_0, var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 17/24 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_other_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'Config'
    var_13 = ()
    var_14 = '__init__'
    var_15 = lambda self, settings_file='', settings_path='', config=None, **config_overrides: var_9
    var_16 = {var_14: var_15}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_indent_in_combined_config_evaluates_to_true. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'indent'
    var_1 = 'quiet'
    var_2 = 4
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2}



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = 'future'
    var_1 = 'stdlib'
    var_2 = '# Future imports'
    var_3 = '# Standard library'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_known_other_predicate_evaluates_to_true. Retrieved 13/21 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/test'
    var_1 = {}
    var_2 = 'known_django'
    var_3 = 'profile'
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = ''
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.Config(**var_7)
    var_9 = 'known_other'
    var_10 = hasattr(var_8, var_9)
    var_11 = [var_4]
    var_12 = frozenset(var_11)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_skipped_predicate_line_3_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '/some/file/path.py'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_skipped_with_skip_path. Retrieved 3/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = "'  '"
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'known_other'
    var_4 = hasattr(var_2, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = module_0.Config()
    var_2 = 'import_headings'
    var_3 = hasattr(var_1, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'Standard library'
    var_1 = module_0.Config()
    var_2 = 'import_footers'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_73_evaluates_to_true. Retrieved 10/28 statements.


def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\n'
    var_2 = 'indent_style = space\n'
    var_3 = 'indent_size = 4\n'
    var_4 = 'force_alphabetical_sort = true\n'
    assert var_4 is True
    var_5 = 'force_single_line = false\n'
    var_6 = '*.py'
    var_7 = (var_6,)
    var_8 = 'force_alphabetical_sort'
    var_9 = 'force_single_line'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_setting'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_ini_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_tab_indent. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_boolean_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_boolean_string_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_frozenset_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_integer. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_single_quotes. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_double_quotes. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_wildcard_extension_matching. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_list_conversion_with_newlines. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_editorconfig_tab_width_fallback. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\nline_length = 88\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\nline_length = 88\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip_gitignore = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip_gitignore = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_django = django,rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nignore_whitespace = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 2\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = "[isort]\ncomment_prefix = '# '\n"
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\ncomment_prefix = "# "\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[*.{py,pyx}]\nprofile = black\n'
    var_2 = '*.{py,pyx}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_third_party = requests\n    django\n    rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_third_party'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\nline_length = 100\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\ntab_width = 4\n'
    var_2 = '*.py'
    var_3 = (var_2,)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/9 statements.
# Partially parsed test_config_init_lazy_properties. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = "'  '"
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'DJANGO'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.known_patterns

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = 'build'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = var_4.skips

import isort.settings as module_0

def test_case_0():
    var_0 = '*.egg-info'
    var_1 = [var_0]
    var_2 = 'dist'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = var_4.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'Future imports'
    var_3 = 'Standard library'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Config()
    var_6 = var_5.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'End future'
    var_3 = 'End stdlib'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Config()
    var_6 = var_5.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 2
    var_2 = 3
    var_3 = True
    var_4 = module_0.Config()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_line_159_predicate_evaluates_to_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = '/path/to/config/file.cfg'
    var_2 = {var_0: var_1}
    var_3 = None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_with_absolute_path_match. Retrieved 4/7 statements.
# Partially parsed test_is_skipped_with_no_skip. Retrieved 4/7 statements.
# Partially parsed test_is_skipped_with_directory_component. Retrieved 5/8 statements.
# Partially parsed test_is_skipped_with_glob_pattern. Retrieved 5/8 statements.
# Partially parsed test_is_skipped_with_nonexistent_path. Retrieved 4/7 statements.
# Partially parsed test_is_skipped_with_skip_glob_pattern. Retrieved 5/8 statements.
# Partially parsed test_is_skipped_with_normalized_path. Retrieved 4/8 statements.
# Partially parsed test_is_skipped_returns_boolean. Retrieved 6/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = module_0.Config()
    var_3 = 'test_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = '__pycache__/module.pyc'

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = 'test.pyc'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = module_0.Config()
    var_3 = '/nonexistent/path/to/file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_*.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()
    var_4 = 'test_module.py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = []
    var_3 = frozenset(var_2)
    var_4 = module_0.Config()
    var_5 = 'nonexistent_file_xyz.py'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_known_patterns_property. Retrieved 2/3 statements.
# Partially parsed test_config_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_section_comments_end_property. Retrieved 2/3 statements.
# Partially parsed test_config_skips_property. Retrieved 2/3 statements.
# Partially parsed test_config_skip_globs_property. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile_xyz'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/setup.cfg'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.known_patterns

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skips

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_sort'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_ini_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_tab_indent. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_boolean_value. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_value. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 5/10 statements.
# Partially parsed test_get_config_data_editorconfig_glob_pattern. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_multiline_list_value. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 88\nskip = ["file1.py", "file2.py"]\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 100\nindent = 4\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 80\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_single_line = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = file1.py,file2.py,file3.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 3\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\ncomment_prefix = "# "\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 120\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyi}]\nindent_style = space\nindent_size = 2\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = \n    file1.py\n    file2.py\n    file3.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_supported_filetype. Retrieved 12/33 statements.


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
    var_9 = 'import os\n'
    assert var_9 is True
    var_10 = b'#!/usr/bin/env python\nimport os\n'
    assert var_10 is True
    var_11 = 'some text\n'
    assert var_11 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/6 statements.
# Failed to parse test_config_init_with_src_paths.
# Failed to parse test_config_init_with_directory.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=80\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '# Future'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '# End stdlib'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 120
    var_1 = 3
    var_2 = True
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.Config()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_deprecated_options_used_predicate. Retrieved 15/18 statements.


def test_case_0():
    var_0 = 'old_option1'
    var_1 = 'old_option2'
    var_2 = {var_0, var_1}
    var_3 = 'valid_option'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = [option for option in var_7 if option in var_2]
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'valid_option1'
    var_11 = 'valid_option2'
    var_12 = {var_10: var_4, var_11: var_5}
    var_13 = [option for option in var_12 if option in var_2]
    var_14 = bool(var_13)
    assert var_14 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_supported_filetype_with_shebang. Retrieved 2/11 statements.
# Partially parsed test_is_supported_filetype_without_shebang. Retrieved 2/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'test.py'
    var_5 = var_3.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyc'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.pyc'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/nonexistent/path/to/file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '#!/usr/bin/env python\nimport os'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'not a python file'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_other_attr'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_setting'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 8/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'pyc'
    var_4 = 'pyo'
    var_5 = [var_3, var_4]
    var_6 = 'test.pyc'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is False



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'hello,world,test'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'hello\nworld\ntest'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'hello,world\ntest,example'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '  hello  ,  world  ,  test  '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'hello,,world,,test'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = 'test'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = '  world  '
    var_2 = '  test  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '   ,   ,   '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '\n\n\n'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '  hello  '
    var_1 = module_0._as_list(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_config_init_with_settings_path.
# Partially parsed test_config_init_preserves_sources. Retrieved 6/7 statements.
# Failed to parse test_config_init_with_src_paths.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()
    var_3 = 'sources'
    var_4 = hasattr(var_2, var_3)
    var_5 = var_2.sources

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'DJANGO'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'black'
    var_3 = 'migrations'
    var_4 = [var_3]
    var_5 = 'build'
    var_6 = [var_5]
    var_7 = module_0.Config()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_post_init_auto_py_version. Retrieved 3/4 statements.
# Partially parsed test_post_init_vertical_grid_grouped_no_comma. Retrieved 1/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '2.7'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = frozenset()
    var_2 = module_0._Config(var_0, known_standard_library=var_1)
    var_3 = var_2.known_standard_library
    var_4 = len(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = frozenset(var_2)
    var_4 = '3.8'
    var_5 = module_0._Config(var_4, known_standard_library=var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = True
    var_2 = module_0._Config(var_0, force_alphabetical_sort=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 100
    var_2 = 80
    var_3 = module_0._Config(var_0, line_length=var_2, wrap_length=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 80
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 60
    var_2 = 80
    var_3 = module_0._Config(var_0, line_length=var_2, wrap_length=var_1)

def test_case_0():
    var_0 = '3.8'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_formatter_in_combined_config_evaluates_to_true. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = var_0 in var_3
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_predicate_at_line_165_evaluates_to_false.




# Parsed testcases at query #17
#--------------------------

# Failed to parse test_multi_line_output_vertical_grid_grouped_no_comma.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 24/31 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'TestConfig'
    var_13 = ()
    var_14 = '__init__'
    var_15 = '_known_patterns'
    var_16 = None
    var_17 = '_section_comments'
    var_18 = '_section_comments_end'
    var_19 = '_skips'
    var_20 = '_skip_globs'
    var_21 = '_sorting_function'
    var_22 = lambda self, config=None, **kwargs: (setattr(self, var_15, var_16), setattr(self, var_17, var_16), setattr(self, var_18, var_16), setattr(self, var_19, var_16), setattr(self, var_20, var_16), setattr(self, var_21, var_16))
    var_23 = {var_14: var_22}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_supported_filetype_opens_file. Retrieved 6/31 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'is_supported_filetype'
    var_1 = 'supported_extensions'
    var_2 = 'blocked_extensions'
    var_3 = [var_0, var_1, var_2]
    var_4 = b'#!/usr/bin/env python\n'
    var_5 = module_0.Config()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_deprecated_options_used_predicate_evaluates_to_true. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'some_deprecated_option'
    var_1 = [var_0]
    var_2 = 'some_other_option'
    var_3 = 'value'
    var_4 = 'value2'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = [option for option in var_5 if option in var_1]
    var_7 = len(var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 1/7 statements.
# Partially parsed test_config_init_sets_src_paths. Retrieved 1/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()

def test_case_0():
    var_0 = 'test_project'

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

def test_case_0():
    var_0 = 'test_project'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_is_supported_filetype_with_supported_extension. Retrieved 3/4 statements.
# Partially parsed test_config_known_patterns_property. Retrieved 3/6 statements.
# Partially parsed test_config_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_section_comments_end_property. Retrieved 2/3 statements.
# Partially parsed test_config_skips_property. Retrieved 2/3 statements.
# Partially parsed test_config_skip_globs_property. Retrieved 2/3 statements.
# Partially parsed test_config_parse_known_pattern. Retrieved 2/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyc'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.pyc'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.known_patterns
    var_2 = 2

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skips

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_pattern'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = 3
    var_3 = module_0.Config()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_line_43_evaluates_to_false. Retrieved 7/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'some_setting'
    var_1 = 'value'
    var_2 = '/test'
    var_3 = {}
    var_4 = '/test/dir/setup.cfg'
    var_5 = False
    var_6 = module_0.Config(var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test.ini'
    var_1 = '[section1]\nkey=value\n'
    var_2 = {}
    var_3 = '.editorconfig'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/10 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_indent_space. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_tab. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_bool_value. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_value. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_frozenset_value. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_force_grid_wrap_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_multiple_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_with_section_header. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_extension_pattern. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_multiline_list_value. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\nline_length = 88\n"
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\nline_length = 88\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = space\nindent_size = 2\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nmax_line_length = 100\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nmax_line_length = off\n'
    var_2 = '*'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_alphabetical_sort = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_django = django,rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = __init__.py,migrations\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 2\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = "[isort]\ncomment_prefix = '# '\n"
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\n[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = 'other'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool]\n[tool.isort]\nprofile = 'black'\n"
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = space\nindent_size = 4\n[*.py]\nindent_size = 2\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyi}]\nindent_size = 2\nindent_style = space\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_first_party = myproject\n    submodule\n    another\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_first_party'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_config_init_with_config_object. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_setting'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #28
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'docs/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = '/home/user/docs/'
    var_5 = {var_4}

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/etc/config'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = {var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'docs'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = {var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'docs/'
    var_2 = '/etc/config'
    var_3 = 'file.txt'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = '/home/user/docs/'
    var_7 = {var_6, var_2, var_3}

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/etc/config/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = {var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'path/to/docs/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = '/home/user/path/to/docs/'
    var_5 = {var_4}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_144_evaluates_to_true. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'sections'
    var_1 = 'FUTURE'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'FIRSTPARTY'
    var_5 = 'LOCALFOLDER'
    var_6 = (var_1, var_2, var_3, var_4, var_5)
    var_7 = {var_0: var_6}
    var_8 = ()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 13/19 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_setting'
    var_8 = 'py38'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_editorconfig. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_editorconfig_tab_indent. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/14 statements.
# Partially parsed test_get_config_data_boolean_conversion. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/15 statements.
# Partially parsed test_get_config_data_force_grid_wrap_numeric. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_force_grid_wrap_legacy_false. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_force_grid_wrap_legacy_true. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/12 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\nskip_gitignore = true\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 100\nprofile = black\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 1\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip_gitignore = true\nforce_alphabetical_sort = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_django = django\nknown_rest_framework = rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 2\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = "[isort]\ncomment_prefix = '# '\n"
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool]\n[tool.isort]\nline_length = 79\n'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_toml_file_path_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '.toml'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_config_init_sets_directory_from_current_working_directory. Retrieved 1/3 statements.
# Partially parsed test_config_init_with_src_paths_override. Retrieved 2/7 statements.
# Partially parsed test_config_init_creates_sources_tuple. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End stdlib imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

def test_case_0():
    var_0 = 'src'
    var_1 = 'lib'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.sources



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/6 statements.
# Partially parsed test_config_init_with_settings_path. Retrieved 1/5 statements.
# Failed to parse test_config_init_with_src_paths.
# Partially parsed test_config_init_with_multiple_overrides. Retrieved 3/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'

def test_case_0():
    var_0 = 'project'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_config_constructor_sets_git_ls_files. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_initializes_src_paths. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '_known_patterns'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_section_comments'
    var_4 = hasattr(var_0, var_3)
    var_5 = '_section_comments_end'
    var_6 = hasattr(var_0, var_5)
    var_7 = '_skips'
    var_8 = hasattr(var_0, var_7)
    var_9 = '_skip_globs'
    var_10 = hasattr(var_0, var_9)
    var_11 = '_sorting_function'
    var_12 = hasattr(var_0, var_11)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'git_ls_files'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.git_ls_files

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = "'    '"
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'directory'
    var_2 = hasattr(var_0, var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src_paths'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.src_paths



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_config_init_with_config_parameter.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_config_init_known_patterns_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_end_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_skips_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_skip_globs_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_directory_set_from_cwd. Retrieved 2/3 statements.
# Partially parsed test_config_init_src_paths_set. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '"  "'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.known_patterns

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skips

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = module_0.Config()
    var_2 = var_1.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = 3
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.directory

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_find_all_configs. Retrieved 8/32 statements.
# Partially parsed test_find_all_configs_empty_directory. Retrieved 2/12 statements.
# Partially parsed test_find_all_configs_nested_structure. Retrieved 6/21 statements.
# Partially parsed test_find_all_configs_with_pyproject_toml. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'subdir1'
    var_2 = 'subdir2'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=88\n'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nline_length=100\n'
    var_7 = 'test_file.py'

def test_case_0():
    var_0 = 'empty'
    var_1 = 'test_file.py'

def test_case_0():
    var_0 = 'root'
    var_1 = 'level1'
    var_2 = 'level2'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=80\n'
    var_5 = '[settings]\nline_length=120\n'

def test_case_0():
    var_0 = 'toml_test'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.isort]\nline_length = 100\nprofile = "black"\n'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_line_159_predicate_evaluates_to_true. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = '/path/to/config/file.cfg'
    var_2 = {var_0: var_1}
    var_3 = None



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = 'deprecated_option_1'
    var_1 = 'deprecated_option_2'
    var_2 = {var_0, var_1}
    var_3 = 'valid_option'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = [option for option in var_7 if option in var_2]
    var_9 = len(var_8)
    assert var_9 == 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_line_98_evaluates_to_true. Retrieved 27/36 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 98 evaluates to True for a custom known section.'
    var_1 = 'known_'
    var_2 = 'known_custom_section'
    var_3 = 'known_standard_library'
    var_4 = 'module1'
    var_5 = 'module2'
    var_6 = [var_4, var_5]
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = [var_7, var_8]
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = 'known_custom_section'
    var_12 = 'known_future_library'
    var_13 = 'known_third_party'
    var_14 = 'known_first_party'
    var_15 = 'known_local_folder'
    var_16 = (var_3, var_12, var_13, var_14, var_15)
    var_17 = var_11 not in var_16
    var_18 = 'known_standard_library'
    var_19 = (var_3, var_12, var_13, var_14, var_15)
    var_20 = var_18 not in var_19
    var_21 = 'some_other_config'
    var_22 = (var_3, var_12, var_13, var_14, var_15)
    var_23 = var_21 not in var_22
    var_24 = 'known_my_custom_libs'
    var_25 = (var_3, var_12, var_13, var_14, var_15)
    var_26 = var_24 not in var_25



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 1/6 statements.
# Failed to parse test_config_init_sets_src_paths.
# Partially parsed test_config_init_with_custom_src_paths. Retrieved 1/8 statements.
# Partially parsed test_config_init_with_known_prefix. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

def test_case_0():
    var_0 = 'test_config'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()

def test_case_0():
    var_0 = 0

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = frozenset()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End stdlib'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile_xyz'
    var_1 = module_0.Config()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 16/34 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_other_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = []
    var_13 = '__init__'
    var_14 = [var_13]
    var_15 = None



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = False



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 'old_option'
    var_1 = 'legacy_setting'
    var_2 = {var_0, var_1}
    var_3 = 'normal_option'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = [option for option in var_7 if option in var_2]
    var_9 = len(var_8)
    assert var_9 == 2



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_is_skipped_normalized_path_predicate_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '/home/user/test.py'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_config_settings_predicate_at_line_76. Retrieved 19/33 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = '/path/to/settings.cfg'
    var_3 = module_0.Config(var_2)
    var_4 = 'line_length'
    var_5 = 'profile'
    var_6 = 100
    var_7 = 'black'
    var_8 = '/path/to/.isort.cfg'
    var_9 = module_0.Config(var_8)
    var_10 = 'line_length'
    var_11 = 'multi_line_mode'
    var_12 = 'skip'
    var_13 = 120
    var_14 = 2
    var_15 = 'migrations'
    var_16 = [var_15]
    var_17 = '/path/to/pyproject.toml'
    var_18 = module_0.Config(var_17)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_config_predicate_line_7_evaluates_to_false. Retrieved 2/21 statements.


def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #49
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is True
    var_2 = 'True'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is True
    var_4 = 'TRUE'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is True
    var_6 = 'yes'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is True
    var_8 = 'Yes'
    var_9 = module_0._as_bool(var_8)
    assert var_9 is True
    var_10 = 'y'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is True
    var_12 = '1'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is True
    var_14 = 'on'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is False
    var_2 = 'False'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is False
    var_4 = 'FALSE'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is False
    var_6 = 'no'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is False
    var_8 = 'No'
    var_9 = module_0._as_bool(var_8)
    assert var_9 is False
    var_10 = 'n'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is False
    var_12 = '0'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is False
    var_14 = 'off'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._as_bool(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_bool(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'maybe'
    var_1 = module_0._as_bool(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'TrUe'
    var_1 = module_0._as_bool(var_0)
    assert var_1 is True
    var_2 = 'FaLsE'
    var_3 = module_0._as_bool(var_2)
    assert var_3 is False
    var_4 = 'YeS'
    var_5 = module_0._as_bool(var_4)
    assert var_5 is True
    var_6 = 'nO'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_is_skipped_predicate_line_3_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '/some/file.py'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 4/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'File not found'
    var_2 = 'test_file.py'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_get_config_data_toml_file_predicate. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '[tool]\nkey = "value"\n'
    var_1 = '.toml'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/6 statements.
# Partially parsed test_config_init_known_other_section. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = module_0.Config()

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = "'  '"
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.src_paths
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'lib'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = frozenset()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of stdlib'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 120
    var_2 = 3
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'migrations'
    var_1 = [var_0]
    var_2 = '*.pyi'
    var_3 = [var_2]
    var_4 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = module_0.Config(config=var_1)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_indent_in_combined_config_evaluates_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'indent'
    var_1 = 4
    var_2 = {var_0: var_1}



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_import_footers_predicate_evaluates_to_true. Retrieved 16/24 statements.


def test_case_0():
    var_0 = 'import_footer_mylib'
    var_1 = 'import_footer_other'
    var_2 = 'Footer for mylib'
    var_3 = 'Footer for other'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'import_footer_mylib'
    var_6 = 'import_footer_other'
    var_7 = 'directory'
    var_8 = 'Footer for mylib'
    var_9 = 'Footer for other'
    var_10 = '/test/dir'
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = 'mylib'
    var_13 = 'other'
    var_14 = {var_12: var_8, var_13: var_9}
    var_15 = len(var_14)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_import_footer_prefix_predicate. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'import_footer_'
    var_1 = 'import_footer_future'
    var_2 = '# Future imports'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_in_stat. Retrieved 3/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_config_predicate_line_6_evaluates_to_false.




# Parsed testcases at query #59
#--------------------------

# Partially parsed test_get_str_to_type_converter_returns_int_type_for_int_setting. Retrieved 2/4 statements.
# Partially parsed test_get_str_to_type_converter_returns_bool_type_for_bool_setting. Retrieved 2/4 statements.
# Partially parsed test_get_str_to_type_converter_returns_wrap_mode_converter_for_wrap_modes. Retrieved 2/4 statements.
# Partially parsed test_get_str_to_type_converter_returns_float_type_for_float_setting. Retrieved 2/4 statements.
# Partially parsed test_get_str_to_type_converter_returns_list_type_for_list_setting. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'unknown_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_int_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_bool_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap_mode_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_float_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'test_list_setting'
    var_4 = module_0._get_str_to_type_converter(var_3)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_at_line_159_evaluates_to_true. Retrieved 6/12 statements.


import posixpath as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = '/path/to/config/file.cfg'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = var_2[var_0]
    var_5 = module_0.dirname(var_4)



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_config_init_with_config_parameter.




# Parsed testcases at query #62
#--------------------------

# Partially parsed test_import_headings_predicate_evaluates_to_true. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'import_heading_future'
    var_1 = 'import_heading_stdlib'
    var_2 = 'Future imports'
    var_3 = 'Standard library imports'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'future'
    var_6 = 'stdlib'
    var_7 = 'Future imports'
    var_8 = 'Standard library imports'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_src_paths_not_in_combined_config. Retrieved 6/21 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'directory'
    var_2 = '/test/dir'
    var_3 = {var_1: var_2}
    var_4 = 'src_paths'
    var_5 = var_4 not in var_3
    assert var_5 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_field'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_setting'
    var_8 = 'py311'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



