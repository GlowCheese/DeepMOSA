####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_init_with_settings_path_invalid. Retrieved 2/7 statements.
# Partially parsed test_config_init_with_profile_non_existent. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = module_0._Config(var_0)
    var_2 = 'py_version'
    var_3 = '310'
    var_4 = {var_2: var_3}
    var_5 = 'py_version'
    var_6 = {var_5: var_3}
    var_7 = module_0.Config(config=var_1, **var_6)
    var_8 = var_7.py_version
    assert var_8 == '310'

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_file.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '3.x'

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/non_existent_path_isort_test'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(var_0)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile_isort_test'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_0)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_init_profile_not_in_profiles_triggers_entry_points. Retrieved 5/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'some'
    var_1 = 'config'
    var_2 = 'test_profile'
    var_3 = 'profile'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_profile'
    var_7 = 'isort.profiles'



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._settings_file
    assert var_2 == ''
    var_3 = var_1._settings_path
    assert var_3 == ''
    var_4 = var_1._config
    assert var_4 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_init_skips_warning_when_section_is_in_sections. Retrieved 6/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'std'
    var_2 = 'third_party'
    var_3 = 'my_section'
    var_4 = (var_1, var_2, var_3)
    var_5 = 'known_my_section'
    var_6 = 'sections'
    var_7 = {var_5: var_0, var_6: var_4}
    var_8 = module_0.Config(**var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_constructor_with_overrides_and_config_object. Retrieved 17/23 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py310'
    var_8 = []
    var_9 = ()
    var_10 = ()
    var_11 = frozenset()
    var_12 = frozenset()
    var_13 = None
    var_14 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = 4
    var_16 = 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = "'2'"
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'my_group'
    var_1 = 'known_import_group'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'my_group'
    var_5 = bool('my_group' in var_3.known_other_logic_check_placeholder)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile_xyz_123'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'ProfileDoesNotExist not raised'
    var_6 = AssertionError(var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_formatter_xyz_123'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'FormattingPluginDoesNotExist not raised'
    var_6 = AssertionError(var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_sort_order_xyz_123'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'SortingFunctionDoesNotExist not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_config_post_init_multi_line_output_normalization.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.py_version
    assert var_1 == 'py3'
    var_2 = var_0.line_length
    assert var_2 == 79
    var_3 = var_0.wrap_length
    assert var_3 == 0

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py3.9'

import isort.settings as module_0

def test_case_0():
    var_0 = '99.9'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 60
    var_2 = module_0._Config(line_length=var_0, wrap_length=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0._Config(force_alphabetical_sort=var_0)
    var_2 = var_1.force_alphabetical_sort_within_sections
    assert var_2 is True
    var_3 = var_1.no_sections
    assert var_3 is True
    var_4 = var_1.lines_between_types
    assert var_4 == 1
    var_5 = var_1.from_first
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_constructor_with_existing_config_object. Retrieved 3/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = 4
    var_2 = True
    var_3 = 'py_version'
    var_4 = 'indent'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

def test_case_0():
    var_0 = 'py_version'
    var_1 = [var_0]
    var_2 = '  '

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = 'InvalidSettingsPath not raised'
    var_4 = AssertionError(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = "'  '"
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'unsupported_option'
    var_5 = 'UnsupportedSettings not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_all_configs_returns_trie_with_inserted_data. Retrieved 3/24 statements.
# Failed to parse test_find_all_configs_empty_directory_returns_default_trie.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.isort]\nprofile = "black"\n'



# Parsed testcases at query #10
#--------------------------




import posixpath as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/tmp/test'
    var_2 = 'subdir/file.txt'
    var_3 = 'folder/'
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_3]
    var_6 = module_0.join(var_0, *var_5)
    var_7 = {var_1, var_2, var_6}
    var_8 = module_1._abspaths(var_0, var_4)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = set()
    var_3 = module_0._abspaths(var_0, var_1)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/a'
    var_2 = '/b/c'
    var_3 = [var_1, var_2]
    var_4 = {var_1, var_2}
    var_5 = module_0._abspaths(var_0, var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'file.txt'
    var_2 = 'dir/file.txt'
    var_3 = [var_1, var_2]
    var_4 = {var_1, var_2}
    var_5 = module_0._abspaths(var_0, var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import posixpath as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir/'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = module_0.join(var_0, *var_3)
    var_5 = {var_4}
    var_6 = module_1._abspaths(var_0, var_2)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_explicit_skip_path. Retrieved 3/10 statements.
# Partially parsed test_is_skipped_returns_true_for_skip_glob_match. Retrieved 4/11 statements.
# Partially parsed test_is_skipped_returns_true_for_parent_directory_in_skips. Retrieved 4/11 statements.
# Partially parsed test_is_skipped_returns_false_for_valid_file_not_in_skips. Retrieved 4/12 statements.
# Partially parsed test_is_skipped_returns_true_for_non_existent_path. Retrieved 2/10 statements.
# Partially parsed test_is_skipped_returns_true_for_git_ignored_file_when_skip_gitignore_is_true. Retrieved 5/13 statements.
# Partially parsed test_is_skipped_returns_true_for_git_dot_folder_when_skip_gitignore_is_true. Retrieved 2/10 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/skip'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)

import pathlib as module_0

def test_case_0():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = '/path/to/project/test_file.tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)

import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/project/ignored_folder'
    var_1 = [var_0]
    var_2 = '/path/to/project/ignored_folder/sub/file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)

import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/project/ignored'
    var_1 = [var_0]
    assert var_1 is False
    var_2 = '/path/to/project/valid_file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)

import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/project/ghost_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)

import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/project'
    assert var_0 is True
    var_1 = '/path/to/project/tracked.py'
    var_2 = {var_1}
    var_3 = '/path/to/project/ignored_by_git.py'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)

import pathlib as module_0

def test_case_0():
    var_0 = '/path/to/project/.git/config'
    assert var_0 is True
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_init_with_overrides. Retrieved 2/14 statements.
# Partially parsed test_config_init_with_settings_file_not_found_warning. Retrieved 3/7 statements.
# Partially parsed test_config_init_with_profile_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = '39'

import isort.settings as module_0

def test_case_0():
    var_0 = 'settings.ini'
    var_1 = False
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'settings.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.indent
    assert var_3 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'settings.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.indent
    assert var_3 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'settings.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'ProfileDoesNotExist'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_skipped_directory_not_in_parents_evaluates_to_false. Retrieved 2/15 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/other_dir/file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_formatter_exists. Retrieved 3/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = {}
    var_3 = 'black'
    var_4 = 'formatter'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._settings_file
    assert var_2 == ''
    var_3 = var_1._settings_path
    assert var_3 == ''
    var_4 = var_1._config
    assert var_4 is None



