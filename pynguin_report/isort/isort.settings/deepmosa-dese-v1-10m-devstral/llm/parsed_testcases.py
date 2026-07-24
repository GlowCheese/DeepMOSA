####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_known_prefix_condition. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'known_custom_section'



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.exe'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/dev/zero'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_script'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'txt'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.txt'
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
    var_1 = '/dev/null'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.sh'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_init_with_src_paths. Retrieved 5/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'known_foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = [var_1]
    var_7 = frozenset(var_6)
    var_8 = {var_5: var_7}

import isort.settings as module_0

def test_case_0():
    var_0 = 'import_heading_foo'
    var_1 = 'Bar'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import_footer_foo'
    var_1 = 'Bar'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip'
    var_1 = 'extend_skip'
    var_2 = 'foo'
    var_3 = [var_2]
    var_4 = 'bar'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2, var_4]
    var_9 = frozenset(var_8)

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = 'extend_skip_glob'
    var_2 = 'foo'
    var_3 = [var_2]
    var_4 = 'bar'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2, var_4]
    var_9 = frozenset(var_8)

import isort.settings as module_0

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'custom'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'txt'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'file.txt'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'file.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/dev/stdin'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'script'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__get_config_data_toml_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_editorconfig_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_other_config_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_empty_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_editorconfig_max_line_length_digit. Retrieved 6/7 statements.
# Partially parsed test__get_config_data_editorconfig_filter_keys. Retrieved 4/6 statements.
# Partially parsed test__get_config_data_tuple_conversion. Retrieved 4/7 statements.
# Partially parsed test__get_config_data_frozenset_conversion. Retrieved 4/7 statements.
# Partially parsed test__get_config_data_bool_conversion. Retrieved 4/7 statements.
# Partially parsed test__get_config_data_known_prefix_conversion. Retrieved 4/7 statements.
# Partially parsed test__get_config_data_force_grid_wrap_conversion. Retrieved 6/8 statements.
# Partially parsed test__get_config_data_comment_prefix_conversion. Retrieved 6/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'empty.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'inf'
    var_5 = float(var_4)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'line_length'
    var_5 = var_3[var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'force_grid_wrap'
    var_5 = var_3[var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'comment_prefix'
    var_5 = var_3[var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_profile_not_in_profiles. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'profile'
    var_1 = 'non_existent_profile'
    var_2 = {var_0: var_1}
    var_3 = ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'some_key'
    var_1 = 'known_'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_maps_to_section_in_known_section_mapping. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 'known_custom_section'
    var_1 = 'custom_module'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'CUSTOM_SECTION'
    var_5 = 'CUSTOM'
    var_6 = {var_4: var_5}
    var_7 = 'known_'
    var_8 = len(var_7)
    var_9 = var_0[var_8:]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__find_config_found. Retrieved 2/4 statements.
# Partially parsed test__find_config_exception. Retrieved 2/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_file.cfg'
    var_1 = False
    var_2 = module_0.Config(var_0)



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_config_finds_pyproject_toml. Retrieved 1/4 statements.
# Partially parsed test_find_config_finds_setup_cfg. Retrieved 1/4 statements.
# Partially parsed test_find_config_finds_tox_ini. Retrieved 1/4 statements.
# Partially parsed test_find_config_finds_setup_cfg_in_parent_dir. Retrieved 1/5 statements.
# Partially parsed test_find_config_stops_at_git_dir. Retrieved 2/6 statements.
# Partially parsed test_find_config_stops_at_hg_dir. Retrieved 2/6 statements.
# Partially parsed test_find_config_stops_at_svn_dir. Retrieved 2/6 statements.
# Partially parsed test_find_config_returns_correct_config_data. Retrieved 1/4 statements.
# Partially parsed test_find_config_ignores_invalid_config_files. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 1
    var_1 = '/nonexistent/path'
    var_2 = _find_config(var_1)[var_0]

def test_case_0():
    var_0 = 0
    var_1 = '.git'

def test_case_0():
    var_0 = 0
    var_1 = '.hg'

def test_case_0():
    var_0 = 0
    var_1 = '.svn'

def test_case_0():
    var_0 = 0
    var_1 = '/'
    var_2 = _find_config(var_1)[var_0]
    assert var_2 == '/'

def test_case_0():
    var_0 = 'pyproject.toml'

def test_case_0():
    var_0 = 1
    var_1 = 'invalid_config.toml'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'a,b,c'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'a,b\nc,d'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ' a , b , c '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'a,,b,c'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ' a '
    var_1 = ' b '
    var_2 = ' c '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ',,,'
    var_1 = module_0._as_list(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_config_constructor_default. Retrieved 2/9 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/8 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'

def test_case_0():
    var_0 = '[isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length=120\n'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 120
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
    var_0 = 'nonexistent'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = 'rest_framework'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Django imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of Django imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'force_single_line'
    var_3 = hasattr(var_1, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 120
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = module_0.Config(config=var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_post_init_known_standard_library. Retrieved 2/5 statements.
# Failed to parse test_post_init_vertical_grid_grouped_no_comma.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0._Config(force_alphabetical_sort=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_skipped_file. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_returns_false_for_non_skipped_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_returns_true_for_file_matching_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_false_for_file_not_matching_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_true_for_file_in_skipped_directory. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_false_for_file_not_in_skipped_directory. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_true_for_non_existent_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_returns_true_for_editor_backup_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_returns_false_for_non_editor_backup_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_returns_true_for_git_ignored_file. Retrieved 5/10 statements.
# Partially parsed test_is_skipped_returns_false_for_git_tracked_file. Retrieved 4/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '.gitignore'
    var_1 = {var_0}
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'

import isort.settings as module_0

def test_case_0():
    var_0 = '*.tmp'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'temp.tmp'

import isort.settings as module_0

def test_case_0():
    var_0 = '*.tmp'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'test.py'

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = '__pycache__/test.py'

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'src/test.py'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = '/test'
    var_3 = '/test/file.py'
    var_4 = '/test/ignored.py'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = '/test'
    var_3 = '/test/file.py'



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'known_foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = [var_1]
    var_7 = frozenset(var_6)
    var_8 = {var_5: var_7}

import isort.settings as module_0

def test_case_0():
    var_0 = 'import_heading_foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import_footer_foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'line_length'
    var_2 = 100
    var_3 = {var_1: var_2}
    var_4 = module_0.Config(config=var_0, **var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4.src_paths
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4.src_paths[var_7]
    var_9 = str(var_8)

import isort.settings as module_0

def test_case_0():
    var_0 = 'directory'
    var_1 = '.'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'sections'
    var_1 = 'FUTURE'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'FIRSTPARTY'
    var_5 = 'LOCALFOLDER'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.Config(**var_7)

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = '*.py'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Config(**var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_str_to_type_converter_with_int_setting. Retrieved 2/3 statements.
# Partially parsed test_get_str_to_type_converter_with_float_setting. Retrieved 2/3 statements.
# Partially parsed test_get_str_to_type_converter_with_bool_setting. Retrieved 2/3 statements.
# Partially parsed test_get_str_to_type_converter_with_wrap_modes. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'int_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'float_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'bool_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap_mode'
    var_1 = module_0._get_str_to_type_converter(var_0)



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap_mode'
    var_1 = module_0._get_str_to_type_converter(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_123_evaluates_to_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'sections'
    var_1 = ()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__get_config_data_toml_file. Retrieved 6/9 statements.
# Partially parsed test__get_config_data_editorconfig_file. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_ini_file. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_empty_file. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_missing_sections. Retrieved 6/9 statements.
# Partially parsed test__get_config_data_force_grid_wrap. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_comment_prefix. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_abspaths. Retrieved 11/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = "[section1]\nkey1 = 'value1'\nkey2 = 123\n\n[section2]\nkey3 = true\nkey4 = [1, 2, 3]"
    var_5 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88'
    var_4 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nkey1 = value1\nkey2 = 123'
    var_4 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = ''
    var_4 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = '[section3]\nkey1 = value1'
    var_5 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nforce_grid_wrap = false'
    var_4 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = "[section1]\ncomment_prefix = '#'"
    var_4 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nknown_prefix.paths = path1, path2'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = 'known_prefix.paths'
    var_6 = 'source'
    var_7 = module_1.dirname(var_0)
    var_8 = 'path1'
    var_9 = module_1.dirname(var_0)
    var_10 = 'path2'



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = module_0.Config(config=var_0)



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_vertical_grid_grouped_no_comma_conversion.




# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap_mode'
    var_1 = module_0._get_str_to_type_converter(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_exact_skip_path. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_returns_true_for_parent_directory_in_skip. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_true_for_skip_glob_match. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_true_for_skip_glob_match_with_path. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_false_for_non_matching_file. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_returns_true_for_non_existent_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_returns_true_for_gitignore_when_skip_gitignore_enabled. Retrieved 6/10 statements.
# Partially parsed test_is_skipped_returns_false_for_git_tracked_file_when_skip_gitignore_enabled. Retrieved 5/9 statements.
# Partially parsed test_is_skipped_returns_true_for_git_directory. Retrieved 3/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'test_dir/subfile.py'

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'test.pyc'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_*'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'test_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'other_file.py'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'test_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = '/test'
    var_3 = '/test/committed_file.py'
    var_4 = {var_3}
    var_5 = '/test/ignored_file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = '/test'
    var_3 = '/test/committed_file.py'
    var_4 = {var_3}

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = '.git'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_init_with_defaults. Retrieved 2/9 statements.
# Partially parsed test_config_init_with_settings_file. Retrieved 1/6 statements.
# Partially parsed test_config_init_with_settings_path. Retrieved 1/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'

def test_case_0():
    var_0 = '[isort]\nprofile=black\n'

def test_case_0():
    var_0 = 'test'

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = module_0.Config(config=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'foo'
    var_4 = [var_0]
    var_5 = frozenset(var_4)
    var_6 = {var_3: var_5}

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.Config()



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_settings.cfg'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_config_returns_correct_config_when_file_exists. Retrieved 1/7 statements.
# Partially parsed test_find_config_stops_search_on_stop_dir. Retrieved 2/8 statements.
# Partially parsed test_find_config_returns_correct_directory_and_config. Retrieved 1/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._find_config(var_0)

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'stop_dir'
    var_1 = {}

def test_case_0():
    var_0 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_import_heading_prefix_detection. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 'import_heading_prefix_test'
    var_1 = 'some_other_key'
    var_2 = 'test_value'
    var_3 = 'other_value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'import_heading_prefix_test'
    var_6 = 'test_value'
    var_7 = 'import_heading_prefix'
    var_8 = len(var_7)
    var_9 = var_5[var_8:]
    var_10 = str(var_6)



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1, item2, item3'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1\nitem2\nitem3'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1, item2\nitem3'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '  item1  ,  item2  '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1, , item2'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1'
    var_1 = ' item2 '
    var_2 = 'item3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1'
    var_1 = ''
    var_2 = 'item2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_path_root_is_dir. Retrieved 3/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/existing/directory'
    var_1 = module_0.Config(settings_path=var_0)
    var_2 = False



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'test.cfg'
    var_1 = False
    var_2 = module_0.Config(var_0)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_skipped_predicate_false. Retrieved 4/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/some/path'
    var_2 = var_0.directory
    var_3 = var_0.directory



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path1/'
    var_2 = '/absolute/path2'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'relative/path1/'
    var_2 = 'relative/path2'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path1/'
    var_2 = 'relative/path2/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path1/'
    var_2 = [var_1, var_1]
    var_3 = module_0._abspaths(var_0, var_2)



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_supported_filetype_supported_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_fifo. Retrieved 5/10 statements.
# Partially parsed test_is_supported_filetype_shebang. Retrieved 4/7 statements.
# Partially parsed test_is_supported_filetype_no_shebang. Retrieved 4/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'py'
    var_2 = 'test.py'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'txt'
    var_2 = 'test.txt'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py~'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/tmp'
    var_2 = True
    var_3 = '/tmp/test_fifo'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = "#!/usr/bin/env python\nprint('hello')"
    var_2 = 'test_shebang.py'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = "print('hello')"
    var_2 = 'test_no_shebang.py'
    var_3 = var_0.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_import_footers_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'import_footer_test'
    var_1 = 'footer_value'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_config_initialization_with_defaults. Retrieved 2/9 statements.
# Partially parsed test_config_initialization_with_settings_file. Retrieved 1/6 statements.
# Partially parsed test_config_initialization_with_settings_path. Retrieved 1/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'

def test_case_0():
    var_0 = '[isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'test'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = module_0.Config()
    var_3 = 120
    var_4 = module_0.Config(config=var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_profile'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()
    var_2 = 'tab'
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = module_0.Config()
    var_2 = 'foo'
    var_3 = [var_0]
    var_4 = frozenset(var_3)
    var_5 = {var_2: var_4}

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()
    var_2 = var_1.formatting_function
    var_3 = callable(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_formatter'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()
    var_2 = 'native'
    var_3 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_sort_order'
    var_1 = module_0.Config()



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_0.Config(config=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.Config()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_all_configs_empty_path. Retrieved 2/3 statements.
# Failed to parse test_find_all_configs_no_config_files.
# Partially parsed test_find_all_configs_with_valid_config. Retrieved 5/15 statements.
# Partially parsed test_find_all_configs_with_invalid_config. Retrieved 2/8 statements.
# Partially parsed test_find_all_configs_nested_configs. Retrieved 9/29 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.find_all_configs(var_0)

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88\n'
    var_2 = 'line_length'
    var_3 = 1
    var_4 = -1

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = 'invalid config content'

def test_case_0():
    var_0 = 'subdir'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=88\n'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.isort]\nline_length=120\n'
    var_5 = 'line_length'
    var_6 = 1
    var_7 = -1
    var_8 = -1



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_constructor_with_no_arguments. Retrieved 2/9 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/8 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'src'

def test_case_0():
    var_0 = '[isort]\nline_length = 100\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 120\n'

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = module_0.Config()
    var_2 = 100
    var_3 = module_0.Config(config=var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
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
    var_0 = 'bar'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'foo'
    var_4 = [var_0]
    var_5 = frozenset(var_4)
    var_6 = {var_3: var_5}

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar Imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of Bar Imports'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.Config()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_editorconfig_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_other_config_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_editorconfig_filtered_keys. Retrieved 4/6 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 4/7 statements.
# Partially parsed test_get_config_data_frozenset_conversion. Retrieved 4/7 statements.
# Partially parsed test_get_config_data_bool_conversion. Retrieved 4/7 statements.
# Partially parsed test_get_config_data_known_prefix_conversion. Retrieved 4/7 statements.
# Partially parsed test_get_config_data_force_grid_wrap_conversion. Retrieved 6/8 statements.
# Partially parsed test_get_config_data_comment_prefix_conversion. Retrieved 6/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'empty.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'inf'
    var_5 = float(var_4)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'force_grid_wrap'
    var_5 = var_3[var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'comment_prefix'
    var_5 = var_3[var_4]



# Parsed testcases at query #21
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_with_shebang. Retrieved 4/8 statements.
# Partially parsed test_is_supported_filetype_without_shebang. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'txt'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.txt'
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
    var_1 = 'test_fifo'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'nonexistent_file.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_shebang'
    var_2 = '#!/usr/bin/env python\n'
    var_3 = var_0.is_supported_filetype(var_1)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_no_shebang'
    var_2 = "print('hello')\n"
    var_3 = var_0.is_supported_filetype(var_1)
    assert var_3 is False



# Parsed testcases at query #23
#--------------------------

# Failed to parse test___post_init___vertical_grid_grouped_no_comma_converted.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.known_standard_library
    var_2 = len(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0._Config(force_alphabetical_sort=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__get_config_data_with_toml_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_with_editorconfig_file. Retrieved 4/5 statements.
# Partially parsed test__get_config_data_with_ini_file. Retrieved 4/5 statements.
# Partially parsed test__get_config_data_with_empty_sections. Retrieved 3/4 statements.
# Partially parsed test__get_config_data_with_nonexistent_file. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = ()
    var_2 = module_0._get_config_data(var_0, var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_file.toml'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_file.cfg'
    var_1 = True
    var_2 = module_0.Config(var_0)



