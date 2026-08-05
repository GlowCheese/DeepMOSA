####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_constructor_with_overrides. Retrieved 22/38 statements.
# Partially parsed test_config_constructor_indent_parsing. Retrieved 22/30 statements.
# Partially parsed test_config_constructor_known_prefix_logic. Retrieved 22/26 statements.
# Partially parsed test_config_constructor_unsupported_settings_raises. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_defaults'
    var_4 = 'some_setting'
    var_5 = 'py39'
    var_6 = []
    var_7 = ()
    var_8 = {}
    var_9 = 'value'
    var_10 = 'indent'
    var_11 = 'quiet'
    var_12 = 4
    var_13 = True
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'py'
    var_16 = ''
    var_17 = '_section_comments_end'
    var_18 = None
    var_19 = '_skips'
    var_20 = '_skip_globs'
    var_21 = '_sorting_function'

def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = str(var_3)
    var_5 = ' '
    var_6 = int(var_4)
    var_7 = var_5 * var_6
    assert var_7 == '    '
    var_8 = 'tab'
    var_9 = {var_5: var_8}
    var_10 = var_9[var_5]
    var_11 = str(var_10)
    var_12 = "'"
    var_13 = '"'
    var_14 = '\t'
    assert var_14 == '\t'
    var_15 = "'2'"
    var_16 = {var_5: var_15}
    var_17 = var_16[var_5]
    var_18 = str(var_17)
    var_19 = ' '
    var_20 = int(var_18)
    var_21 = var_19 * var_20
    assert var_21 == '  '

def test_case_0():
    var_0 = 'known_'
    var_1 = 'std'
    var_2 = 'standard_library'
    var_3 = {var_1: var_2}
    var_4 = 'known_std'
    var_5 = 'sections'
    var_6 = 'import_headings'
    var_7 = 'os'
    var_8 = 'sys'
    var_9 = [var_7, var_8]
    var_10 = 'standard'
    var_11 = 'other'
    var_12 = (var_10, var_11)
    var_13 = 'test'
    var_14 = 'test_header'
    var_15 = {var_13: var_14}
    var_16 = {var_4: var_9, var_5: var_12, var_6: var_15}
    var_17 = 'known_std'
    var_18 = [var_7, var_8]
    var_19 = len(var_0)
    var_20 = var_17[var_19:]
    var_21 = f'known_{var_0.lower()}'
    assert var_21 == 'known_standard_library'

def test_case_0():
    var_0 = 'unsupported_key'
    var_1 = 'known_standard_library'
    var_2 = 'value'
    var_3 = []
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'source'
    var_6 = 'runtime'
    var_7 = {var_5: var_6, var_0: var_2}
    var_8 = [var_7]
    var_9 = {}
    var_10 = 'value'
    var_11 = 'source'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_with_overrides. Retrieved 17/25 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_key'
    var_8 = 'py39'
    var_9 = []
    var_10 = ()
    var_11 = ()
    var_12 = frozenset()
    var_13 = frozenset()
    var_14 = None
    var_15 = 'other_value'
    var_16 = 'extra_val'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'indent'
    var_2 = 'black'
    var_3 = 4
    var_4 = '/tmp/test.ini'
    var_5 = module_0.Config(var_4)

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.Config(settings_path=var_0)
    var_2 = 'InvalidSettingsPath not raised'
    var_3 = AssertionError(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = module_0.Config()
    var_2 = 'tab'
    var_3 = module_0.Config()
    var_4 = "'2'"
    var_5 = module_0.Config()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_initialization_with_custom_section_triggering_predicate. Retrieved 2/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'some_value'
    var_1 = module_0.Config()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_path_root_is_not_directory_logic. Retrieved 4/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/test_file.txt'
    var_1 = module_0.Config(settings_path=var_0)

def test_case_0():
    var_0 = '/fake/file.txt'
    var_1 = '/tmp/file.txt'
    var_2 = False
    var_3 = '/tmp'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_init_from_existing_config. Retrieved 1/8 statements.
# Partially parsed test_config_init_with_settings_file_not_found. Retrieved 3/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = 4
    var_2 = 'black'
    var_3 = module_0.Config()

def test_case_0():
    var_0 = 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_config.ini'
    var_1 = False
    var_2 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile'
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
    var_0 = 1
    var_1 = module_0.Config()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_init_settings_file_not_empty. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = '/tmp'
    var_3 = {}
    var_4 = ''
    var_5 = module_0.Config(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_data_toml_success. Retrieved 16/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = b'section1 = { key1 = "value1", key2 = 10 }\n[section2]\nkey3 = "value3"'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 10
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'key3'
    var_9 = 'value3'
    var_10 = {var_8: var_9}
    var_11 = 'config.toml'
    var_12 = 'section1'
    var_13 = 'section2'
    var_14 = (var_12, var_13)
    var_15 = module_0._get_config_data(var_11, var_14)

import isort.settings as module_0

def test_case_0():
    var_0 = '[section1]\nkey1 = value1\nkey2 = value2'
    var_1 = 'config.ini'
    var_2 = 'section1'
    var_3 = (var_2,)
    var_4 = module_0._get_config_data(var_1, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent_style = space\nindent_size = 2\nmax_line_length = 80'
    var_1 = '.editorconfig'
    var_2 = 'default'
    var_3 = (var_2,)
    var_4 = module_0._get_config_data(var_1, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'empty.toml'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = '[section]\nkey_bool = true'
    var_1 = 'config.ini'
    var_2 = 'section'
    var_3 = (var_2,)
    var_4 = module_0._get_config_data(var_1, var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = '[*.{py,python}]\nkey = value'
    var_1 = 'config.ini'
    var_2 = '*.{py,python}'
    var_3 = (var_2,)
    var_4 = module_0._get_config_data(var_1, var_3)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'indent'
    var_2 = 'black'
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test.ini'
    var_6 = module_0.Config(var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_supported_filetype_returns_false_for_fifo_files. Retrieved 3/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.txt'
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
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'script.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'unknown.unknown'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_constructor_with_existing_config_object. Retrieved 13/21 statements.
# Partially parsed test_config_constructor_attribute_removal. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'indent'
    var_1 = 'line_length'
    var_2 = 4
    var_3 = 88
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    pass

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'line_length'
    var_2 = '_known_patterns'
    var_3 = '_section_comments'
    var_4 = '_section_comments_end'
    var_5 = '_skips'
    var_6 = '_skip_globs'
    var_7 = '_sorting_function'
    var_8 = 'py39'
    var_9 = 88
    var_10 = []
    var_11 = None
    var_12 = 100

def test_case_0():
    var_0 = 'line_length'
    var_1 = 'source'
    var_2 = 88
    var_3 = 'test_source'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_config_init_with_overrides. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py3erm'
    var_8 = []
    var_9 = None
    var_10 = 'value'

def test_case_0():
    pass

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = module_0.Config()



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap_modes'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = module_0._get_str_to_type_converter(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_config_data_toml_parsing. Retrieved 8/16 statements.
# Partially parsed test_get_config_data_ini_parsing. Retrieved 6/15 statements.
# Partially parsed test_get_config_data_editorconfig_parsing. Retrieved 8/16 statements.
# Partially parsed test_get_config_data_empty_sections. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_wildcard_section_ini. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = b'[section]\nkey = "value"\nnum = 10'
    var_2 = 'key'
    var_3 = 'num'
    var_4 = ''
    var_5 = 0
    var_6 = 'section'
    var_7 = (var_6,)

def test_case_0():
    var_0 = 'config.ini'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = ''
    var_4 = 'section'
    var_5 = (var_4,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = 'root\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 80'
    var_2 = 'indent_style'
    var_3 = 'indent_size'
    var_4 = 'max_line_length'
    var_5 = ''
    var_6 = '*'
    var_7 = (var_6,)

def test_case_0():
    var_0 = 'empty.toml'
    var_1 = b'[]'
    var_2 = 'nonexistent'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'wildcard.ini'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = ''
    var_4 = '*.{py}'
    var_5 = (var_4,)



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'sections'
    var_1 = 'test_key'
    var_2 = 'custom_section'
    var_3 = (var_2,)
    var_4 = 456
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'File not found'
    var_1 = 'test.py'



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'apple, banana, cherry'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'apple\nbanana\ncherry'
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '  apple , \n banana,cherry  '
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = ' apple '
    var_1 = 'banana\n'
    var_2 = '  cherry  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = ' '
    var_2 = ''
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_skips_exact_path. Retrieved 8/54 statements.
# Partially parsed test_is_skipped_with_directory_context. Retrieved 5/34 statements.
# Partially parsed test_is_skipped_normalized_windows_paths. Retrieved 3/23 statements.


def test_case_0():
    var_0 = '/tmp/skip_me'
    var_1 = 'ignored_dir'
    var_2 = [var_0, var_1]
    var_3 = '*.tmp'
    var_4 = [var_3]
    var_5 = '/tmp/ignored_dir/file.txt'
    var_6 = 'test.tmp'
    var_7 = '/tmp/keep_me.txt'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'file.py'
    var_4 = [var_2]

def test_case_0():
    var_0 = 'C:/skip/this'
    var_1 = [var_0]
    var_2 = 'C:\\skip\\this'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_handles_known_prefix. Retrieved 5/8 statements.


def test_case_0():
    pass

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()

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
    var_0 = 'non_existent_profile_12345'
    var_1 = module_0.Config()
    var_2 = 'ProfileDoesNotExist should have been raised'
    var_3 = AssertionError(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'py3'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_lib'
    var_1 = module_0.Config()
    var_2 = 'known_custom_library'
    var_3 = hasattr(var_1, var_2)
    var_4 = var_1.sections

import isort.settings as module_0

def test_case_0():
    var_0 = 'some_value'
    var_1 = module_0.Config()



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'import_heading_test'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'import_heading_abc'
    var_4 = 'def'
    var_5 = {var_3: var_4}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_supported_filetype_returns_false_for_fifo_file. Retrieved 3/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.txt'
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
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_init_signature. Retrieved 8/19 statements.


import isort.settings as module_0
import inspect as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'settings.ini'
    var_2 = 'some/path'
    var_3 = None
    var_4 = True
    var_5 = 'val'
    var_6 = module_0.Config(var_1, var_2, var_3)
    var_7 = module_1.signature(var_1)



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'test_profile'
    var_1 = module_0.Config()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_data_toml_success. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_sections. Retrieved 5/10 statements.
# Partially parsed test_get_config_data_editorconfig_logic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_empty_sections. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_wildcard_extension. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '["section1"]\nkey1 = "value1"\nkey2 = 42'
    var_2 = 'section1'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'config.ini'
    var_1 = '[section1]\nkey1 = value1\n[section2]\nkey2 = value2'
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 80'
    var_2 = '*.{py}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'empty.ini'
    var_1 = '[section1]\nkey=val'
    var_2 = 'non_existent'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'config.ini'
    var_1 = '[*.{py,js}]\nkey1 = value1'
    var_2 = '*.{py}'
    var_3 = (var_2,)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_init_settings_file_empty_config_triggers_warning. Retrieved 3/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = False
    var_2 = module_0.Config(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_config_constructor_from_existing_config. Retrieved 1/7 statements.
# Partially parsed test_config_constructor_with_settings_file_not_found_warning. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = 4
    var_2 = True
    var_3 = module_0.Config()

def test_case_0():
    var_0 = '4'

import isort.settings as module_0

def test_case_0():
    var_0 = 'dummy.ini'
    var_1 = module_0.Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.Config(settings_path=var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile'
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = module_0.Config()

import isort.settings as module_0

def test_case_0():
    var_0 = 'some_value'
    var_1 = module_0.Config()



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_config_post_init_multi_line_output_transformation.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()

import isort.settings as module_0

def test_case_0():
    var_0 = '99'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '38'
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

import isort.settings as module_0

def test_case_0():
    var_0 = '39'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)



