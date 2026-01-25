####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_post_init_py_version_auto. Retrieved 3/4 statements.
# Failed to parse test_post_init_multi_line_output_vertical_grid_grouped_no_comma.


import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py3.8'

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'

import isort.settings as module_0

def test_case_0():
    var_0 = '2.7'
    var_1 = module_0._Config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'not supported'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_module'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '3.9'
    var_4 = module_0._Config(var_3, known_standard_library=var_2)
    var_5 = var_4.known_standard_library
    var_6 = bool(var_4.known_standard_library == var_2)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = True
    var_2 = module_0._Config(var_0, force_alphabetical_sort=var_1)
    var_3 = var_2.force_alphabetical_sort_within_sections
    assert var_3 is True
    var_4 = var_2.no_sections
    assert var_4 is True
    var_5 = var_2.lines_between_types
    assert var_5 == 1
    var_6 = var_2.from_first
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = 79
    var_2 = 100
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = 79
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 79
    var_4 = var_2.line_length
    assert var_4 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.line_length
    assert var_2 == 79
    var_3 = var_1.indent
    var_4 = bool(var_1.indent == ' ' * 4)
    assert var_4 is True
    var_5 = var_1.lines_between_sections
    assert var_5 == 1



# Parsed testcases at query #2
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
    var_12 = 'Y'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is True
    var_14 = '1'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is True
    var_16 = 'on'
    var_17 = module_0._as_bool(var_16)
    assert var_17 is True
    var_18 = 'On'
    var_19 = module_0._as_bool(var_18)
    assert var_19 is True

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
    var_12 = 'N'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is False
    var_14 = '0'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is False
    var_16 = 'off'
    var_17 = module_0._as_bool(var_16)
    assert var_17 is False
    var_18 = 'Off'
    var_19 = module_0._as_bool(var_18)
    assert var_19 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value'

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value'

import isort.settings as module_0

def test_case_0():
    var_0 = 'maybe'
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_indent_space. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_editorconfig_indent_tab. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 7/13 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_number. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_ini_with_multiline_values. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_toml_nested_sections. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_wildcard_extension. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_multiple_sections. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\n"
    var_2 = 'tool.isort'
    var_3 = (var_2,)
    var_4 = 'source'

def test_case_0():
    var_0 = 'test.ini'
    var_1 = '[settings]\nline_length = 88\n'
    var_2 = 'settings'
    var_3 = (var_2,)
    var_4 = 'source'

def test_case_0():
    var_0 = 'empty.ini'
    var_1 = ''
    var_2 = 'settings'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'indent'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'indent'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'line_length'
    var_5 = 'inf'
    var_6 = float(var_5)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = 100\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'line_length'

def test_case_0():
    var_0 = 'test.ini'
    var_1 = '[settings]\nknown_first_party = module1,module2\n'
    var_2 = 'settings'
    var_3 = (var_2,)
    var_4 = 'source'

def test_case_0():
    var_0 = 'test.toml'
    var_1 = "[tool]\n[tool.isort]\nprofile = 'black'\nline_length = 88\n"
    var_2 = 'tool.isort'
    var_3 = (var_2,)
    var_4 = 'source'

def test_case_0():
    var_0 = 'test.ini'
    var_1 = '[*.{py,pyi}]\nline_length = 88\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)
    var_4 = 'source'

def test_case_0():
    var_0 = 'test.ini'
    var_1 = '[section1]\nkey1 = value1\n[section2]\nkey2 = value2\n'
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = (var_2, var_3)
    var_5 = 'source'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_init_section_comments_property. Retrieved 5/6 statements.
# Partially parsed test_config_init_skips_property. Retrieved 5/6 statements.
# Partially parsed test_config_init_skip_globs_property. Retrieved 5/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1._known_patterns
    assert var_3 is None
    var_4 = var_1._section_comments
    assert var_4 is None
    var_5 = var_1._section_comments_end
    assert var_5 is None
    var_6 = var_1._skips
    assert var_6 is None
    var_7 = var_1._skip_globs
    assert var_7 is None
    var_8 = var_1._sorting_function
    assert var_8 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3._known_patterns
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = 4
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
    var_0 = '  '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.src_paths
    var_3 = bool(var_1.src_paths is not None)
    assert var_3 is True
    var_4 = var_1.src_paths
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom/path'
    var_1 = [var_0]
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4.src_paths
    var_6 = bool(var_4.src_paths is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    assert var_4 == '.'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.known_patterns
    var_3 = var_1.known_patterns
    var_4 = bool(var_2 is var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'Future imports'
    var_2 = {var_0: var_1}
    var_3 = 'import_headings'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.skips
    var_7 = 'file.py'
    var_8 = bool('file.py' in var_6)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.skip_globs
    var_7 = '*.pyc'
    var_8 = bool('*.pyc' in var_6)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = callable(var_4)
    var_7 = bool(var_6)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = 3
    var_3 = 'quiet'
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = 'include_trailing_comma'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_0}
    var_8 = module_0.Config(**var_7)
    var_9 = var_8.quiet
    assert var_9 is True
    var_10 = var_8.line_length
    assert var_10 == 88
    var_11 = var_8.multi_line_mode
    assert var_11 == 3
    var_12 = var_8.include_trailing_comma
    assert var_12 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_path_root_is_dir_predicate_evaluates_to_false.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_line_43_evaluates_to_true. Retrieved 9/17 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'builtins.warn'
    var_1 = None
    var_2 = lambda *args, **kwargs: var_1
    var_3 = 'isort.Config._get_config_data'
    var_4 = {}
    var_5 = 'isort.Config.__bases__[0].__init__'
    var_6 = '/test/settings.cfg'
    var_7 = False
    var_8 = 'quiet'
    var_9 = {var_8: var_7}
    var_10 = module_0.Config(var_6, **var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_all_configs. Retrieved 8/24 statements.
# Failed to parse test_find_all_configs_empty_directory.
# Partially parsed test_find_all_configs_with_invalid_config. Retrieved 2/7 statements.
# Partially parsed test_find_all_configs_nested_directories. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length=88\n'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nline_length=100\n'
    var_6 = 'pyproject.toml'
    var_7 = '[tool.isort]\nline_length=120\n'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[invalid section without closing\n'

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=80\n'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nline_length=120\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_all_configs. Retrieved 8/28 statements.
# Partially parsed test_find_all_configs_empty_directory. Retrieved 1/6 statements.
# Partially parsed test_find_all_configs_with_pyproject_toml. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'project'
    var_1 = 'src'
    var_2 = 'module'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nline_length=100\n'
    var_5 = 'setup.cfg'
    var_6 = '[isort]\nprofile=black\n'
    var_7 = 'test.py'

def test_case_0():
    var_0 = 'empty'

def test_case_0():
    var_0 = 'project'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.isort]\nline_length = 88\nprofile = "black"\n'
    var_3 = 'test.py'



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyc'
    var_1 = [var_0]
    var_2 = 'blocked_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.pyc'
    var_6 = var_4.is_supported_filetype(var_5)
    assert var_6 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py~'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/to/file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_post_init_py_version_auto. Retrieved 3/4 statements.
# Partially parsed test_config_post_init_multi_line_output_vertical_grid_grouped_no_comma. Retrieved 1/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py3'

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = '2.7'
    var_1 = module_0._Config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'not supported'

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_module'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '3'
    var_4 = module_0._Config(var_3, known_standard_library=var_2)
    var_5 = var_4.known_standard_library
    var_6 = bool(var_4.known_standard_library == var_2)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = True
    var_2 = module_0._Config(var_0, force_alphabetical_sort=var_1)
    var_3 = var_2.force_alphabetical_sort_within_sections
    assert var_3 is True
    var_4 = var_2.no_sections
    assert var_4 is True
    var_5 = var_2.lines_between_types
    assert var_5 == 1
    var_6 = var_2.from_first
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = 100
    var_2 = 79
    var_3 = module_0._Config(var_0, line_length=var_2, wrap_length=var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = 79
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 79
    var_4 = var_2.line_length
    assert var_4 == 79

def test_case_0():
    var_0 = '3'

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.line_length
    assert var_2 == 79
    var_3 = var_1.indent
    var_4 = bool(var_1.indent == ' ' * 4)
    assert var_4 is True
    var_5 = var_1.lines_between_sections
    assert var_5 == 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_config_settings_predicate_line_76. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = bool(var_2)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_line_66_predicate_evaluates_to_true. Retrieved 4/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = 'black'
    var_3 = 'profile'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_config_init_directory_set_to_cwd. Retrieved 2/4 statements.
# Partially parsed test_config_init_with_src_paths. Retrieved 5/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1._known_patterns
    assert var_3 is None
    var_4 = var_1._section_comments
    assert var_4 is None
    var_5 = var_1._section_comments_end
    assert var_5 is None
    var_6 = var_1._skips
    assert var_6 is None
    var_7 = var_1._skip_globs
    assert var_7 is None
    var_8 = var_1._sorting_function
    assert var_8 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'quiet'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5._known_patterns
    assert var_7 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.line_length
    assert var_7 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = True
    var_2 = 'indent'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.indent
    assert var_6 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = True
    var_2 = 'indent'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.indent
    assert var_6 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = '  '
    var_1 = True
    var_2 = 'indent'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.indent
    assert var_6 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sources
    var_5 = bool(var_3.sources is not None)
    assert var_5 is True
    var_6 = var_3.sources
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.src_paths
    var_5 = bool(var_3.src_paths is not None)
    assert var_5 is True
    var_6 = var_3.src_paths
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True
    var_9 = var_3.src_paths

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'known_other'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import_headings'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import_footers'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_skip_globs_initialization.




# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyc'
    var_1 = [var_0]
    var_2 = 'blocked_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.pyc'
    var_6 = var_4.is_supported_filetype(var_5)
    assert var_6 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py~'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/6 statements.
# Partially parsed test_config_init_with_known_section. Retrieved 4/5 statements.
# Failed to parse test_config_init_with_src_paths.
# Partially parsed test_config_init_with_wildcard_src_paths. Retrieved 2/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1._known_patterns
    assert var_3 is None
    var_4 = var_1._section_comments
    assert var_4 is None
    var_5 = var_1._section_comments_end
    assert var_5 is None
    var_6 = var_1._skips
    assert var_6 is None
    var_7 = var_1._skip_globs
    assert var_7 is None
    var_8 = var_1._sorting_function
    assert var_8 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'quiet'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5._known_patterns
    assert var_7 is None

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'InvalidSettingsPath'

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

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
    var_0 = '"tab"'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = frozenset()
    var_7 = 'django'

import isort.settings as module_0

def test_case_0():
    var_0 = 'from __future__ imports'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'future'
    var_6 = bool('future' in var_3.import_headings)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'stdlib footer'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'stdlib'
    var_6 = bool('stdlib' in var_3.import_footers)
    assert var_6 is True

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile_xyz'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ProfileDoesNotExist'

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    assert var_4 == '/tmp'

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = True
    var_2 = 2
    var_3 = 'tests'
    var_4 = [var_3]
    var_5 = 'profile'
    var_6 = 'quiet'
    var_7 = 'indent'
    var_8 = 'skip'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True
    var_12 = var_10.quiet
    assert var_12 is True
    var_13 = var_10.indent
    assert var_13 == '  '



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_config_init_with_directory.
# Failed to parse test_config_init_with_src_paths.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._known_patterns
    assert var_2 is None
    var_3 = var_1._section_comments
    assert var_3 is None
    var_4 = var_1._section_comments_end
    assert var_4 is None
    var_5 = var_1._skips
    assert var_5 is None
    var_6 = var_1._skip_globs
    assert var_6 is None
    var_7 = var_1._sorting_function
    assert var_7 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = var_3._known_patterns
    assert var_4 is None
    var_5 = var_3._section_comments
    assert var_5 is None
    var_6 = var_3._section_comments_end
    assert var_6 is None
    var_7 = var_3._skips
    assert var_7 is None
    var_8 = var_3._skip_globs
    assert var_8 is None
    var_9 = var_3._sorting_function
    assert var_9 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'InvalidSettingsPath'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 4
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
    var_0 = '  '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'natural'

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'native'

import isort.settings as module_0

def test_case_0():
    var_0 = '39'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.py_version
    assert var_4 == '39'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 0
    var_3 = 'quiet'
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.quiet
    assert var_8 is True
    var_9 = var_7.line_length
    assert var_9 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '_known_patterns'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = '_section_comments'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = '_section_comments_end'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = '_skips'
    var_12 = hasattr(var_1, var_11)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = '_skip_globs'
    var_15 = hasattr(var_1, var_14)
    var_16 = bool(var_15)
    assert var_16 is True
    var_17 = '_sorting_function'
    var_18 = hasattr(var_1, var_17)
    var_19 = bool(var_18)
    assert var_19 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'known_first_party'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'mymodule'
    var_6 = bool('mymodule' in var_4.known_first_party)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.py'
    var_6 = bool('test.py' in var_4.skip)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'build'
    var_1 = 'dist'
    var_2 = [var_0, var_1]
    var_3 = 'extend_skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'build'
    var_7 = bool('build' in var_5.extend_skip)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'sections'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = 'FUTURE'
    var_10 = bool('FUTURE' in var_8.sections)
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/9 statements.
# Failed to parse test_config_init_with_directory_override.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1._known_patterns
    assert var_3 is None
    var_4 = var_1._section_comments
    assert var_4 is None
    var_5 = var_1._section_comments_end
    assert var_5 is None
    var_6 = var_1._skips
    assert var_6 is None
    var_7 = var_1._skip_globs
    assert var_7 is None
    var_8 = var_1._sorting_function
    assert var_8 is None

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\n'

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'black'
    var_2 = 'line_length'
    var_3 = 'profile'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.line_length
    assert var_6 == 100
    var_7 = var_5.profile
    assert var_7 == 'black'

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
    var_0 = 80
    var_1 = 'django'
    var_2 = 'line_length'
    var_3 = 'profile'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 100
    var_7 = 'line_length'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(config=var_5, **var_8)
    var_10 = var_9.line_length
    assert var_10 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/settings'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 120
    var_2 = 'profile'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.profile
    assert var_6 == 'black'
    var_7 = var_5.line_length
    assert var_7 == 120

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._known_patterns
    assert var_2 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._section_comments
    assert var_2 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'venv'
    var_1 = [var_0]
    var_2 = '*.egg-info'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = bool('venv' in var_7.skip or var_7 is not None)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.src_paths
    var_3 = bool(var_1.src_paths is not None)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3.10'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.py_version
    assert var_4 == '3.10'

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'black'
    var_2 = 3
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'profile'
    var_6 = 'multi_line_mode'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = var_9.line_length
    assert var_10 == 100
    var_11 = var_9.profile
    assert var_11 == 'black'
    var_12 = var_9.include_trailing_comma
    assert var_12 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/6 statements.
# Partially parsed test_config_init_known_patterns_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_init_skips_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_init_skip_globs_lazy_loading. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1._known_patterns
    assert var_3 is None
    var_4 = var_1._section_comments
    assert var_4 is None
    var_5 = var_1._section_comments_end
    assert var_5 is None
    var_6 = var_1._skips
    assert var_6 is None
    var_7 = var_1._skip_globs
    assert var_7 is None
    var_8 = var_1._sorting_function
    assert var_8 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3._known_patterns
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 80

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=100\n'

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._known_patterns
    assert var_2 is None
    var_3 = var_1.known_patterns
    var_4 = var_1._known_patterns
    var_5 = bool(var_1._known_patterns is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._section_comments
    assert var_2 is None
    var_3 = var_1.section_comments
    var_4 = var_1._section_comments
    var_5 = bool(var_1._section_comments is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._skips
    assert var_2 is None
    var_3 = var_1.skips
    var_4 = var_1._skips
    var_5 = bool(var_1._skips is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._skip_globs
    assert var_2 is None
    var_3 = var_1.skip_globs
    var_4 = var_1._skip_globs
    var_5 = bool(var_1._skip_globs is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._sorting_function
    assert var_2 is None
    var_3 = var_1.sorting_function
    var_4 = var_1._sorting_function
    var_5 = bool(var_1._sorting_function is not None)
    assert var_5 is True
    var_6 = callable(var_3)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_66_predicate_evaluates_to_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = {}
    var_3 = 'black'
    var_4 = var_3 not in var_2
    assert var_4 is True



