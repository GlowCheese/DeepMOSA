####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_post_init_py_version_auto. Retrieved 2/4 statements.
# Failed to parse test_post_init_multi_line_output_vertical_grid_grouped_no_comma.


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
    var_2 = var_1.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The python version invalid is not supported'

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

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = 79
    var_1 = module_0._Config(line_length=var_0, wrap_length=var_0)
    var_2 = var_1.wrap_length
    assert var_2 == 79
    var_3 = var_1.line_length
    assert var_3 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = var_2.wrap_length
    assert var_3 == 50
    var_4 = var_2.line_length
    assert var_4 == 79



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir/'
    var_2 = 'subdir/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = '/home/user/dir/'
    var_6 = '/home/user/subdir/'
    var_7 = {var_5, var_6}
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/usr/local/'
    var_2 = '/tmp/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = {var_1, var_2}
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir/'
    var_2 = '/usr/local/'
    var_3 = 'file.txt'
    var_4 = '/tmp/'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0._abspaths(var_0, var_5)
    var_7 = '/home/user/dir/'
    var_8 = '/home/user/file.txt'
    var_9 = {var_7, var_2, var_8, var_4}
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir'
    var_2 = 'file.txt'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = '/home/user/dir'
    var_6 = '/home/user/file.txt'
    var_7 = {var_5, var_6}
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = '/home/user/dir'
    var_5 = {var_4}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/usr/local'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = {var_1}
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/7 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 3/10 statements.
# Partially parsed test_config_constructor_without_parameters. Retrieved 1/2 statements.
# Partially parsed test_config_constructor_with_quiet_false_and_warnings. Retrieved 2/8 statements.
# Partially parsed test_config_constructor_with_directory_and_src_paths. Retrieved 1/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = module_0._Config(var_0)
    var_2 = True
    var_3 = 'quiet'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = var_5.py_version
    assert var_6 == '310'

import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = module_0._Config(var_0)
    var_2 = 'py38'
    var_3 = 'py_version'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = var_5.py_version
    assert var_6 == '38'

def test_case_0():
    var_0 = '[tool.isort]\nprofile = "black"\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'CUSTOM'
    var_3 = [var_2]
    var_4 = 'known_custom'
    var_5 = 'sections'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'custom'
    var_9 = bool('custom' in var_7.known_other)
    assert var_9 is True
    var_10 = 'mypackage'
    var_11 = bool('mypackage' in var_7.known_other['custom'])
    assert var_11 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Standard Library'
    var_1 = 'import_heading_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['stdlib']
    assert var_4 == 'Standard Library'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End Standard Library'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['stdlib']
    assert var_4 == 'End Standard Library'

def test_case_0():
    var_0 = 'src'
    var_1 = '.'
    var_2 = [var_0, var_1]

import isort.settings as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'atomic'
    var_2 = 'quiet'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'atomic'
    var_6 = hasattr(var_4, var_5)
    var_7 = bool(not var_6)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory

import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = 80
    var_2 = module_0._Config(var_0, line_length=var_1)
    var_3 = 100
    var_4 = 'line_length'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(config=var_2, **var_5)
    var_7 = var_6.py_version
    assert var_7 == '310'
    var_8 = var_6.line_length
    assert var_8 == 100

def test_case_0():
    var_0 = '[settings]\nline_length = 100\n'
    var_1 = False

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'known_standard_library'
    var_6 = 'known_stdlib'
    var_7 = 'quiet'
    var_8 = {var_5: var_1, var_6: var_3, var_7: var_4}
    var_9 = module_0.Config(**var_8)
    var_10 = 'os'
    var_11 = bool('os' in var_9.known_standard_library)
    assert var_11 is True
    var_12 = 'sys'
    var_13 = bool('sys' not in var_9.known_standard_library)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'sections'
    var_4 = 'quiet'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'CUSTOM'
    var_8 = bool('CUSTOM' in var_6.sections)
    assert var_8 is True

def test_case_0():
    var_0 = 'src'

import isort.settings as module_0

def test_case_0():
    var_0 = 120
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 120

import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = 80
    var_2 = module_0._Config(var_0, line_length=var_1)
    var_3 = 100
    var_4 = 'py310'
    var_5 = 'line_length'
    var_6 = 'py_version'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_0.Config(config=var_2, **var_7)
    var_9 = var_8.py_version
    assert var_9 == '310'
    var_10 = var_8.line_length
    assert var_10 == 100



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_supported_filetype_with_supported_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_shebang. Retrieved 2/12 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_no_shebang. Retrieved 2/12 statements.
# Partially parsed test_is_supported_filetype_with_editor_backup_file. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 2/12 statements.
# Partially parsed test_is_supported_filetype_with_nonexistent_file. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'log'
    var_3 = 'error.log'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'#!/usr/bin/env python\n'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'no shebang here'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'test.py~'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'nonexistent.py'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_all_configs_returns_trie_with_default_config. Retrieved 2/3 statements.
# Partially parsed test_find_all_configs_inserts_valid_config. Retrieved 2/8 statements.
# Partially parsed test_find_all_configs_ignores_invalid_config. Retrieved 2/8 statements.
# Partially parsed test_find_all_configs_prefers_first_valid_config_in_directory. Retrieved 4/13 statements.
# Partially parsed test_find_all_configs_walks_subdirectories. Retrieved 3/11 statements.
# Partially parsed test_find_all_configs_handles_editorconfig. Retrieved 2/8 statements.
# Partially parsed test_find_all_configs_handles_empty_directory. Retrieved 1/5 statements.
# Partially parsed test_find_all_configs_handles_nested_configs. Retrieved 6/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info[0]
    assert var_2 == 'default'
    var_3 = var_1.root.config_info[1]
    var_4 = bool(var_1.root.config_info[1] == {})
    assert var_4 is True

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nline_length = 120'

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.isort]\nline_length = 100'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 2'

def test_case_0():
    var_0 = 'some_file.py'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100'
    var_2 = 'subdir'
    var_3 = '[tool.isort]\nline_length = 200'
    var_4 = 'root.py'
    var_5 = 'sub.py'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_section_in_section_defaults. Retrieved 7/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'sections'
    var_3 = 'STDLIB'
    var_4 = (var_3,)
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 3/10 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = True
    var_2 = 4
    var_3 = module_0._Config(var_0, indent=var_2, quiet=var_1)
    var_4 = False
    var_5 = 2
    var_6 = 'quiet'
    var_7 = 'indent'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = module_0.Config(config=var_3, **var_8)
    var_10 = var_9.py_version
    assert var_10 == '310'
    var_11 = var_9.quiet
    assert var_11 is False
    var_12 = var_9.indent
    assert var_12 == '  '

def test_case_0():
    var_0 = '[tool.isort]\nprofile = "black"\nline_length = 100'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nprofile = "black"'

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 100
    var_2 = 'profile'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.profile
    assert var_6 == 'black'
    var_7 = var_5.line_length
    assert var_7 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = 'FUTURE'
    var_4 = 'STDLIB'
    var_5 = 'FOO'
    var_6 = 'THIRDPARTY'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = 'known_foo'
    var_9 = 'sections'
    var_10 = {var_8: var_2, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = 'foo'
    var_13 = bool('foo' in var_11.known_other)
    assert var_13 is True
    var_14 = [var_0, var_1]
    var_15 = frozenset(var_14)
    var_16 = var_11.known_other['foo']
    var_17 = bool(var_11.known_other['foo'] == var_15)
    assert var_17 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Standard Library'
    var_1 = 'import_heading_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['stdlib']
    assert var_4 == 'Standard Library'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End Standard Library'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['stdlib']
    assert var_4 == 'End Standard Library'

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
    var_0 = '2'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]

import isort.settings as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(var_3.formatting_function is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'force_alphabetical_sort'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.force_alphabetical_sort
    var_5 = bool(var_3.force_alphabetical_sort is not True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'

def test_case_0():
    var_0 = '[settings]\nline_length = 100'
    var_1 = True

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_all_configs_break_on_first_valid_config. Retrieved 25/42 statements.


import isort.utils as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '.isort.cfg'
    var_2 = 'setup.cfg'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'tool.isort'
    var_5 = 'settings'
    var_6 = 'tool:isort'
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = '/test/path'
    var_9 = 'default'
    var_10 = {}
    var_11 = module_0.Trie(var_9, var_10)
    var_12 = '/test/path'
    var_13 = []
    var_14 = 'file1.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = '/test/path/sub'
    var_18 = []
    var_19 = 'file2.py'
    var_20 = [var_19]
    var_21 = (var_17, var_18, var_20)
    var_22 = [var_16, var_21]
    var_23 = False
    var_24 = True
    var_25 = var_11.root.config_info
    var_26 = bool(var_11.root.config_info == ('default', {}))
    assert var_26 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_config_with_existing_toml. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_editorconfig. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_setup_cfg. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_tox_ini. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_isort_cfg. Retrieved 2/9 statements.
# Failed to parse test_find_config_with_no_config.
# Partially parsed test_find_config_stops_at_stop_dir. Retrieved 4/15 statements.
# Partially parsed test_find_config_searches_upwards. Retrieved 4/13 statements.
# Partially parsed test_find_config_with_max_depth. Retrieved 2/14 statements.
# Partially parsed test_find_config_with_invalid_config_file. Retrieved 4/18 statements.
# Partially parsed test_find_config_editorconfig_with_off_max_line_length. Retrieved 4/11 statements.
# Partially parsed test_find_config_editorconfig_with_tab_indent. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_force_grid_wrap_backwards_compat. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_comment_prefix_stripping. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_known_prefix_paths. Retrieved 3/11 statements.
# Partially parsed test_find_config_editorconfig_with_extension_section. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 100\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 120\n'

def test_case_0():
    var_0 = 'tox.ini'
    var_1 = '[isort]\nline_length = 80\n'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length = 90\n'

def test_case_0():
    var_0 = '.git'
    var_1 = 'subdir'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'
    var_2 = 'subdir'
    var_3 = 'nested'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content'
    var_2 = 'always'
    var_3 = 0
    var_4 = 'Failed to pull configuration information'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = 'inf'
    var_3 = float(var_2)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = false\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\ncomment_prefix = "# "\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_local_folder = ./local\n'
    var_2 = 'local'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyi}]\nindent_size = 2\n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_constructor_with_config_and_overrides. Retrieved 4/6 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = 'overridden'
    var_2 = True
    var_3 = 'some_setting'
    var_4 = 'quiet'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(config=var_0, **var_5)
    var_7 = var_6.py_version
    assert var_7 == '310'
    var_8 = var_6.some_setting
    assert var_8 == 'overridden'
    var_9 = var_6.quiet
    assert var_9 is True

def test_case_0():
    var_0 = '[tool.isort]\nprofile = "black"\nline_length = 100'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nprofile = "django"'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'MYSECTION'
    var_4 = [var_2, var_3]
    var_5 = 'known_mysection'
    var_6 = 'sections'
    var_7 = {var_5: var_1, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = 'mysection'
    var_10 = bool('mysection' in var_8.known_other)
    assert var_10 is True
    var_11 = 'mypackage'
    var_12 = bool('mypackage' in var_8.known_other['mysection'])
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'My Section'
    var_1 = 'import_heading_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['mysection']
    assert var_4 == 'My Section'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of My Section'
    var_1 = 'import_footer_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['mysection']
    assert var_4 == 'End of My Section'

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_setting'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = 'quiet'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = dir(var_4)
    var_6 = 'skip_gitignore'
    var_7 = bool('skip_gitignore' not in var_5)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'example_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths
    var_7 = len(var_6)
    assert var_7 == 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_supported_filetype_with_supported_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_shebang. Retrieved 4/9 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_no_shebang. Retrieved 4/9 statements.
# Partially parsed test_is_supported_filetype_with_editor_backup_file. Retrieved 4/9 statements.
# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_with_nonexistent_file. Retrieved 3/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'log'
    var_3 = 'test.log'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.sh'
    var_3 = '#!/bin/bash\n'
    var_4 = var_1.is_supported_filetype(var_2)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.xyz'
    var_3 = 'no shebang here\n'
    var_4 = var_1.is_supported_filetype(var_2)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py~'
    var_3 = '#!/usr/bin/env python\n'
    var_4 = var_1.is_supported_filetype(var_2)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_fifo'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent.xyz'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__get_config_data_with_toml. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_ini. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_editorconfig_indent_spaces. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_editorconfig_indent_tabs. Retrieved 5/12 statements.
# Partially parsed test__get_config_data_with_editorconfig_extension_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_unknown_keys_filtered. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_bool_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_backwards_compat_false. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_backwards_compat_true. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_comment_prefix_stripping. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_known_prefix_paths. Retrieved 6/20 statements.
# Partially parsed test__get_config_data_empty_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_multiple_sections. Retrieved 4/11 statements.


def test_case_0():
    var_0 = b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'root = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 100\n'
    var_1 = '*'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*]\nindent_style = tab\nindent_size = tab\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*]\nindent_size = 4\n\n[*.{py,pyi}]\nindent_size = 8\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*]\nindent_size = 4\nunknown_key = value\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'unknown_key'
    var_4 = 'indent'

def test_case_0():
    var_0 = '[black]\nskip_string_normalization = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = true\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\ncomment_prefix = "# "\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nknown_first_party = mod1,mod2\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'mod1'
    var_4 = 'mod2'
    var_5 = 'known_first_party'

def test_case_0():
    var_0 = '[black]\nline_length = 88\n\n[other]\nkey = value\n'
    var_1 = 'nonexistent'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\n\n[pycodestyle]\nmax_line_length = 79\n'
    var_1 = 'black'
    var_2 = 'pycodestyle'
    var_3 = (var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_config_settings_source_exists. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = '/some/path'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_supported_filetype_with_supported_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_shebang. Retrieved 4/9 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_no_shebang. Retrieved 4/9 statements.
# Partially parsed test_is_supported_filetype_with_editor_backup_file. Retrieved 3/5 statements.
# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 3/6 statements.
# Partially parsed test_is_supported_filetype_with_file_open_error. Retrieved 3/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'log'
    var_3 = 'error.log'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'#!/usr/bin/env python\n'
    var_3 = 'script'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b"print('hello')\n"
    var_3 = 'script'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

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
    var_2 = 'fifo'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_init_with_config_and_overrides. Retrieved 15/28 statements.


def test_case_0():
    var_0 = 'py310'
    var_1 = 'value'
    var_2 = 'quiet'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'py_version'
    var_6 = 'py'
    var_7 = ''
    var_8 = '_known_patterns'
    var_9 = None
    var_10 = '_section_comments'
    var_11 = '_section_comments_end'
    var_12 = '_skips'
    var_13 = '_skip_globs'
    var_14 = '_sorting_function'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_config_initialization_with_config_parameter. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '310'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'txt'
    var_3 = 'test.txt'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_config_with_valid_toml_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_valid_editorconfig_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_invalid_config_file. Retrieved 2/9 statements.
# Failed to parse test_find_config_with_no_config_file.
# Partially parsed test_find_config_stop_search_on_stop_dir. Retrieved 4/15 statements.
# Partially parsed test_find_config_search_up_directory_tree. Retrieved 4/13 statements.
# Partially parsed test_find_config_max_search_depth. Retrieved 2/14 statements.
# Partially parsed test_find_config_prioritize_pyproject_over_editorconfig. Retrieved 4/14 statements.
# Partially parsed test_find_config_with_force_grid_wrap_backwards_compatibility. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_comment_prefix_stripping. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nline_length = 100\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content'

def test_case_0():
    var_0 = '.git'
    var_1 = 'subdir'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.black]\nline_length = 88\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nline_length = 120\n'
    var_2 = 'subdir'
    var_3 = 'nested'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nline_length = 79\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nline_length = 100\n'
    var_2 = '.editorconfig'
    var_3 = '[*.py]\nline_length = 80\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nforce_grid_wrap = false\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\ncomment_prefix = "# "\n'



# Parsed testcases at query #19
#--------------------------




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
    var_0 = 'pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory
    var_4 = bool(var_2.directory is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.directory
    var_4 = bool(var_2.directory is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'quiet'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = var_5.quiet
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = '2'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

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
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'known_custom'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'custom'
    var_6 = bool('custom' in var_4.known_other)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Custom Imports'
    var_1 = 'import_heading_custom'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'custom'
    var_5 = bool('custom' in var_3.import_headings)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End Custom Imports'
    var_1 = 'import_footer_custom'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'custom'
    var_5 = bool('custom' in var_3.import_footers)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths
    var_7 = len(var_6)
    assert var_7 == 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'unknown_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = 'force_alphabetical_sort'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.force_alphabetical_sort
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = True
    var_2 = 4
    var_3 = 'profile'
    var_4 = 'quiet'
    var_5 = 'indent'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.profile
    assert var_8 == 'black'
    var_9 = var_7.quiet
    assert var_9 is True
    var_10 = var_7.indent
    assert var_10 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = {}
    var_5 = module_0.Config(config=var_3, **var_4)
    var_6 = var_5.py_version
    assert var_6 == '310'

import isort.settings as module_0

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = [var_2]
    var_4 = 'sections'
    var_5 = 'known_custom'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'CUSTOM'
    var_9 = bool('CUSTOM' in var_7.sections)
    assert var_9 is True
    var_10 = 'custom'
    var_11 = bool('custom' in var_7.known_other)
    assert var_11 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = 'directory'
    var_4 = 'src_paths'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = var_6.directory
    assert var_7 == '/tmp'
    var_8 = var_6.src_paths
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_38_true. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '[section1]\nkey1 = value1\nkey2 = value2\n'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = 'source'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_initialization_with_config_parameter. Retrieved 2/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '310'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_warning_when_settings_file_empty_and_not_quiet. Retrieved 2/11 statements.


def test_case_0():
    var_0 = ''
    var_1 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_config_and_overrides. Retrieved 4/6 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 2/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/6 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_number. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_other. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 5/12 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_settings. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 3/7 statements.
# Partially parsed test_config_constructor_with_invalid_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_invalid_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_py_version. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_combined_config. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_config_and_no_overrides. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_settings_file_and_overrides. Retrieved 3/7 statements.
# Partially parsed test_config_constructor_with_settings_path_and_overrides. Retrieved 3/7 statements.
# Partially parsed test_config_constructor_with_profile_and_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_indent_and_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_known_other_and_overrides. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_import_headings_and_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_import_footers_and_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_src_paths_and_overrides. Retrieved 7/14 statements.
# Partially parsed test_config_constructor_with_formatter_and_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_deprecated_options_and_overrides. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_with_unsupported_settings_and_overrides. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_empty_settings_file_and_overrides. Retrieved 3/7 statements.
# Partially parsed test_config_constructor_with_invalid_profile_and_overrides. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_invalid_formatter_and_overrides. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order_and_overrides. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_py_version_and_overrides. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[isort]\nquiet = true'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[isort]\nquiet = true'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 4
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_mysection'
    var_1 = 'mypackage'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'mysection'
    var_5 = 'mypackage'

def test_case_0():
    var_0 = 'import_heading_mysection'
    var_1 = 'My Section'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'import_footer_mysection'
    var_1 = 'End of My Section'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = 'tests'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'console'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'force_sort_within_sections'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'force_sort_within_sections'

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nquiet = true'
    var_2 = True

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'py310'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = 'profile'
    var_2 = 'indent'
    var_3 = True
    var_4 = 'black'
    var_5 = 4
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.quiet
    assert var_2 is False

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[isort]\nquiet = false'
    var_2 = True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[isort]\nquiet = false'
    var_2 = True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'quiet'
    var_2 = 'black'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'quiet'
    var_2 = 2
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'known_mysection'
    var_1 = 'quiet'
    var_2 = 'mypackage'
    var_3 = [var_2]
    var_4 = True
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'mysection'
    var_7 = 'mypackage'

def test_case_0():
    var_0 = 'import_heading_mysection'
    var_1 = 'quiet'
    var_2 = 'My Section'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'mysection'

def test_case_0():
    var_0 = 'import_footer_mysection'
    var_1 = 'quiet'
    var_2 = 'End of My Section'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'mysection'

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'quiet'
    var_2 = 'src'
    var_3 = 'tests'
    var_4 = [var_2, var_3]
    var_5 = True
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'quiet'
    var_2 = 'console'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'force_sort_within_sections'
    var_1 = 'quiet'
    var_2 = True
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 'force_sort_within_sections'

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'quiet'
    var_2 = 'value'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nquiet = false'
    var_2 = True

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = True
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(settings_path=var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'quiet'
    var_2 = 'invalid'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'quiet'
    var_2 = 'invalid'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'quiet'
    var_2 = 'invalid'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'py310'
    var_2 = {var_0: var_1}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_supported_filetype_with_supported_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_blocked_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_shebang. Retrieved 2/12 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_no_shebang. Retrieved 2/12 statements.
# Partially parsed test_is_supported_filetype_with_backup_file. Retrieved 4/6 statements.
# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 2/13 statements.
# Partially parsed test_is_supported_filetype_with_nonexistent_file. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'log'
    var_3 = 'tmp'
    var_4 = 'test.log'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'#!/usr/bin/env python\n'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'no shebang here\n'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'test.py~'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'nonexistent.py'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'txt'
    var_3 = 'test.txt'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_constructor_with_config_parameter. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/7 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_constructor_with_deprecated_option. Retrieved 4/10 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 6/10 statements.
# Partially parsed test_config_constructor_with_glob_src_paths. Retrieved 5/13 statements.
# Partially parsed test_config_constructor_with_quiet_false_and_warnings. Retrieved 4/7 statements.
# Partially parsed test_config_constructor_with_quiet_true_and_no_warnings. Retrieved 3/6 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 3/15 statements.
# Partially parsed test_config_constructor_with_sections_and_known_other_mismatch. Retrieved 16/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '39'

import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = True
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(config=var_0, **var_3)
    var_5 = var_4.quiet
    assert var_5 is True

def test_case_0():
    var_0 = '[tool.isort]\nprofile = "black"\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

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
    var_0 = 'nonexistent_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = '2'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

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
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'known_mysection'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'mysection'
    var_6 = bool('mysection' in var_4.known_other)
    assert var_6 is True
    var_7 = 'mypackage'
    var_8 = bool('mypackage' in var_4.known_other['mysection'])
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'My Section'
    var_1 = 'import_heading_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['mysection']
    assert var_4 == 'My Section'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of My Section'
    var_1 = 'import_footer_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['mysection']
    assert var_4 == 'End of My Section'

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_setting'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = True
    var_2 = 'force_alphabetical_sort'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 0
    var_6 = 'Deprecated config options were used'

import isort.settings as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths
    var_7 = var_5.src_paths

def test_case_0():
    var_0 = 'src'
    var_1 = 'module1'
    var_2 = 'module2'
    var_3 = 'src/*'
    var_4 = [var_3]

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_nonexistent'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = True
    var_2 = False
    var_3 = 'force_alphabetical_sort'
    var_4 = 'quiet'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = True
    var_2 = 'force_alphabetical_sort'
    var_3 = 'quiet'
    var_4 = {var_2: var_1, var_3: var_1}
    var_5 = module_0.Config(**var_4)

def test_case_0():
    var_0 = '[settings]\nline_length = 100\n'
    var_1 = 'always'
    var_2 = False
    var_3 = 'no configuration was found inside'

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'STDLIB'
    var_2 = 'CUSTOM'
    var_3 = (var_1, var_2)
    var_4 = 'mypackage'
    var_5 = [var_4]
    var_6 = 'sections'
    var_7 = 'known_custom'
    var_8 = {var_6: var_3, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = 'STDLIB'
    var_11 = 'CUSTOM'
    var_12 = (var_10, var_11)
    assert var_12 == 1
    var_13 = 'sections'
    var_14 = {var_13: var_12}
    var_15 = module_0.Config(**var_14)
    var_16 = 'always'
    var_17 = var_15.known_patterns
    var_18 = 0
    var_19 = var_3.message
    var_20 = str(var_19)
    var_21 = 'no known_custom is defined'
    var_22 = bool('no known_custom is defined' in var_20)
    assert var_22 is True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir/'
    var_2 = 'subdir/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = '/home/user/dir/'
    var_6 = '/home/user/subdir/'
    var_7 = {var_5, var_6}
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/usr/local/'
    var_2 = '/tmp/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = {var_1, var_2}
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir/'
    var_2 = '/usr/local/'
    var_3 = 'file.txt'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = '/home/user/dir/'
    var_7 = '/home/user/file.txt'
    var_8 = {var_6, var_2, var_7}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir'
    var_2 = 'file.txt'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = '/home/user/dir'
    var_6 = '/home/user/file.txt'
    var_7 = {var_5, var_6}
    var_8 = bool(var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_constructor_with_empty_settings. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = True
    var_2 = 4
    var_3 = module_0._Config(var_0, indent=var_2, quiet=var_1)
    var_4 = False
    var_5 = 2
    var_6 = 'quiet'
    var_7 = 'indent'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = module_0.Config(config=var_3, **var_8)
    var_10 = var_9.py_version
    assert var_10 == '310'
    var_11 = var_9.quiet
    assert var_11 is False
    var_12 = var_9.indent
    assert var_12 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_file.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = '*.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'skip_glob'
    var_4 = 'quiet'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'skip_glob'
    var_8 = bool('skip_glob' not in var_6.__dict__)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'MYSECTION'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'known_mysection'
    var_6 = 'sections'
    var_7 = 'quiet'
    var_8 = {var_5: var_1, var_6: var_3, var_7: var_4}
    var_9 = module_0.Config(**var_8)
    var_10 = 'known_other'
    var_11 = bool('known_other' in var_9.__dict__)
    assert var_11 is True
    var_12 = 'mysection'
    var_13 = bool('mysection' in var_9.known_other)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'My Section'
    var_1 = True
    var_2 = 'import_heading_mysection'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import_headings'
    var_7 = bool('import_headings' in var_5.__dict__)
    assert var_7 is True
    var_8 = 'mysection'
    var_9 = bool('mysection' in var_5.import_headings)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Footer'
    var_1 = True
    var_2 = 'import_footer_mysection'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import_footers'
    var_7 = bool('import_footers' in var_5.__dict__)
    assert var_7 is True
    var_8 = 'mysection'
    var_9 = bool('mysection' in var_5.import_footers)
    assert var_9 is True

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
    var_7 = 'tab'
    var_8 = 'indent'
    var_9 = 'quiet'
    var_10 = {var_8: var_7, var_9: var_1}
    var_11 = module_0.Config(**var_10)
    var_12 = var_11.indent
    assert var_12 == '\t'
    var_13 = "'  '"
    var_14 = 'indent'
    var_15 = 'quiet'
    var_16 = {var_14: var_13, var_15: var_1}
    var_17 = module_0.Config(**var_16)
    var_18 = var_17.indent
    assert var_18 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.src_paths
    var_5 = len(var_4)
    assert var_5 == 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'lib'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'src_paths'
    var_5 = 'quiet'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.src_paths
    var_9 = len(var_8)
    var_10 = bool(var_9 >= 2)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = 'directory'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.directory
    assert var_6 == '/some/path'

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
    var_0 = 'py39'
    var_1 = True
    var_2 = 'py_version'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.py_version
    assert var_6 == '39'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'py_version'
    var_5 = ''
    var_6 = 'py'
    var_7 = var_3.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._known_patterns
    assert var_4 is None
    var_5 = var_3.known_patterns
    var_6 = var_3._known_patterns
    var_7 = bool(var_3._known_patterns is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._skips
    assert var_4 is None
    var_5 = var_3.skips
    var_6 = var_3._skips
    var_7 = bool(var_3._skips is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._skip_globs
    assert var_4 is None
    var_5 = var_3.skip_globs
    var_6 = var_3._skip_globs
    var_7 = bool(var_3._skip_globs is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._sorting_function
    assert var_4 is None
    var_5 = var_3.sorting_function
    var_6 = var_3._sorting_function
    var_7 = bool(var_3._sorting_function is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._section_comments
    assert var_4 is None
    var_5 = var_3.section_comments
    var_6 = var_3._section_comments
    var_7 = bool(var_3._section_comments is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._section_comments_end
    assert var_4 is None
    var_5 = var_3.section_comments_end
    var_6 = var_3._section_comments_end
    var_7 = bool(var_3._section_comments_end is not None)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/7 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_constructor_with_deprecated_setting. Retrieved 4/10 statements.
# Partially parsed test_config_constructor_with_src_paths_expansion. Retrieved 3/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = True
    var_2 = 4
    var_3 = module_0._Config(var_0, indent=var_2, quiet=var_1)
    var_4 = False
    var_5 = 100
    var_6 = 'quiet'
    var_7 = 'line_length'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = module_0.Config(config=var_3, **var_8)
    var_10 = var_9.py_version
    assert var_10 == '310'
    var_11 = var_9.quiet
    assert var_11 is False
    var_12 = var_9.line_length
    assert var_12 == 100

def test_case_0():
    var_0 = '[isort]\nline_length = 88\nprofile = "black"\n'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length = 79\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

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
    var_0 = 'nonexistent_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = '"\\t"'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'STDLIB'
    var_3 = 'MYSECTION'
    var_4 = [var_2, var_3]
    var_5 = 'known_mysection'
    var_6 = 'sections'
    var_7 = {var_5: var_1, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = 'mysection'
    var_10 = bool('mysection' in var_8.known_other)
    assert var_10 is True
    var_11 = 'mypackage'
    var_12 = bool('mypackage' in var_8.known_other['mysection'])
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Standard Library'
    var_1 = 'import_heading_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['stdlib']
    assert var_4 == 'Standard Library'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of Standard Library'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['stdlib']
    assert var_4 == 'End of Standard Library'

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_setting'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = True
    var_2 = 'force_sort_within_sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 0
    var_6 = 'Deprecated config options were used'

import isort.settings as module_0

def test_case_0():
    var_0 = 'example_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

def test_case_0():
    var_0 = 'src'
    var_1 = '.'
    var_2 = [var_0, var_1]

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_config_and_overrides. Retrieved 4/6 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 2/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 3/9 statements.
# Partially parsed test_config_constructor_with_profile_override. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_other_section. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_number. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 5/8 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_option. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_unsupported_setting. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_config_object. Retrieved 6/8 statements.
# Partially parsed test_config_constructor_with_py_version_conversion. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 3/7 statements.
# Partially parsed test_config_constructor_with_nonexistent_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_combined_sources. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_directory_auto_detection. Retrieved 2/7 statements.
# Partially parsed test_config_constructor_with_skip_gitignore. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = "[tool.isort]\nprofile = 'black'"

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'pyproject.toml'
    var_2 = "[tool.isort]\nprofile = 'black'"

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_mysection'
    var_1 = 'mypackage'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'mysection'
    var_5 = 'mypackage'

def test_case_0():
    var_0 = 'import_heading_mysection'
    var_1 = 'My Section'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'import_footer_mysection'
    var_1 = 'My Footer'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'indent'
    var_1 = 4
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = 'tests'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'console'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'force_alphabetical_sort'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'force_alphabetical_sort'

def test_case_0():
    var_0 = 'unsupported_setting'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'quiet'
    var_1 = 'profile'
    var_2 = False
    var_3 = 'black'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'py310'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = ''
    var_2 = True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'line_length'
    var_1 = 'profile'
    var_2 = 100
    var_3 = 'black'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 120'

def test_case_0():
    var_0 = 'skip_gitignore'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_config_with_existing_toml_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_editorconfig_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_setup_cfg_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_tox_ini_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_existing_isort_cfg_file. Retrieved 2/9 statements.
# Failed to parse test_find_config_with_no_config_file.
# Partially parsed test_find_config_stops_at_stop_dir. Retrieved 4/15 statements.
# Partially parsed test_find_config_searches_upwards. Retrieved 4/13 statements.
# Partially parsed test_find_config_limits_search_depth. Retrieved 2/14 statements.
# Partially parsed test_find_config_with_invalid_config_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_prioritizes_first_found_config. Retrieved 4/14 statements.
# Partially parsed test_find_config_with_editorconfig_max_line_length_off. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_editorconfig_tab_indent. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_editorconfig_no_indent_size. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_editorconfig_tab_width. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_editorconfig_wildcard_extension. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_force_grid_wrap_backwards_compat. Retrieved 2/9 statements.
# Failed to parse test_find_config_with_comment_prefix.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 100\n'

def test_case_0():
    var_0 = 'tox.ini'
    var_1 = '[isort]\nline_length = 120\n'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length = 80\n'

def test_case_0():
    var_0 = '.git'
    var_1 = 'subdir'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'
    var_2 = 'subdir1'
    var_3 = 'subdir2'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length = 100\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = off\n'
    var_2 = 'inf'
    var_3 = float(var_2)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = tab\n'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyi}]\nindent_style = space\nindent_size = 4\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nforce_grid_wrap = false\n'



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'deprecated_option'
    var_4 = {var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'deprecated_option'
    var_7 = bool('deprecated_option' not in var_5._config)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_initialization_with_config_parameter. Retrieved 3/7 statements.


def test_case_0():
    var_0 = '_Config'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_find_all_configs_with_no_configs.
# Partially parsed test_find_all_configs_with_single_config. Retrieved 3/11 statements.
# Partially parsed test_find_all_configs_with_nested_configs. Retrieved 5/21 statements.
# Partially parsed test_find_all_configs_with_multiple_config_formats. Retrieved 5/15 statements.
# Partially parsed test_find_all_configs_with_invalid_config_file. Retrieved 4/12 statements.
# Partially parsed test_find_all_configs_with_editorconfig. Retrieved 3/11 statements.
# Partially parsed test_find_all_configs_with_skip_config. Retrieved 3/10 statements.
# Partially parsed test_find_all_configs_with_complex_nesting. Retrieved 6/24 statements.


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100'
    var_2 = 'file.py'

def test_case_0():
    var_0 = 'subdir'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=80'
    var_3 = '[settings]\nline_length=120'
    var_4 = 'file.py'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length=90'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length=110'
    var_4 = 'file.py'

def test_case_0():
    var_0 = 'isort.settings.warn'
    var_1 = '.isort.cfg'
    var_2 = 'invalid content'
    var_3 = 'file.py'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style=space\nindent_size=2'
    var_2 = 'file.py'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.js]\nindent_style=tab'
    var_2 = 'file.py'

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nline_length=70'
    var_4 = '[settings]\nline_length=130'
    var_5 = 'file.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test___post_init___py_version_auto. Retrieved 2/4 statements.
# Failed to parse test___post_init___multi_line_output_vertical_grid_grouped_no_comma.


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
    var_2 = var_1.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The python version invalid is not supported'

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
    var_0 = '3'
    var_1 = 'os'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = module_0._Config(var_0, known_standard_library=var_3)
    var_5 = 'os'
    var_6 = bool('os' in var_4.known_standard_library)
    assert var_6 is True

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

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = 79
    var_1 = module_0._Config(line_length=var_0, wrap_length=var_0)
    var_2 = var_1.wrap_length
    assert var_2 == 79
    var_3 = var_1.line_length
    assert var_3 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = var_2.wrap_length
    assert var_3 == 50
    var_4 = var_2.line_length
    assert var_4 == 79



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'some_deprecated_option'
    var_1 = 'quiet'
    var_2 = 'value'
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'some_deprecated_option'
    var_6 = 'quiet'
    var_7 = {var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = 'some_deprecated_option'
    var_10 = bool('some_deprecated_option' not in var_8._deprecated_options_used)
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_config_and_overrides. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/9 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/12 statements.
# Partially parsed test_config_constructor_with_invalid_settings_path. Retrieved 1/9 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_nonexistent_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_indent_as_number. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_string. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_other. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_option. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_unsupported_setting. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/6 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 5/8 statements.
# Partially parsed test_config_constructor_with_skip_and_extend_skip. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_skip_glob_and_extend_skip_glob. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_py_version. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_config_copy. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_config_and_overrides_combined. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = '[tool.isort]\nprofile = "black"\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nprofile = "black"\n'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'InvalidSettingsPath'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = 'ProfileDoesNotExist'

def test_case_0():
    var_0 = 'indent'
    var_1 = 4
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '2'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_mysection'
    var_1 = 'mypackage'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'mysection'
    var_5 = 'mypackage'

def test_case_0():
    var_0 = 'import_heading_mysection'
    var_1 = 'My Section'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'import_footer_mysection'
    var_1 = 'End of My Section'
    var_2 = {var_0: var_1}
    var_3 = 'mysection'

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_setting'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'UnsupportedSettings'

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'example'
    var_2 = {var_0: var_1}
    var_3 = 'FormattingPluginDoesNotExist'

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 'SortingFunctionDoesNotExist'

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = 'tests'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'skip'
    var_1 = 'extend_skip'
    var_2 = 'skip1'
    var_3 = [var_2]
    var_4 = 'skip2'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'skip1'
    var_8 = 'skip2'

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = 'extend_skip_glob'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = '*.log'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = '*.txt'
    var_8 = '*.log'

def test_case_0():
    var_0 = 'py_version'
    var_1 = 'py310'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = 'profile'
    var_2 = True
    var_3 = 'black'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'quiet'
    var_1 = 'profile'
    var_2 = False
    var_3 = 'black'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = {var_0: var_5}



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.py_version
    assert var_4 == '310'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__get_config_data_toml. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_ini. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_editorconfig_indent_spaces. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_editorconfig_indent_tabs. Retrieved 5/14 statements.
# Partially parsed test__get_config_data_editorconfig_wildcard_extension. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_force_grid_wrap_numeric. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_true. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_false. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_comment_prefix. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_known_prefix. Retrieved 4/15 statements.
# Partially parsed test__get_config_data_bool_from_string. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_tuple. Retrieved 4/15 statements.
# Partially parsed test__get_config_data_frozenset. Retrieved 4/15 statements.
# Partially parsed test__get_config_data_empty_section. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_multiple_sections. Retrieved 4/13 statements.
# Partially parsed test__get_config_data_toml_nested. Retrieved 3/12 statements.
# Partially parsed test__get_config_data_editorconfig_skip_non_relevant. Retrieved 3/12 statements.


def test_case_0():
    var_0 = b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'root = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 100\n'
    var_1 = '*'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*]\nindent_style = tab\nindent_size = tab\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*{py,pyi}]\nindent_style = space\nindent_size = 4\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = 3\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = true\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\ncomment_prefix = "# "\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nextend-exclude = ./exclude_dir/\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'extend-exclude'

def test_case_0():
    var_0 = '[black]\nskip_string_normalization = yes\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nknown_third_party = requests,django\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'known_third_party'
    var_4 = 'requests'
    var_5 = 'django'

def test_case_0():
    var_0 = '[black]\nextend_ignore = E501,W503\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'extend_ignore'
    var_4 = 'E501'
    var_5 = 'W503'

def test_case_0():
    var_0 = '[black]\nline_length = 88\n\n[other]\nkey = value\n'
    var_1 = 'missing'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\n\n[pycodestyle]\nmax_line_length = 100\n'
    var_1 = 'black'
    var_2 = 'pycodestyle'
    var_3 = (var_1, var_2)

def test_case_0():
    var_0 = b'[tool.black.format]\nline_length = 88\n'
    var_1 = 'tool.black.format'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*]\nindent_style = space\nindent_size = 4\ncharset = utf-8\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'charset'
    var_4 = 'indent'



# Parsed testcases at query #17
#--------------------------




import isort.utils as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = {}
    var_5 = module_0.Trie(var_3, var_4)
    var_6 = 'some_path'
    var_7 = var_5.insert(var_6, var_2)
    var_8 = var_5.root.nodes
    var_9 = bool(var_5.root.nodes != {})
    assert var_9 is True



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_multi_line_output_vertical_grid_grouped_no_comma.




