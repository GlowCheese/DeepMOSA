####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2._known_patterns
    assert var_3 is None
    var_4 = var_2._section_comments
    assert var_4 is None
    var_5 = var_2._section_comments_end
    assert var_5 is None
    var_6 = var_2._skips
    assert var_6 is None
    var_7 = var_2._skip_globs
    assert var_7 is None
    var_8 = var_2._sorting_function
    assert var_8 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2._known_patterns
    assert var_3 is None
    var_4 = var_2._section_comments
    assert var_4 is None
    var_5 = var_2._section_comments_end
    assert var_5 is None
    var_6 = var_2._skips
    assert var_6 is None
    var_7 = var_2._skip_globs
    assert var_7 is None
    var_8 = var_2._sorting_function
    assert var_8 is None

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
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_editorconfig_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_other_file. Retrieved 5/6 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_digit. Retrieved 6/7 statements.
# Partially parsed test_get_config_data_known_prefix_paths. Retrieved 6/7 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 6/7 statements.
# Partially parsed test_get_config_data_type_conversion. Retrieved 4/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    assert var_7 == 'test.toml'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = '*.{js,ts}'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    assert var_7 == 'test.editorconfig'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    assert var_7 == 'test.ini'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = ()
    var_2 = module_0._get_config_data(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'indent'
    var_5 = bool('indent' in var_3)
    assert var_5 is True
    var_6 = var_3['indent']
    assert var_6 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'indent'
    var_5 = bool('indent' in var_3)
    assert var_5 is True
    var_6 = var_3['indent']
    assert var_6 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'line_length'
    var_5 = bool('line_length' in var_3)
    assert var_5 is True
    var_6 = 'inf'
    var_7 = float(var_6)
    var_8 = var_3['line_length']
    var_9 = bool(var_3['line_length'] == var_7)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'line_length'
    var_5 = bool('line_length' in var_3)
    assert var_5 is True
    var_6 = 'line_length'
    var_7 = var_3[var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'known_prefix_paths'
    var_5 = bool('known_prefix_paths' in var_3)
    assert var_5 is True
    var_6 = 'known_prefix_paths'
    var_7 = var_3[var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'force_grid_wrap'
    var_5 = bool('force_grid_wrap' in var_3)
    assert var_5 is True
    var_6 = var_3['force_grid_wrap']
    assert var_6 == 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'force_grid_wrap'
    var_5 = bool('force_grid_wrap' in var_3)
    assert var_5 is True
    var_6 = var_3['force_grid_wrap']
    assert var_6 == 0

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'comment_prefix'
    var_5 = bool('comment_prefix' in var_3)
    assert var_5 is True
    var_6 = 'comment_prefix'
    var_7 = var_3[var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = [var_0]
    var_5 = bool(var_1)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'valid_config_file.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2._config_settings
    var_4 = bool(var_2._config_settings == {})
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.min.js'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.py~'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/dev/null'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'script'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'empty_file'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__find_config_handles_exception_during_config_parsing. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/non/existent/path', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {'key': 'value'}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path/with/stop_dir'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path/with/stop_dir', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path/child'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {'key': 'value'}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {}))
    assert var_2 is True
    var_3 = 'Failed to pull configuration information from /some/path/pyproject.toml'
    var_4 = 2



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_import_heading_prefix_check. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'import_heading_prefix_test'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = 'other_value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'import_heading_prefix_test'
    var_6 = 'import_heading_'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ensure_deprecated_options_used_predicate. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'deprecated_option'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_all_configs_with_config_file. Retrieved 6/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/empty_directory'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info
    var_3 = bool(var_1.root.config_info == ('default', {}))
    assert var_3 is True
    var_4 = var_1.root.nodes
    var_5 = bool(var_1.root.nodes == {})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/config'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info
    var_3 = bool(var_1.root.config_info == ('default', {}))
    assert var_3 is True
    var_4 = '/path/to/config/.isort.cfg'
    var_5 = bool('/path/to/config/.isort.cfg' in var_1.root.nodes)
    assert var_5 is True
    var_6 = var_1.root.nodes['/path/to/config/.isort.cfg'].config_info[0]
    assert var_6 == '/path/to/config/.isort.cfg'
    var_7 = 1
    var_8 = '/path/to/config/.isort.cfg'
    var_9 = var_1.root.nodes[var_8]
    var_10 = var_9.config_info[var_7]

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/multiple_configs'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info
    var_3 = bool(var_1.root.config_info == ('default', {}))
    assert var_3 is True
    var_4 = '/path/to/multiple_configs/.isort.cfg'
    var_5 = bool('/path/to/multiple_configs/.isort.cfg' in var_1.root.nodes)
    assert var_5 is True
    var_6 = '/path/to/multiple_configs/setup.cfg'
    var_7 = bool('/path/to/multiple_configs/setup.cfg' in var_1.root.nodes)
    assert var_7 is True
    var_8 = var_1.root.nodes['/path/to/multiple_configs/.isort.cfg'].config_info[0]
    assert var_8 == '/path/to/multiple_configs/.isort.cfg'
    var_9 = var_1.root.nodes['/path/to/multiple_configs/setup.cfg'].config_info[0]
    assert var_9 == '/path/to/multiple_configs/setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/nested'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info
    var_3 = bool(var_1.root.config_info == ('default', {}))
    assert var_3 is True
    var_4 = '/path/to/nested/subdir/.isort.cfg'
    var_5 = bool('/path/to/nested/subdir/.isort.cfg' in var_1.root.nodes['/path/to/nested'].nodes)
    assert var_5 is True
    var_6 = var_1.root.nodes['/path/to/nested'].nodes['subdir'].nodes['.isort.cfg'].config_info[0]
    assert var_6 == '/path/to/nested/subdir/.isort.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/invalid_config'
    var_1 = module_0.find_all_configs(var_0)
    var_2 = var_1.root.config_info
    var_3 = bool(var_1.root.config_info == ('default', {}))
    assert var_3 is True
    var_4 = '/path/to/invalid_config/invalid.cfg'
    var_5 = bool('/path/to/invalid_config/invalid.cfg' not in var_1.root.nodes)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_config_data_with_toml_file. Retrieved 4/5 statements.
# Partially parsed test_get_config_data_with_editorconfig_file. Retrieved 4/5 statements.
# Partially parsed test_get_config_data_with_ini_file. Retrieved 4/5 statements.
# Partially parsed test_get_config_data_with_empty_sections. Retrieved 3/4 statements.
# Partially parsed test_get_config_data_with_nonexistent_file. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'source'
    var_5 = bool('source' in var_3)
    assert var_5 is True
    var_6 = var_3['source']
    var_7 = bool(var_3['source'] == var_0)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'source'
    var_5 = bool('source' in var_3)
    assert var_5 is True
    var_6 = var_3['source']
    var_7 = bool(var_3['source'] == var_0)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.ini'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'source'
    var_5 = bool('source' in var_3)
    assert var_5 is True
    var_6 = var_3['source']
    var_7 = bool(var_3['source'] == var_0)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = ()
    var_2 = module_0._get_config_data(var_0, var_1)
    var_3 = 'source'
    var_4 = bool('source' in var_2)
    assert var_4 is True
    var_5 = var_2['source']
    var_6 = bool(var_2['source'] == var_0)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = 'source'
    var_5 = bool('source' not in var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_supported_filetype_returns_true_for_fifo_file. Retrieved 5/9 statements.


import isort.settings as module_0
import posixpath as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file'
    var_3 = module_1.dirname(var_2)
    var_4 = True
    var_5 = var_1.is_supported_filetype(var_2)
    assert var_5 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_import_footer_prefix_condition. Retrieved 7/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = 'import_footer_test'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}
    var_5 = 'import_footer_test'
    var_6 = {var_5: var_3}
    var_7 = module_0.Config(var_0, var_0, var_1, **var_6)
    var_8 = 'test'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_config_constructor_with_config_and_overrides. Retrieved 5/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 120
    var_3 = 'line_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = var_5.line_length
    assert var_6 == 120
    var_7 = 'py'
    var_8 = ''
    var_9 = var_5.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == '.'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

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

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = '    '
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.line_length
    assert var_6 == 88
    var_7 = var_5.indent
    assert var_7 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.include_trailing_comma
    assert var_4 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'baz'
    var_2 = [var_0, var_1]
    var_3 = 'known_foo'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'foo'
    var_7 = [var_0, var_1]
    var_8 = frozenset(var_7)
    var_9 = {var_6: var_8}
    var_10 = var_5.known_other
    var_11 = bool(var_5.known_other == var_9)
    assert var_11 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = 'import_heading_foo'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings
    var_5 = bool(var_3.import_headings == {'foo': 'Bar'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Baz'
    var_1 = 'import_footer_foo'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers
    var_5 = bool(var_3.import_footers == {'foo': 'Baz'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatter
    assert var_4 == 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_0 = 'nonexistent'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/dir1'
    var_2 = 'dir2/'
    var_3 = 'dir3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = bool(var_5 == {'/dir1', '/home/user/dir2', '/home/user/dir3'})
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/dir1/'
    var_2 = '/dir2'
    var_3 = '/dir3/'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = bool(var_5 == {'/dir1/', '/dir2', '/dir3/'})
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir1'
    var_2 = '/dir2/'
    var_3 = 'dir3/'
    var_4 = '/dir4'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0._abspaths(var_0, var_5)
    var_7 = bool(var_6 == {'/home/user/dir1', '/dir2/', '/dir3/', '/dir4'})
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_166_predicate_false. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2._known_patterns
    assert var_3 is None
    var_4 = var_2._section_comments
    assert var_4 is None
    var_5 = var_2._section_comments_end
    assert var_5 is None
    var_6 = var_2._skips
    assert var_6 is None
    var_7 = var_2._skip_globs
    assert var_7 is None
    var_8 = var_2._sorting_function
    assert var_8 is None



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test___post_init___with_py_version_auto. Retrieved 3/4 statements.
# Failed to parse test___post_init___with_vertical_grid_grouped_no_comma.


import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '38'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py38'

import isort.settings as module_0

def test_case_0():
    var_0 = '38'
    var_1 = frozenset()
    var_2 = module_0._Config(var_0, known_standard_library=var_1)
    var_3 = var_2.known_standard_library
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

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
    var_0 = 80
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_config. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_deprecated_config. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_known_other. Retrieved 9/10 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 4/7 statements.
# Partially parsed test_config_constructor_with_skip. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_skip_glob. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_digit. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_as_spaces. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_patterns. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_directory. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_extend_skip. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_extend_skip_glob. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_skip_gitignore. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_patterns_directory. Retrieved 9/10 statements.
# Partially parsed test_config_constructor_with_known_patterns_wildcard. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_question_mark. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_mixed. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_known_patterns_empty. Retrieved 4/5 statements.
# Partially parsed test_config_constructor_with_known_patterns_none. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_patterns_invalid. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_known_patterns_duplicate. Retrieved 6/7 statements.
# Partially parsed test_config_constructor_with_known_patterns_case_sensitive. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_special_characters. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_unicode. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_whitespace. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_newline. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_tab. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_backslash. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_forward_slash. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_backtick. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_single_quote. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_known_patterns_double_quote. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 'indent'
    var_2 = 100
    var_3 = '    '
    var_4 = {var_0: var_2, var_1: var_3}

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == '.'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_other'
    var_1 = 'custom'
    var_2 = 'module'
    var_3 = {var_2}
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = {var_2}
    var_7 = frozenset(var_6)
    var_8 = {var_1: var_7}

def test_case_0():
    var_0 = 'import_heading_custom'
    var_1 = 'Custom Heading'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'import_footer_custom'
    var_1 = 'Custom Footer'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 'skip'
    var_1 = 'file.py'
    var_2 = {var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = '*.py'
    var_2 = {var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '    '
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'sections'
    var_1 = 'FUTURE'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = 'FIRSTPARTY'
    var_5 = 'LOCALFOLDER'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = {var_0: var_6}

def test_case_0():
    var_0 = 'directory'
    var_1 = '/path/to/project'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'extend_skip'
    var_1 = 'file.py'
    var_2 = {var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'extend_skip_glob'
    var_1 = '*.py'
    var_2 = {var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'skip_gitignore'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os/'
    var_2 = 'sys/'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = {var_5, var_6}
    var_8 = frozenset(var_7)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os.*'
    var_2 = 'sys.*'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os?'
    var_2 = 'sys?'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os'
    var_2 = 'sys.*'
    var_3 = 're?'
    var_4 = {var_1, var_2, var_3}
    var_5 = {var_0: var_4}
    var_6 = {var_1, var_2, var_3}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = frozenset()

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = None
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'invalid_pattern'
    var_2 = {var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os'
    var_2 = {var_1, var_1}
    var_3 = {var_0: var_2}
    var_4 = {var_1}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'OS'
    var_2 = 'os'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os-path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os-路径'
    var_2 = 'sys.路径'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os\npath'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os\tpath'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os\\path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os/path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os`path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = "os'path"
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'os"path'
    var_2 = 'sys.path'
    var_3 = {var_1, var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1, var_2}
    var_6 = frozenset(var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_initialization_with_deprecated_settings. Retrieved 5/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.settings_file
    assert var_2 == ''
    var_3 = var_1.settings_path
    assert var_3 == ''
    var_4 = var_1._known_patterns
    assert var_4 is None
    var_5 = var_1._section_comments
    assert var_5 is None
    var_6 = var_1._section_comments_end
    assert var_6 is None
    var_7 = var_1._skips
    assert var_7 is None
    var_8 = var_1._skip_globs
    assert var_8 is None
    var_9 = var_1._sorting_function
    assert var_9 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'
    var_4 = var_2.settings_path
    assert var_4 == ''
    var_5 = var_2._known_patterns
    assert var_5 is None
    var_6 = var_2._section_comments
    assert var_6 is None
    var_7 = var_2._section_comments_end
    assert var_7 is None
    var_8 = var_2._skips
    assert var_8 is None
    var_9 = var_2._skip_globs
    assert var_9 is None
    var_10 = var_2._sorting_function
    assert var_10 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == ''
    var_4 = var_2.settings_path
    assert var_4 == '.'
    var_5 = var_2._known_patterns
    assert var_5 is None
    var_6 = var_2._section_comments
    assert var_6 is None
    var_7 = var_2._section_comments_end
    assert var_7 is None
    var_8 = var_2._skips
    assert var_8 is None
    var_9 = var_2._skip_globs
    assert var_9 is None
    var_10 = var_2._sorting_function
    assert var_10 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = var_3.settings_file
    assert var_4 == ''
    var_5 = var_3.settings_path
    assert var_5 == ''
    var_6 = var_3._known_patterns
    assert var_6 is None
    var_7 = var_3._section_comments
    assert var_7 is None
    var_8 = var_3._section_comments_end
    assert var_8 is None
    var_9 = var_3._skips
    assert var_9 is None
    var_10 = var_3._skip_globs
    assert var_10 is None
    var_11 = var_3._sorting_function
    assert var_11 is None

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 120
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.settings_file
    assert var_6 == ''
    var_7 = var_5.settings_path
    assert var_7 == ''
    var_8 = var_5._known_patterns
    assert var_8 is None
    var_9 = var_5._section_comments
    assert var_9 is None
    var_10 = var_5._section_comments_end
    assert var_10 is None
    var_11 = var_5._skips
    assert var_11 is None
    var_12 = var_5._skip_globs
    assert var_12 is None
    var_13 = var_5._sorting_function
    assert var_13 is None
    var_14 = var_5.quiet
    assert var_14 is True
    var_15 = var_5.line_length
    assert var_15 == 120

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_setting'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = True
    var_2 = 'force_single_line'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = -1
    var_6 = -1
    var_7 = 'W0503: Deprecated config options were used: force_single_line.'

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'baz'
    var_2 = [var_0, var_1]
    var_3 = 'FOO'
    var_4 = [var_3]
    var_5 = 'known_foo'
    var_6 = 'sections'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = 'foo'
    var_10 = [var_0, var_1]
    var_11 = frozenset(var_10)
    var_12 = {var_9: var_11}
    var_13 = var_8.known_other
    var_14 = bool(var_8.known_other == var_12)
    assert var_14 is True
    var_15 = 'FOO'
    var_16 = bool('FOO' in var_8.sections)
    assert var_16 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = 'Qux'
    var_2 = 'import_heading_foo'
    var_3 = 'import_heading_baz'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.import_headings
    var_7 = bool(var_5.import_headings == {'foo': 'Bar', 'baz': 'Qux'})
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = 'Qux'
    var_2 = 'import_footer_foo'
    var_3 = 'import_footer_baz'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.import_footers
    var_7 = bool(var_5.import_footers == {'foo': 'Bar', 'baz': 'Qux'})
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

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
    var_0 = 'black'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(var_3.formatting_function is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_0 = 'invalid'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'STANDARD_LIBRARY'
    var_1 = 'THIRD_PARTY'
    var_2 = (var_0, var_1)
    var_3 = 'sections'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'STANDARD_LIBRARY'
    var_7 = 'THIRD_PARTY'



# Parsed testcases at query #24
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'a, b, c'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a\nb\nc'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a, b\nc, d'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c', 'd'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a, , b, , c'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '  a  ,  b  ,  c  '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = ' b '
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ',,,'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_when_file_in_skips. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_when_file_not_in_skips. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_when_parent_dir_in_skips. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_when_file_matches_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_when_file_does_not_match_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_when_file_is_not_regular_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_when_skip_gitignore_and_file_not_in_git. Retrieved 3/7 statements.
# Partially parsed test_is_skipped_when_skip_gitignore_and_file_in_git. Retrieved 3/7 statements.
# Partially parsed test_is_skipped_when_file_is_git_directory. Retrieved 3/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = {var_0}
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = 'other.py'
    var_1 = {var_0}
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = 'tests'
    var_1 = {var_0}
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'tests/test.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_*.py'
    var_1 = {var_0}
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_example.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_*.py'
    var_1 = {var_0}
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'example.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/git/repo/untracked.py'
    var_5 = [var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/git/repo/tracked.py'
    var_5 = [var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '.git'
    var_5 = [var_4]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_init_with_config_object. Retrieved 4/5 statements.
# Partially parsed test_config_init_with_deprecated_options. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = 'py'
    var_4 = ''
    var_5 = var_2.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.line_length
    assert var_3 == 120

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = 'bar'
    var_1 = [var_0]
    var_2 = 'known_foo'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = [var_0]
    var_7 = frozenset(var_6)
    var_8 = {var_5: var_7}
    var_9 = var_4.known_other
    var_10 = bool(var_4.known_other == var_8)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar'
    var_1 = 'import_heading_foo'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings
    var_5 = bool(var_3.import_headings == {'foo': 'Bar'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'old_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'old_option'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___post_init___auto_py_version. Retrieved 3/4 statements.
# Failed to parse test___post_init___multi_line_output.


import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = '38'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py38'

import isort.settings as module_0

def test_case_0():
    var_0 = '38'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0._Config(force_alphabetical_sort=var_0)
    var_2 = bool(var_1.force_alphabetical_sort_within_sections)
    assert var_2 is True
    var_3 = bool(var_1.no_sections)
    assert var_3 is True
    var_4 = var_1.lines_between_types
    assert var_4 == 1
    var_5 = bool(var_1.from_first)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.exe'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.py~'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/dev/stdin'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__get_config_data_with_toml_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_with_editorconfig_file. Retrieved 5/6 statements.
# Partially parsed test__get_config_data_with_ini_file. Retrieved 5/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    var_8 = bool(var_4['source'] == var_0)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.editorconfig'
    var_1 = '*.py'
    var_2 = '*.js'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    var_8 = bool(var_4['source'] == var_0)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.ini'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = 'source'
    var_6 = bool('source' in var_4)
    assert var_6 is True
    var_7 = var_4['source']
    var_8 = bool(var_4['source'] == var_0)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'empty_config.toml'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_config.toml'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent.toml'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_editorconfig_file_path_ends_with_editorconfig. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '.editorconfig'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_78. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'known_'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_unsupported_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_known_patterns. Retrieved 7/8 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 120
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == '.'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'profile'
    var_1 = 'invalid_profile'
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'invalid_formatter'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_third_party'
    var_1 = 'numpy'
    var_2 = 'pandas'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'import_heading_stdlib'
    var_1 = 'Standard Library'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'import_footer_stdlib'
    var_1 = 'End of Standard Library'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'invalid_sort_order'
    var_2 = {var_0: var_1}



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/non/existent/path', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {'key': 'value'}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some', {'key': 'value'}))
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/some/path', {}))
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__get_str_to_type_converter_with_int_setting. Retrieved 2/3 statements.
# Partially parsed test__get_str_to_type_converter_with_float_setting. Retrieved 2/3 statements.
# Partially parsed test__get_str_to_type_converter_with_bool_setting. Retrieved 2/3 statements.
# Partially parsed test__get_str_to_type_converter_with_wrap_modes. Retrieved 2/3 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_setting'
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_123_evaluates_to_false. Retrieved 16/20 statements.


def test_case_0():
    var_0 = 'sections'
    var_1 = 'known_other'
    var_2 = 'STANDARD_LIB'
    var_3 = 'THIRD_PARTY'
    var_4 = (var_2, var_3)
    var_5 = 'custom'
    var_6 = 'module1'
    var_7 = 'module2'
    var_8 = [var_6, var_7]
    var_9 = frozenset(var_8)
    var_10 = {var_5: var_9}
    var_11 = {var_0: var_4, var_1: var_10}
    var_12 = 'known_custom'
    var_13 = 'CUSTOM'
    var_14 = True
    var_15 = ()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 4/7 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_config. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_sort_order. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 120
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == '.'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '    '
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'foo'
    var_5 = [var_1]
    var_6 = frozenset(var_5)
    var_7 = {var_4: var_6}

def test_case_0():
    var_0 = 'import_heading_foo'
    var_1 = 'Bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'import_footer_foo'
    var_1 = 'Bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'venv'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_config_init_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_invalid_profile. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_quiet_mode. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_tab_indent. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_custom_indent. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_init_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_deprecated_options. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_unsupported_options. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_formatter_plugin. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_invalid_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_invalid_sort_order. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_src_paths. Retrieved 4/8 statements.
# Partially parsed test_config_init_with_skip_gitignore. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 120
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'setup.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == '.'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'profile'
    var_1 = 'invalid_profile'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = '    '
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'known_foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'foo'
    var_5 = [var_1]
    var_6 = frozenset(var_5)
    var_7 = {var_4: var_6}

def test_case_0():
    var_0 = 'import_heading_foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'import_footer_foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'virtual_env'

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'invalid_formatter'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'invalid_sort'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 'skip_gitignore'
    var_1 = True
    var_2 = {var_0: var_1}



