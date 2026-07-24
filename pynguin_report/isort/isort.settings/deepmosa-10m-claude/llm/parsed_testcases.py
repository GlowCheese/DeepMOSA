####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_constructor_initializes_git_ls_files. Retrieved 4/5 statements.


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
    var_0 = '/nonexistent/path/to/settings'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

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
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

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

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = 3
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_mode'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.line_length
    assert var_8 == 88
    var_9 = var_7.use_parentheses
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'git_ls_files'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.git_ls_files

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 88
    var_5 = 'line_length'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(config=var_3, **var_6)
    var_8 = var_7.line_length
    assert var_8 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.sources
    var_3 = bool(var_1.sources is not None)
    assert var_3 is True
    var_4 = var_1.sources
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/10 statements.
# Partially parsed test_get_config_data_toml_nested_section. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_multiple_sections. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_frozenset_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_bool_conversion_string. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_bool_false_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_int. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_double_quotes. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_space_indent. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_tab_indent. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_tab_width. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_nonexistent_section. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 88\nprofile = "black"\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 100\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 79\nprofile = django\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 88\n[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'other'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_first_party = module1,module2\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = file1.py,file2.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'
    var_5 = 'file1.py'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\nuse_parentheses = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nuse_parentheses = false\n'
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
    var_1 = '[*]\nindent_style = space\nindent_size = tab\ntab_width = 4\n'
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
    var_0 = '.editorconfig'
    var_1 = '[*]\nmax_line_length = 120\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = (var_2,)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/6 statements.
# Partially parsed test_config_init_with_known_other_sections. Retrieved 4/5 statements.
# Partially parsed test_config_init_with_import_headings. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_import_footers. Retrieved 3/5 statements.
# Failed to parse test_config_init_with_src_paths.
# Failed to parse test_config_init_with_directory.


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
    var_1 = 88
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 88

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=100\n'

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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

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
    var_0 = 'Future imports'
    var_1 = 'Standard library'
    var_2 = 'import_heading_future'
    var_3 = 'import_heading_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'Future imports'
    var_8 = 'Standard library'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = 'import_footer_future'
    var_3 = 'import_footer_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'End future'
    var_8 = 'End stdlib'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1.src_paths
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 3
    var_2 = 'migrations'
    var_3 = [var_2]
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = 'skip'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True
    var_10 = var_8.line_length
    assert var_10 == 100
    var_11 = var_8.multi_line_mode
    assert var_11 == 3

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
    var_9 = bool(var_8 is not None)
    assert var_9 is True
    var_10 = 'FUTURE'
    var_11 = bool('FUTURE' in var_8.sections)
    assert var_11 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_is_supported_filetype_with_py_extension. Retrieved 3/4 statements.
# Partially parsed test_config_is_skipped_with_skip_list. Retrieved 3/7 statements.
# Partially parsed test_config_known_patterns_property. Retrieved 3/6 statements.
# Partially parsed test_config_section_comments_property. Retrieved 5/6 statements.
# Partially parsed test_config_section_comments_end_property. Retrieved 5/6 statements.
# Partially parsed test_config_skips_property. Retrieved 6/7 statements.
# Partially parsed test_config_skip_globs_property. Retrieved 6/7 statements.
# Partially parsed test_config_parse_known_pattern. Retrieved 2/4 statements.


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
    var_7 = var_5.line_length
    assert var_7 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

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
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)

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
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.known_patterns
    var_3 = 2

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
    var_0 = 'FUTURE'
    var_1 = 'End of future imports'
    var_2 = {var_0: var_1}
    var_3 = 'import_footers'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = 'another.py'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skips
    var_9 = 'test.py'
    var_10 = bool('test.py' in var_8)
    assert var_10 is True
    var_11 = 'another.py'
    var_12 = bool('another.py' in var_8)
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = '__pycache__/*'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skip_globs
    var_9 = '*.pyc'
    var_10 = bool('*.pyc' in var_8)
    assert var_10 is True
    var_11 = '__pycache__/*'
    var_12 = bool('__pycache__/*' in var_8)
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'mymodule'
    var_3 = 'mymodule'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_constructor_sets_directory. Retrieved 2/3 statements.


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
    var_6 = var_3._section_comments
    assert var_6 is None

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.line_length
    assert var_7 == 88

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
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True
    var_4 = var_1.directory

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
    var_0 = 'src'
    var_1 = 'lib'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths
    var_7 = bool(var_5.src_paths is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'sources'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'migrations'
    var_3 = [var_2]
    var_4 = 'quiet'
    var_5 = 'line_length'
    var_6 = 'skip'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = var_8.quiet
    assert var_9 is True
    var_10 = var_8.line_length
    assert var_10 == 100



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_172_predicate_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '/test/glob/result'
    var_1 = [var_0]
    var_2 = 'src/**/test'
    var_3 = '*'
    var_4 = str(var_2)
    var_5 = var_3 in var_4
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_98_evaluates_to_false. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 98 evaluates to False for known_standard_library.'
    var_1 = 'known_standard_library'
    var_2 = 'known_'
    var_3 = bool(var_1 in ('known_standard_library', 'known_future_library', 'known_third_party', 'known_first_party', 'known_local_folder'))
    assert var_3 is True
    var_4 = 'known_standard_library'
    var_5 = 'known_future_library'
    var_6 = 'known_third_party'
    var_7 = 'known_first_party'
    var_8 = 'known_local_folder'
    var_9 = (var_4, var_5, var_6, var_7, var_8)
    var_10 = var_1 not in var_9



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_post_init_auto_py_version. Retrieved 3/4 statements.
# Partially parsed test_config_post_init_vertical_grid_grouped_no_comma. Retrieved 1/3 statements.


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
    var_0 = '3.8'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = (var_0, var_1)
    var_3 = frozenset(var_2)
    var_4 = '3.8'
    var_5 = module_0._Config(var_4, known_standard_library=var_3)
    var_6 = var_5.known_standard_library
    var_7 = bool(var_5.known_standard_library == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = '3.8'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
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
    var_0 = '3.8'
    var_1 = 79
    var_2 = 100
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 79
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 79
    var_4 = var_2.line_length
    assert var_4 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 100
    var_2 = 80
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = var_3.wrap_length
    assert var_4 == 80
    var_5 = var_3.line_length
    assert var_5 == 100



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_config_init_creates_src_paths. Retrieved 4/5 statements.
# Partially parsed test_config_init_sets_directory. Retrieved 4/5 statements.


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
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5._known_patterns
    assert var_6 is None
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 100

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
    var_0 = ''
    var_1 = True
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)
    var_5 = var_4._known_patterns
    assert var_5 is None
    var_6 = var_4.directory
    var_7 = bool(var_4.directory == var_4.directory)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = False
    var_7 = 'quiet'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(config=var_5, **var_8)
    var_10 = var_9.quiet
    assert var_10 is False
    var_11 = var_9.line_length
    assert var_11 == 80

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
    var_2 = 'src_paths'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.src_paths

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'directory'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.directory

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'black'
    var_3 = 3
    var_4 = 'quiet'
    var_5 = 'line_length'
    var_6 = 'profile'
    var_7 = 'multi_line_mode'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = var_9.quiet
    assert var_10 is True
    var_11 = var_9.line_length
    assert var_11 == 100
    var_12 = var_9.multi_line_mode
    assert var_12 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'git_ls_files'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 120
    var_7 = 'line_length'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(config=var_5, **var_8)
    var_10 = var_9.line_length
    assert var_10 == 120
    var_11 = var_9.quiet
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #11
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
    var_7 = 'some_other_setting'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_config_constructor_with_settings_path.


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
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 100
    var_7 = 'line_length'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(config=var_5, **var_8)
    var_10 = var_9.line_length
    assert var_10 == 100
    var_11 = var_9.quiet
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_supported_filetype. Retrieved 15/30 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'test.pyc'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = 'test.py~'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = '/nonexistent/path/file.txt'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    assert var_9 is True
    var_10 = b'#!/usr/bin/env python\nimport os'
    var_11 = b'import os\nimport sys'
    var_12 = 'test.pyi'
    var_13 = var_1.is_supported_filetype(var_12)
    assert var_13 is True
    var_14 = 'test.pyx'
    var_15 = var_1.is_supported_filetype(var_14)
    assert var_15 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_line_66_predicate_evaluates_to_true. Retrieved 5/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_key'
    var_1 = 'test_value'
    var_2 = 'test_profile'
    var_3 = 'profile'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'isort.profiles'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_known_section_mapping_predicate_true. Retrieved 16/22 statements.


def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = 'FUTURE'
    var_2 = 'third_party'
    var_3 = 'future_library'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'known_thirdparty'
    var_6 = 'some_other_key'
    var_7 = 'requests'
    var_8 = 'django'
    var_9 = [var_7, var_8]
    var_10 = 'value'
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = 'known_'
    var_13 = 'known_thirdparty'
    var_14 = len(var_12)
    var_15 = var_13[var_14:]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_config_init_with_non_none_config_parameter. Retrieved 12/24 statements.


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



# Parsed testcases at query #17
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
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_config_init_with_settings_path.
# Partially parsed test_config_init_src_paths_custom. Retrieved 1/8 statements.


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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'THIRDPARTY'
    var_5 = 'DJANGO'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'known_django'
    var_10 = 'sections'
    var_11 = {var_9: var_1, var_10: var_8}
    var_12 = module_0.Config(**var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'Standard library'
    var_2 = 'import_heading_future'
    var_3 = 'import_heading_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = 'import_footer_future'
    var_3 = 'import_footer_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

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

def test_case_0():
    var_0 = 'src'

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
    var_0 = '/tmp'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    assert var_4 == '/tmp'

import isort.settings as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = 'os'
    var_2 = [var_0, var_1]
    var_3 = 'known_standard_library'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'known_first_party'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'requests'
    var_1 = [var_0]
    var_2 = 'known_third_party'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.skip_gitignore
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_profile_name_not_in_profiles_triggers_entry_points_loop. Retrieved 7/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = 'profile'
    var_3 = 'black'
    var_4 = {var_2: var_3}
    var_5 = 'profile'
    var_6 = {var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'isort.profiles'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 3/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 3/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #22
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
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'override_value'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_73_evaluates_to_true. Retrieved 12/25 statements.


def test_case_0():
    var_0 = False
    var_1 = 'tool:isort'
    var_2 = 'profile'
    var_3 = 'black'
    var_4 = '[tool:isort]\nprofile = black\n'
    var_5 = 'source'
    var_6 = 'some_bool_key'
    var_7 = 'test.cfg'
    var_8 = 'true'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'some_bool_key'
    var_11 = 'true'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_boolean_conversion. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_force_grid_wrap_numeric. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_space. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_tab. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_numeric. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 5/12 statements.
# Partially parsed test_get_config_data_multiline_list. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 3/10 statements.
# Partially parsed test_get_config_data_multiple_sections. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '[tool.isort]\nline_length = 88\nskip = "migrations"\n'
    var_1 = 'tool.isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nline_length = 100\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nprofile = black\nverbose = true\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nforce_grid_wrap = 2\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nforce_grid_wrap = false\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nforce_grid_wrap = true\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\ncomment_prefix = "# "\n'
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_1 = '[*.py]'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_1 = '[*.py]'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nmax_line_length = 100\n'
    var_1 = '[*.py]'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nmax_line_length = off\n'
    var_1 = '[*.py]'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[isort]\nskip = file1.py,\n    file2.py,\n    file3.py\n'
    var_1 = 'isort'
    var_2 = (var_1,)
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = 'file3.py'

def test_case_0():
    var_0 = ''
    var_1 = 'isort'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[isort]\nline_length = 88\n[other]\nkey = value\n'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py310'
    var_8 = None
    var_9 = {var_0: var_7, var_1: var_8, var_2: var_8, var_3: var_8, var_4: var_8, var_5: var_8, var_6: var_8}



# Parsed testcases at query #26
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
    var_10 = 'YES'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is True
    var_12 = 'on'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is True
    var_14 = 'ON'
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
    var_6 = '0'
    var_7 = module_0._as_bool(var_6)
    assert var_7 is False
    var_8 = 'no'
    var_9 = module_0._as_bool(var_8)
    assert var_9 is False
    var_10 = 'NO'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is False
    var_12 = 'off'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is False
    var_14 = 'OFF'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value invalid'

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value'

import isort.settings as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0._as_bool(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'invalid truth value'

import isort.settings as module_0

def test_case_0():
    var_0 = 'tRuE'
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_find_config_returns_path_and_empty_dict_when_no_config_found. Retrieved 6/12 statements.
# Partially parsed test_find_config_returns_config_when_found. Retrieved 11/21 statements.
# Partially parsed test_find_config_stops_at_marker_directory. Retrieved 4/15 statements.
# Partially parsed test_find_config_searches_parent_directories. Retrieved 9/22 statements.
# Partially parsed test_find_config_handles_exception_during_config_read. Retrieved 9/23 statements.
# Partially parsed test_find_config_respects_max_search_depth. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'os.path.isfile'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = 'os.path.isdir'
    var_4 = lambda x: var_1
    var_5 = {}

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'os.path.isfile'
    var_3 = 'os.path.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = '_get_config_data'
    var_7 = 'line_length'
    var_8 = 88
    var_9 = {var_7: var_8}
    var_10 = lambda x, y: var_9

def test_case_0():
    var_0 = '.git'
    var_1 = 'os.path.isfile'
    var_2 = 'os.path.isdir'
    var_3 = {}

def test_case_0():
    var_0 = 'child'
    var_1 = 'setup.cfg'
    var_2 = 'os.path.isfile'
    var_3 = 'os.path.isdir'
    var_4 = '_get_config_data'
    var_5 = 'line_length'
    var_6 = 80
    var_7 = {var_5: var_6}
    var_8 = lambda x, y: var_7

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = 'os.path.isfile'
    var_2 = 'os.path.isdir'
    var_3 = '_get_config_data'
    var_4 = 'Parse error'
    var_5 = [var_4]
    var_6 = 'warn'
    var_7 = None
    var_8 = lambda x, stacklevel: var_7
    var_9 = {}

def test_case_0():
    var_0 = 'os.path.isfile'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = 'os.path.isdir'
    var_4 = lambda x: var_1
    var_5 = 'MAX_CONFIG_SEARCH_DEPTH'
    var_6 = 1



# Parsed testcases at query #28
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
    var_7 = 'some_other_field'
    var_8 = 'py311'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_init_sets_directory_to_current_working_dir. Retrieved 1/3 statements.
# Partially parsed test_config_init_preserves_src_paths. Retrieved 4/7 statements.
# Partially parsed test_config_init_known_patterns_lazy_loads. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_lazy_loads. Retrieved 2/3 statements.
# Partially parsed test_config_init_skips_lazy_loads. Retrieved 2/3 statements.
# Partially parsed test_config_init_skip_globs_lazy_loads. Retrieved 2/3 statements.


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
    var_6 = var_3._section_comments
    assert var_6 is None

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory

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
    var_7 = var_1.src_paths

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

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
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
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._sorting_function
    assert var_4 is None
    var_5 = var_3.sorting_function
    var_6 = var_3._sorting_function
    var_7 = bool(var_3._sorting_function is not None)
    assert var_7 is True
    var_8 = var_3._sorting_function



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_indent_lower_equals_tab_predicate_evaluates_to_false. Retrieved 3/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'spaces'
    var_2 = 'indent'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(settings_path=var_0, **var_3)
    var_5 = var_4.indent
    assert var_5 == 'spaces'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_line_43_evaluates_to_true. Retrieved 5/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/test/path'
    var_2 = '/test/path/setup.cfg'
    var_3 = False
    var_4 = 'quiet'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(var_2, **var_5)
    var_7 = 'A custom settings file was specified'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_str_to_type_converter_returns_int_type. Retrieved 2/6 statements.
# Partially parsed test_get_str_to_type_converter_returns_bool_type. Retrieved 2/6 statements.
# Partially parsed test_get_str_to_type_converter_returns_float_type. Retrieved 2/6 statements.
# Partially parsed test_get_str_to_type_converter_returns_wrap_mode_converter. Retrieved 2/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_setting'
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
    var_0 = 'test_float_setting'
    var_1 = module_0._get_str_to_type_converter(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_wrap_mode'
    var_1 = module_0._get_str_to_type_converter(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_editorconfig. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_editorconfig_tab. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/13 statements.
# Partially parsed test_get_config_data_bool_conversion. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/14 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_nonexistent_section. Retrieved 4/11 statements.
# Partially parsed test_get_config_data_editorconfig_indent_size_tab. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 88\nprofile = "black"\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 88\nprofile = black\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = 'root = true\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
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
    var_1 = '[isort]\nskip_glob = *.pyx\nuse_parentheses = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\ndefault_sections = FUTURE,STDLIB,THIRDPARTY,FIRSTPARTY,LOCALFOLDER\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'default_sections'

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
    var_1 = '[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = tab\ntab_width = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_post_init_py_version_auto. Retrieved 3/4 statements.
# Partially parsed test_post_init_multi_line_output_vertical_grid_grouped_no_comma. Retrieved 1/3 statements.


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
    var_0 = '2.7'
    var_1 = module_0._Config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'python version'
    var_4 = bool('python version' in str(e).lower())
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
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
    var_3 = '3.8'
    var_4 = module_0._Config(var_3, known_standard_library=var_2)
    var_5 = var_4.known_standard_library
    var_6 = bool(var_4.known_standard_library == var_2)
    assert var_6 is True

def test_case_0():
    var_0 = '3.8'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
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
    var_0 = '3.8'
    var_1 = 100
    var_2 = 79
    var_3 = module_0._Config(var_0, line_length=var_2, wrap_length=var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length'
    var_6 = bool('wrap_length' in str(e).lower())
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 79
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 79
    var_4 = var_2.line_length
    assert var_4 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 50
    var_2 = 79
    var_3 = module_0._Config(var_0, line_length=var_2, wrap_length=var_1)
    var_4 = var_3.wrap_length
    assert var_4 == 50



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_config_init_initializes_git_ls_files_cache. Retrieved 4/5 statements.
# Partially parsed test_config_init_creates_sources_tuple. Retrieved 3/4 statements.


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
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.quiet
    assert var_5 is True

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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End stdlib'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/src'
    var_1 = '/lib'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

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
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'git_ls_files'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.git_ls_files

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = 'black'
    var_3 = 'quiet'
    var_4 = 'line_length'
    var_5 = 'profile'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.quiet
    assert var_8 is True
    var_9 = var_7.line_length
    assert var_9 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 100
    var_5 = 'line_length'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(config=var_3, **var_6)
    var_8 = var_7.line_length
    assert var_8 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = '39'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_lib'
    var_1 = [var_0]
    var_2 = 'known_custom'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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
    var_0 = '*.py'
    var_1 = [var_0]
    var_2 = 'test_*.py'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '**/tests/**'
    var_1 = [var_0]
    var_2 = '**/venv/**'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

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
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sources
    var_5 = bool(var_3.sources is not None)
    assert var_5 is True
    var_6 = var_3.sources

import isort.settings as module_0

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = [var_0, var_1]
    var_3 = 'supported_extensions'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyx'
    var_1 = [var_0]
    var_2 = 'blocked_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_all_configs. Retrieved 6/20 statements.
# Partially parsed test_find_all_configs_empty_directory. Retrieved 1/6 statements.
# Partially parsed test_find_all_configs_with_pyproject_toml. Retrieved 3/10 statements.
# Partially parsed test_find_all_configs_nested_directories. Retrieved 5/18 statements.
# Partially parsed test_find_all_configs_invalid_config_file. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'project'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=100\n'
    var_3 = 'subdir'
    var_4 = 'setup.cfg'
    var_5 = '[isort]\nline_length=80\n'

def test_case_0():
    var_0 = 'empty'

def test_case_0():
    var_0 = 'project'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.isort]\nline_length = 120\n'

def test_case_0():
    var_0 = 'root'
    var_1 = 'level1'
    var_2 = 'level2'
    var_3 = '.isort.cfg'
    var_4 = '[settings]\nprofile=black\n'

def test_case_0():
    var_0 = 'project'
    var_1 = '.isort.cfg'
    var_2 = '[invalid\nbroken config'



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = None
    assert var_0 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_post_init_py_version_auto. Retrieved 3/4 statements.
# Partially parsed test_config_post_init_vertical_grid_grouped_no_comma_conversion. Retrieved 1/3 statements.


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
    var_0 = '3.9'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = var_1.known_standard_library
    var_6 = len(var_5)
    var_7 = 0
    var_8 = var_6 > var_7
    var_9 = bool('sys' in var_1.known_standard_library or var_8)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_module'
    var_1 = (var_0,)
    var_2 = frozenset(var_1)
    var_3 = '3.9'
    var_4 = module_0._Config(var_3, known_standard_library=var_2)
    var_5 = var_4.known_standard_library
    var_6 = bool(var_4.known_standard_library == var_2)
    assert var_6 is True

def test_case_0():
    var_0 = '3.9'

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
    var_1 = 80
    var_2 = 100
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length must be set lower than or equal to line_length'

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = 80
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 80
    var_4 = var_2.line_length
    assert var_4 == 80

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = 100
    var_2 = 80
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = var_3.wrap_length
    assert var_4 == 80
    var_5 = var_3.line_length
    assert var_5 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = module_0._Config(var_0)
    var_2 = module_0._Config(var_0)
    var_3 = hash(var_1)
    var_4 = id(var_1)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True
    var_6 = hash(var_2)
    var_7 = id(var_2)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True
    var_9 = hash(var_1)
    var_10 = hash(var_2)
    var_11 = bool(var_9 != var_10)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 4/10 statements.
# Partially parsed test_get_config_data_ini_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_spaces. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_tabs. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_bool_value. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_bool_value_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_numeric. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_double_quote. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_known_prefix_with_paths. Retrieved 6/14 statements.
# Partially parsed test_get_config_data_multiline_list. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_section_not_found. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_wildcard_extension. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_filters_unknown_keys. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_default_indent_size. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 88\nskip = ["file1.py", "file2.py"]\n'
    var_2 = 'tool.isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 100\nknown_django = django\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 2\nmax_line_length = 120\n'
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
    var_1 = '[isort]\nprofile = black\nuse_parentheses = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nuse_parentheses = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_grid_wrap = 3\n'
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
    var_0 = 'config_dir'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nknown_mylib = /absolute/path, relative/path\n'
    var_3 = 'isort'
    var_4 = (var_3,)
    var_5 = '/absolute/path'
    var_6 = 'relative/path'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = file1.py\n file2.py\n file3.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyi}]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nunknown_key = value\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'unknown_key'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\n'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_str_to_type_converter_wrap_modes. Retrieved 4/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'wrap'
    var_1 = 'nowrap'
    var_2 = 'test_setting'
    var_3 = module_0._get_str_to_type_converter(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_skipped_with_file_in_skips. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_file_not_in_skips. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_directory_in_skips. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_glob_pattern. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_glob_pattern_not_matching. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_nonexistent_path. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_extend_skip. Retrieved 7/9 statements.
# Partially parsed test_is_skipped_with_extend_skip_glob. Retrieved 8/10 statements.
# Partially parsed test_is_skipped_with_directory_set. Retrieved 6/8 statements.
# Partially parsed test_is_skipped_normalized_path_windows_style. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'file.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'skip_dir/file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'file.pyc'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '/nonexistent/path/to/file.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'file2.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'skip'
    var_7 = 'extend_skip'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = [var_3]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '*.pyo'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'skip_glob'
    var_7 = 'extend_skip_glob'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = 'file.pyo'
    var_11 = [var_10]

import isort.settings as module_0

def test_case_0():
    var_0 = 'testfile.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '/tmp'
    var_4 = 'skip'
    var_5 = 'directory'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '/tmp/testfile.py'
    var_9 = [var_8]

import isort.settings as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_line_123_predicate_evaluates_to_true. Retrieved 19/28 statements.


def test_case_0():
    var_0 = 'known_custom'
    var_1 = 'sections'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = (var_5, var_6, var_7, var_8, var_9)
    var_11 = {var_0: var_4, var_1: var_10}
    var_12 = 'known_custom'
    var_13 = [var_2, var_3]
    var_14 = 'known_'
    var_15 = len(var_14)
    var_16 = var_12[var_15:]
    var_17 = False
    var_18 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_line_165_predicate_evaluates_to_false. Retrieved 2/30 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_supported_filetype. Retrieved 12/33 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'test.pyc'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = 'test.py~'
    var_7 = var_1.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = '/nonexistent/path/file.py'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False
    var_10 = 'import os\n'
    var_11 = '#!/usr/bin/env python\n'
    var_12 = 'some random text\n'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_skipped_predicate_line_3_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_config_returns_tuple_with_path_and_dict. Retrieved 7/10 statements.
# Partially parsed test_find_config_with_nonexistent_path. Retrieved 2/3 statements.
# Failed to parse test_find_config_returns_original_path_when_no_config_found.
# Partially parsed test_find_config_finds_config_file. Retrieved 7/21 statements.
# Partially parsed test_find_config_respects_max_search_depth. Retrieved 3/12 statements.
# Partially parsed test_find_config_handles_exception_gracefully. Retrieved 4/16 statements.
# Partially parsed test_find_config_stops_at_stop_dir. Retrieved 3/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0._find_config(var_0)
    var_2 = len(var_1)
    assert var_2 == 2
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = module_0._find_config(var_0)
    var_2 = var_1[1]
    var_3 = bool(var_1[1] == {})
    assert var_3 is True

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'os.path.isfile'
    var_3 = 'os.path.isdir'
    var_4 = False
    var_5 = lambda path: var_4
    var_6 = 1

def test_case_0():
    var_0 = 0
    var_1 = 'os.path.isfile'
    var_2 = 'os.path.isdir'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = 'os.path.isfile'
    var_2 = 'os.path.isdir'
    var_3 = '_get_config_data'

def test_case_0():
    var_0 = '.git'
    var_1 = 'os.path.isfile'
    var_2 = 'os.path.isdir'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 9/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'txt'
    var_3 = 'log'
    var_4 = [var_2, var_3]
    var_5 = 'py'
    var_6 = 'js'
    var_7 = [var_5, var_6]
    var_8 = 'file.txt'
    var_9 = var_1.is_supported_filetype(var_8)
    assert var_9 is False



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_config_init_with_settings_path.
# Failed to parse test_config_init_sets_src_paths.
# Partially parsed test_config_init_with_known_prefix. Retrieved 4/5 statements.


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
    var_1 = 88
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 88

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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = 'django'

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'future'
    var_5 = bool('future' in var_3.import_headings)
    assert var_5 is True
    var_6 = var_3.import_headings['future']
    assert var_6 == 'Future imports'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of future imports'
    var_1 = 'import_footer_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'future'
    var_5 = bool('future' in var_3.import_footers)
    assert var_5 is True
    var_6 = var_3.import_footers['future']
    assert var_6 == 'End of future imports'

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 3
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'multi_line_mode'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.line_length
    assert var_8 == 100
    var_9 = var_7.multi_line_mode
    assert var_9 == 3
    var_10 = var_7.quiet
    assert var_10 is False

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
    var_0 = 'Future'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.section_comments
    var_5 = '# Future'
    var_6 = bool('# Future' in var_4)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End'
    var_1 = 'import_footer_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.section_comments_end
    var_5 = '# End'
    var_6 = bool('# End' in var_4)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = 'build'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skips
    var_9 = '__pycache__'
    var_10 = bool('__pycache__' in var_8)
    assert var_10 is True
    var_11 = 'build'
    var_12 = bool('build' in var_8)
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.egg-info'
    var_1 = [var_0]
    var_2 = 'dist'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skip_globs
    var_9 = '*.egg-info'
    var_10 = bool('*.egg-info' in var_8)
    assert var_10 is True
    var_11 = 'dist'
    var_12 = bool('dist' in var_8)
    assert var_12 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_formatter_in_combined_config_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = var_0 in var_2
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 23/51 statements.


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
    var_12 = 'builtins.vars'
    var_13 = {}
    var_14 = 'py_version'
    var_15 = 'py'
    var_16 = ''
    var_17 = '_known_patterns'
    var_18 = '_section_comments'
    var_19 = '_section_comments_end'
    var_20 = '_skips'
    var_21 = '_skip_globs'
    var_22 = '_sorting_function'
    var_23 = '_known_patterns'
    var_24 = '_section_comments'
    var_25 = '_section_comments_end'
    var_26 = '_skips'
    var_27 = '_skip_globs'
    var_28 = '_sorting_function'
    var_29 = 'other_setting'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/10 statements.
# Partially parsed test_config_init_with_settings_path. Retrieved 1/5 statements.
# Partially parsed test_config_init_with_src_paths. Retrieved 1/5 statements.


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

def test_case_0():
    var_0 = '[isort]\nline_length = 88\n'
    var_1 = True

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = True

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
    var_0 = "'  '"
    var_1 = True
    var_2 = 'indent'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.indent
    assert var_6 == '  '

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = True
    var_2 = 'profile'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 3
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = 'include_trailing_comma'
    var_7 = 'force_grid_wrap'
    var_8 = 'use_parentheses'
    var_9 = 'quiet'
    var_10 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_2, var_9: var_2}
    var_11 = module_0.Config(**var_10)
    var_12 = var_11.line_length
    assert var_12 == 100
    var_13 = var_11.multi_line_mode
    assert var_13 == 3
    var_14 = var_11.include_trailing_comma
    assert var_14 is True

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
    var_9 = True
    var_10 = 'known_django'
    var_11 = 'sections'
    var_12 = 'quiet'
    var_13 = {var_10: var_1, var_11: var_8, var_12: var_9}
    var_14 = module_0.Config(**var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True

def test_case_0():
    var_0 = True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'Standard library imports'
    var_2 = True
    var_3 = 'import_heading_future'
    var_4 = 'import_heading_stdlib'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = True
    var_3 = 'import_footer_future'
    var_4 = 'import_footer_stdlib'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 79
    var_5 = var_3.multi_line_mode
    var_6 = bool(var_3.multi_line_mode is not None)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 4/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'File not found'
    var_3 = 'test.py'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_deprecated_options_used_predicate_evaluates_to_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'old_setting'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {var_0}
    var_4 = [option for option in var_2 if option in var_3]
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_4[0]
    var_7 = bool(var_4[0] == var_0)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_deprecated_options_predicate_evaluates_to_true. Retrieved 12/15 statements.


def test_case_0():
    var_0 = 'old_setting_1'
    var_1 = 'old_setting_2'
    var_2 = 'indent'
    var_3 = 'other_setting'
    var_4 = 4
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 'value3'
    var_8 = {var_2: var_4, var_0: var_5, var_1: var_6, var_3: var_7}
    var_9 = {var_0, var_1}
    var_10 = [option for option in var_8 if option in var_9]
    var_11 = bool(var_10 == [var_0, var_1])
    assert var_11 is True
    var_12 = bool(var_10)
    assert var_12 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_config_init_creates_known_patterns_property. Retrieved 2/3 statements.
# Partially parsed test_config_init_creates_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_init_creates_skips_property. Retrieved 2/3 statements.
# Partially parsed test_config_init_creates_skip_globs_property. Retrieved 2/3 statements.


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
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

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
    var_0 = True
    var_1 = 88
    var_2 = 3
    var_3 = 'quiet'
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.quiet
    assert var_8 is True
    var_9 = var_7.line_length
    assert var_9 == 88
    var_10 = var_7.multi_line_mode
    assert var_10 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.known_patterns

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.skips

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.sorting_function
    var_3 = callable(var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------




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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 5/10 statements.
# Partially parsed test_get_config_data_toml_nested_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_multiple_values. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_boolean_conversion. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_space. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_tab. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_force_grid_wrap_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_quoted. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_double_quoted. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_missing_section. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_glob_pattern. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/13 statements.
# Partially parsed test_get_config_data_toml_missing_section. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_editorconfig_filters_unknown_keys. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_size_tab. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 100\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 88\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_first_party = module1,module2\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_first_party'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nuse_parentheses = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nmax_line_length = 120\n'
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
    var_0 = 'setup.cfg'
    var_1 = '[other]\nkey = value\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyx}]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.{py,pyx}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\noverride_profile = black,django\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'override_profile'

def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool.other]\nkey = 'value'\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nunknown_key = value\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'unknown_key'

def test_case_0():
    var_0 = '.editorconfig'



# Parsed testcases at query #31
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
    var_10 = 'YES'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is True
    var_12 = 'y'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is True
    var_14 = 'Y'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is True
    var_16 = '1'
    var_17 = module_0._as_bool(var_16)
    assert var_17 is True
    var_18 = 'on'
    var_19 = module_0._as_bool(var_18)
    assert var_19 is True
    var_20 = 'On'
    var_21 = module_0._as_bool(var_20)
    assert var_21 is True
    var_22 = 'ON'
    var_23 = module_0._as_bool(var_22)
    assert var_23 is True

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
    var_10 = 'NO'
    var_11 = module_0._as_bool(var_10)
    assert var_11 is False
    var_12 = 'n'
    var_13 = module_0._as_bool(var_12)
    assert var_13 is False
    var_14 = 'N'
    var_15 = module_0._as_bool(var_14)
    assert var_15 is False
    var_16 = '0'
    var_17 = module_0._as_bool(var_16)
    assert var_17 is False
    var_18 = 'off'
    var_19 = module_0._as_bool(var_18)
    assert var_19 is False
    var_20 = 'Off'
    var_21 = module_0._as_bool(var_20)
    assert var_21 is False
    var_22 = 'OFF'
    var_23 = module_0._as_bool(var_22)
    assert var_23 is False

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = 'invalid'
    var_2 = module_0._as_bool(var_1)
    var_3 = True
    var_4 = 'invalid truth value invalid'
    var_5 = bool(var_3)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0._as_bool(var_1)
    var_3 = True
    var_4 = 'invalid truth value'
    var_5 = bool(var_3)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = '2'
    var_2 = module_0._as_bool(var_1)
    var_3 = True
    var_4 = 'invalid truth value 2'
    var_5 = bool(var_3)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = '   '
    var_2 = module_0._as_bool(var_1)
    var_3 = True
    var_4 = 'invalid truth value'
    var_5 = bool(var_3)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'item'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1,item2,item3'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2', 'item3'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1\nitem2\nitem3'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2', 'item3'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1,item2\nitem3'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2', 'item3'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '  item1  ,  item2  \n  item3  '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2', 'item3'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '   \n   ,   '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '  item1  '
    var_1 = '  item2  '
    var_2 = '  item3  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['item1', 'item2', 'item3'])
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1,,item2'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'item1\n\nitem2'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['item1', 'item2'])
    assert var_2 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_formatter_in_combined_config. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = var_0 in var_2
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_editorconfig_file_path_predicate. Retrieved 4/15 statements.


def test_case_0():
    var_0 = '[*.py]\n'
    var_1 = 'indent_style = space\n'
    var_2 = 'indent_size = 4\n'
    var_3 = '.editorconfig'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_config_init_initializes_git_ls_files. Retrieved 4/5 statements.


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
    var_0 = "'    '"
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'git_ls_files'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.git_ls_files



# Parsed testcases at query #36
#--------------------------




import isort.utils as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.TrieNode(var_0, var_1)
    var_3 = 'default'
    var_4 = {}
    var_5 = module_0.Trie(var_3, var_4)
    var_6 = 'test'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = '/path/to/config'
    var_10 = var_5.insert(var_9, var_8)
    var_11 = '/path/to/config/file.py'
    var_12 = var_5.search(var_11)
    var_13 = bool(var_12 == ('', {}))
    assert var_13 is True
    var_14 = '/config'
    var_15 = 'root'
    var_16 = 'config'
    var_17 = {var_15: var_16}
    var_18 = var_5.insert(var_14, var_17)
    var_19 = '/config/subdir/file.py'
    var_20 = var_5.search(var_19)
    var_21 = var_20[1]
    var_22 = bool(var_20[1] == {'root': 'config'})
    assert var_22 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 10
    var_3 = var_1 < var_2
    var_4 = var_0 and var_3
    assert var_4 is False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_line_66_predicate_evaluates_to_true. Retrieved 5/17 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 88
    var_2 = 'black'
    var_3 = 'profile'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'isort.profiles'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_config_data_toml_predicate. Retrieved 2/10 statements.


def test_case_0():
    var_0 = b"[tool]\nkey = 'value'\n"
    var_1 = '.toml'



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_config_init_with_settings_path_cwd.
# Partially parsed test_config_init_with_known_patterns. Retrieved 4/5 statements.


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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

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
    var_0 = 'Future imports'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'future'
    var_6 = bool('future' in var_3.import_headings)
    assert var_6 is True
    var_7 = var_3.import_headings['future']
    assert var_7 == 'Future imports'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End stdlib'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'stdlib'
    var_6 = bool('stdlib' in var_3.import_footers)
    assert var_6 is True
    var_7 = var_3.import_footers['stdlib']
    assert var_7 == 'End stdlib'

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = 'black'
    var_2 = 'line_length'
    var_3 = 'profile'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.sources
    var_7 = bool(var_5.sources is not None)
    assert var_7 is True
    var_8 = var_5.sources
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 3
    var_2 = True
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'multi_line_mode'
    var_6 = 'include_trailing_comma'
    var_7 = 'force_single_line'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = var_9.line_length
    assert var_10 == 100
    var_11 = var_9.multi_line_mode
    assert var_11 == 3
    var_12 = var_9.include_trailing_comma
    assert var_12 is True
    var_13 = var_9.force_single_line
    assert var_13 is False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_multi_line_output_vertical_grid_grouped_no_comma_conversion. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '3'



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_path_root_predicate_evaluates_to_false.




# Parsed testcases at query #43
#--------------------------

# Failed to parse test_find_config_no_config_file.
# Partially parsed test_find_config_finds_pyproject_toml. Retrieved 2/9 statements.
# Partially parsed test_find_config_finds_setup_cfg. Retrieved 2/9 statements.
# Partially parsed test_find_config_searches_parent_directories. Retrieved 3/12 statements.
# Partially parsed test_find_config_stops_at_stop_dir. Retrieved 2/10 statements.
# Partially parsed test_find_config_prefers_earlier_config_source. Retrieved 4/14 statements.
# Partially parsed test_find_config_handles_exception_gracefully. Retrieved 2/9 statements.
# Partially parsed test_find_config_respects_max_depth. Retrieved 2/10 statements.
# Failed to parse test_find_config_returns_path_when_no_config_found.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'
    var_2 = 'source'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 88\n'
    var_2 = 'source'

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nline_length = 88\n'
    var_3 = 'source'

def test_case_0():
    var_0 = 'subdir'
    var_1 = '.git'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'setup.cfg'
    var_2 = '[tool.isort]\nline_length = 88\n'
    var_3 = '[isort]\nline_length = 100\n'
    var_4 = 'source'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content [[['

def test_case_0():
    var_0 = 'subdir'
    var_1 = True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 18/25 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_other_setting'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = '_Config'
    var_13 = ()
    var_14 = {}
    var_15 = [var_12, var_13, var_14]
    var_16 = '__iter__'
    var_17 = iter(var_11)
    var_18 = lambda self: var_17



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = space\n'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_find_config_no_config_file. Retrieved 1/5 statements.
# Partially parsed test_find_config_with_setup_cfg. Retrieved 3/10 statements.
# Partially parsed test_find_config_with_pyproject_toml. Retrieved 3/10 statements.
# Partially parsed test_find_config_searches_parent_directories. Retrieved 5/15 statements.
# Partially parsed test_find_config_stops_at_git_directory. Retrieved 2/9 statements.
# Partially parsed test_find_config_with_editorconfig. Retrieved 3/10 statements.
# Partially parsed test_find_config_returns_empty_dict_for_invalid_config. Retrieved 3/9 statements.
# Partially parsed test_find_config_max_search_depth. Retrieved 2/13 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 1

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length = 88\n'
    var_2 = 1

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'nested'
    var_3 = 'deep'
    var_4 = True

def test_case_0():
    var_0 = 'nested'
    var_1 = '.git'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_2 = 1

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[invalid section with no content'
    var_2 = {}

def test_case_0():
    var_0 = 0
    var_1 = 1



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_175_evaluates_to_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '/home/user/project/src'
    var_1 = [var_0]
    var_2 = '/home/user/project'
    var_3 = [var_2]
    var_4 = '/home/user/project/tests'
    var_5 = [var_4]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_abspaths_relative_paths_with_trailing_sep. Retrieved 3/12 statements.
# Partially parsed test_abspaths_relative_paths_without_trailing_sep. Retrieved 4/10 statements.
# Partially parsed test_abspaths_mixed_paths. Retrieved 5/12 statements.
# Partially parsed test_abspaths_single_relative_with_sep. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'relative/'
    var_1 = 'another/path/'
    var_2 = [var_0, var_1]

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path/'
    var_2 = '/another/absolute/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = {var_1, var_2}
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'relative'
    var_1 = 'another/path'
    var_2 = [var_0, var_1]
    var_3 = {var_0, var_1}

def test_case_0():
    var_0 = 'relative/'
    var_1 = '/absolute/'
    var_2 = 'no_sep'
    var_3 = '/abs/no/sep'
    var_4 = [var_0, var_1, var_2, var_3]

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = []
    var_2 = module_0._abspaths(var_0, var_1)
    var_3 = set()
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

def test_case_0():
    var_0 = 'single/'
    var_1 = [var_0]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_toml_nested_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_multiple_sections. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_space. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_tab. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_tuple_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_frozenset_conversion. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_bool_conversion_true. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_bool_conversion_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_integer. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_strip_quotes. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix_double_quotes. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_glob_pattern. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_nonexistent_section. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool.isort]\nprofile = 'black'\nline_length = 88\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool]\n[tool.isort]\nprofile = 'django'\n"
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\nline_length = 100\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[section1]\nkey1 = value1\n[section2]\nkey2 = value2\n'
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
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
    var_1 = '[*.py]\nmax_line_length = 120\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nknown_django = django,rest_framework\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'
    var_5 = 'django'
    var_6 = 'rest_framework'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = __pycache__,*.egg-info\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'skip'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'profile'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nforce_single_line = false\n'
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
    var_0 = '.editorconfig'
    var_1 = '[*.{py,pyx}]\nindent_size = 4\n'
    var_2 = '*.{py,pyx}'
    var_3 = (var_2,)
    var_4 = 0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_get_config_data_predicate_at_line_1_evaluates_to_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'config.ini'
    var_1 = '[settings]\n'
    var_2 = ()



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/9 statements.
# Partially parsed test_config_init_properties_lazy_load. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_init_section_comments_end_property. Retrieved 2/3 statements.
# Failed to parse test_config_init_preserves_directory.


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
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.quiet
    assert var_5 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
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
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.profile
    assert var_5 == 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.indent
    assert var_5 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.indent
    assert var_5 == '\t'

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
    var_9 = bool(var_8 is not None)
    assert var_9 is True
    var_10 = 'FUTURE'
    var_11 = bool('FUTURE' in var_8.sections)
    assert var_11 is True

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
    var_0 = 88
    var_1 = 3
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_mode'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.line_length
    assert var_9 == 88
    var_10 = var_7.quiet
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'migrations'
    var_1 = [var_0]
    var_2 = 'build'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7._skips
    assert var_8 is None
    var_9 = var_7.skips
    var_10 = var_7._skips
    var_11 = bool(var_7._skips is not None)
    assert var_11 is True
    var_12 = 'migrations'
    var_13 = bool('migrations' in var_9)
    assert var_13 is True
    var_14 = 'build'
    var_15 = bool('build' in var_9)
    assert var_15 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.egg-info'
    var_1 = [var_0]
    var_2 = '*.pyc'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7._skip_globs
    assert var_8 is None
    var_9 = var_7.skip_globs
    var_10 = var_7._skip_globs
    var_11 = bool(var_7._skip_globs is not None)
    assert var_11 is True
    var_12 = '*.egg-info'
    var_13 = bool('*.egg-info' in var_9)
    assert var_13 is True
    var_14 = '*.pyc'
    var_15 = bool('*.pyc' in var_9)
    assert var_15 is True

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
    var_2 = var_1._section_comments_end
    assert var_2 is None
    var_3 = var_1.section_comments_end
    var_4 = var_1._section_comments_end
    var_5 = bool(var_1._section_comments_end is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._sorting_function
    assert var_4 is None
    var_5 = var_3.sorting_function
    var_6 = var_3._sorting_function
    var_7 = bool(var_3._sorting_function is not None)
    assert var_7 is True
    var_8 = callable(var_5)
    var_9 = bool(var_8)
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'DJANGO'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'known_django'
    var_10 = 'sections'
    var_11 = {var_9: var_1, var_10: var_8}
    var_12 = module_0.Config(**var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_attr'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = None
    var_13 = None



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_init_with_settings_file. Retrieved 2/6 statements.
# Failed to parse test_config_init_with_src_paths.
# Partially parsed test_config_init_directory_from_config_source. Retrieved 2/7 statements.
# Partially parsed test_config_init_directory_default_to_cwd. Retrieved 1/3 statements.
# Partially parsed test_config_init_with_multiple_config_sources. Retrieved 3/7 statements.


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
    var_1 = 88
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 88

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.profile
    assert var_5 == 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_profile_xyz'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'DJANGO'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'known_django'
    var_10 = 'sections'
    var_11 = {var_9: var_1, var_10: var_8}
    var_12 = module_0.Config(**var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*/src'
    var_1 = [var_0]
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = 'force_single_line'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'Standard library'
    var_2 = 'import_heading_future'
    var_3 = 'import_heading_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'future'
    var_8 = bool('future' in var_5.import_headings)
    assert var_8 is True
    var_9 = 'stdlib'
    var_10 = bool('stdlib' in var_5.import_headings)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = 'import_footer_future'
    var_3 = 'import_footer_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'future'
    var_8 = bool('future' in var_5.import_footers)
    assert var_8 is True
    var_9 = 'stdlib'
    var_10 = bool('stdlib' in var_5.import_footers)
    assert var_10 is True

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
    var_1 = 'nonexistent_setting_xyz'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.known_patterns
    var_3 = bool(var_2 is var_1.known_patterns)
    assert var_3 is True

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=100\n'
    var_2 = 120



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 8/26 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_is_skipped_with_exact_skip_match. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_non_matching_skip. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_directory_in_skips. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_skip_glob_pattern. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_non_matching_skip_glob. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_nonexistent_path. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_with_extend_skip. Retrieved 7/9 statements.
# Partially parsed test_is_skipped_with_extend_skip_glob. Retrieved 8/10 statements.
# Partially parsed test_is_skipped_with_git_folder_name. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_with_directory_set. Retrieved 3/10 statements.
# Partially parsed test_is_skipped_with_nested_directory_skip. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_glob_leading_slash. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = 'other_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'some_dir/__pycache__/file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.pyc'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/that/does/not/exist'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'file2.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'skip'
    var_7 = 'extend_skip'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = [var_3]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '*.pyo'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'skip_glob'
    var_7 = 'extend_skip_glob'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = 'test.pyo'
    var_11 = [var_10]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '.git'
    var_5 = [var_4]

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'tests/unit/test_file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '/test/*.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test/file.py'
    var_7 = [var_6]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 14/19 statements.


def test_case_0():
    var_0 = 'py_version'
    assert var_0 is True
    var_1 = 'quiet'
    var_2 = 'profile'
    var_3 = '_known_patterns'
    var_4 = '_section_comments'
    var_5 = '_section_comments_end'
    var_6 = '_skips'
    var_7 = '_skip_globs'
    var_8 = '_sorting_function'
    var_9 = 'py310'
    var_10 = False
    var_11 = 'default'
    var_12 = None
    var_13 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_12, var_5: var_12, var_6: var_12, var_7: var_12, var_8: var_12}



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_find_config_returns_tuple_with_path_and_dict. Retrieved 7/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._find_config(var_0)
    var_2 = len(var_1)
    assert var_2 == 2
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = '/test/path'
    var_1 = module_0._find_config(var_0)
    var_2 = var_1[0]
    assert var_2 == '/test/path'
    var_3 = var_1[1]
    var_4 = bool(var_1[1] == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = 'line_length'
    var_2 = '    '
    var_3 = 88
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '/test/path'
    var_6 = module_0._find_config(var_5)
    var_7 = var_6[1]
    var_8 = bool(var_6[1] == var_4)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = '/test/path'
    var_3 = module_0._find_config(var_2)
    var_4 = var_3[0]
    assert var_4 == '/test/path'
    var_5 = var_3[1]
    var_6 = bool(var_3[1] == {})
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/test/path'
    var_1 = module_0._find_config(var_0)
    var_2 = var_1[0]
    assert var_2 == '/test/path'
    var_3 = var_1[1]
    var_4 = bool(var_1[1] == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = '/test/path'
    var_3 = module_0._find_config(var_2)
    var_4 = var_3[0]
    assert var_4 == '/test/path'
    var_5 = var_3[1]
    var_6 = bool(var_3[1] == {})
    assert var_6 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.py'
    var_3 = (var_2,)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 4/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'File not found'
    var_3 = 'test.py'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 12/19 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = [var_0, var_1]
    var_3 = 'pyc'
    var_4 = 'pyo'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = 'quiet'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(**var_8)
    var_10 = [var_0, var_1]
    var_11 = [var_3, var_4]
    var_12 = 'test.pyc'
    var_13 = var_9.is_supported_filetype(var_12)
    assert var_13 is False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_line_44_evaluates_to_true. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'source'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_is_supported_filetype_with_shebang. Retrieved 2/10 statements.


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
    var_2 = 'test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = b'#!/usr/bin/env python\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)

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
    var_2 = 'Makefile'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #63
#--------------------------

# Failed to parse test_config_init_with_settings_path.
# Partially parsed test_config_init_with_src_paths. Retrieved 1/6 statements.


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
    var_1 = 'black'
    var_2 = 'quiet'
    var_3 = 'profile'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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
    var_0 = 'end of stdlib'
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
    var_9 = bool(var_8 is not None)
    assert var_9 is True

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
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'natural'

import isort.settings as module_0

def test_case_0():
    var_0 = 'mylib'
    var_1 = [var_0]
    var_2 = 'known_mylib'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = 'mylib'
    var_7 = bool('mylib' in var_4.known_other)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'black'
    var_2 = 88
    var_3 = 'quiet'
    var_4 = 'profile'
    var_5 = 'line_length'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.quiet
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    assert var_4 == '/tmp'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_deprecated_options_used_predicate_evaluates_to_true. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'deprecated_option_1'
    var_1 = 'deprecated_option_2'
    var_2 = {var_0, var_1}
    var_3 = 'other_option'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = [option for option in var_7 if option in var_2]
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 2
    var_11 = 'deprecated_option_1'
    var_12 = bool('deprecated_option_1' in var_8)
    assert var_12 is True
    var_13 = 'deprecated_option_2'
    var_14 = bool('deprecated_option_2' in var_8)
    assert var_14 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_config_init_with_settings_path. Retrieved 2/9 statements.
# Partially parsed test_config_is_supported_filetype_python. Retrieved 3/4 statements.
# Partially parsed test_config_is_supported_filetype_blocked. Retrieved 3/4 statements.
# Partially parsed test_config_known_patterns_property. Retrieved 3/6 statements.
# Partially parsed test_config_section_comments_property. Retrieved 2/3 statements.
# Partially parsed test_config_section_comments_end_property. Retrieved 2/3 statements.
# Partially parsed test_config_skips_property. Retrieved 6/7 statements.
# Partially parsed test_config_skip_globs_property. Retrieved 6/7 statements.
# Partially parsed test_config_parse_known_pattern_simple. Retrieved 2/3 statements.


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
    var_7 = var_5.line_length
    assert var_7 == 100

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88\n'

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
    var_0 = "'  '"
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)

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
    var_2 = 'test.pyc'
    var_3 = var_1.is_supported_filetype(var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.known_patterns
    var_3 = 2

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.section_comments

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.section_comments_end

import isort.settings as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'build'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skips

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = 'build/*'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.skip_globs

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'django'

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'myapp'
    var_3 = [var_2]
    var_4 = 'known_django'
    var_5 = 'known_first_party'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'lib'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/7 statements.
# Partially parsed test_config_init_with_src_paths. Retrieved 1/8 statements.


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
    var_0 = '310'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = {}
    var_5 = module_0.Config(config=var_3, **var_4)
    var_6 = var_5._known_patterns
    assert var_6 is None
    var_7 = var_5._section_comments
    assert var_7 is None

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.quiet
    assert var_6 is True
    var_7 = var_5.line_length
    assert var_7 == 100

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'

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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'FUTURE'
    var_3 = 'STDLIB'
    var_4 = 'DJANGO'
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'known_django'
    var_10 = 'sections'
    var_11 = {var_9: var_1, var_10: var_8}
    var_12 = module_0.Config(**var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'Stdlib imports'
    var_2 = 'import_heading_future'
    var_3 = 'import_heading_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = 'import_footer_future'
    var_3 = 'import_footer_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_formatter_xyz'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'FormattingPluginDoesNotExist'

def test_case_0():
    var_0 = 'src'

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.skips
    var_3 = var_1.skips
    var_4 = bool(var_2 is var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '__pycache__'
    var_1 = [var_0]
    var_2 = 'venv'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '__pycache__'
    var_9 = bool('__pycache__' in var_7.skips)
    assert var_9 is True
    var_10 = 'venv'
    var_11 = bool('venv' in var_7.skips)
    assert var_11 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '*.egg-info'
    var_1 = [var_0]
    var_2 = 'build/*'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '*.egg-info'
    var_9 = bool('*.egg-info' in var_7.skip_globs)
    assert var_9 is True
    var_10 = 'build/*'
    var_11 = bool('build/*' in var_7.skip_globs)
    assert var_11 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_formatter_plugin_loading. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'Plugin not found'
    var_5 = [var_4]
    var_6 = 'formatting_function'
    var_7 = bool('formatting_function' in var_3)
    assert var_7 is True
    var_8 = var_3['formatting_function']
    var_9 = bool(var_3['formatting_function'] is not None)
    assert var_9 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_abspaths_relative_path_with_trailing_sep. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'documents/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = [var_1]



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_44_evaluates_to_true. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'source'



# Parsed testcases at query #70
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
    var_2 = 'test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_line_43_evaluates_to_true. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = {}
    var_3 = False



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_find_all_configs_exception_handling. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'Test that the exception handler at line 18 catches exceptions from _get_config_data'
    var_1 = 'setup.cfg'
    var_2 = '[isort]\nprofile=black\n'
    var_3 = 'isort.settings._get_config_data'
    var_4 = []
    var_5 = 'isort.settings.warn'
    var_6 = len(var_4)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 'Failed to pull configuration information from'
    var_9 = bool('Failed to pull configuration information from' in var_4[0])
    assert var_9 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_config_constructor_known_patterns_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_section_comments_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_section_comments_end_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_skips_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_skip_globs_lazy_loading. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_src_paths_defaults. Retrieved 2/3 statements.


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
    var_2 = var_1._section_comments_end
    assert var_2 is None
    var_3 = var_1.section_comments_end
    var_4 = var_1._section_comments_end
    var_5 = bool(var_1._section_comments_end is not None)
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'natural'
    var_5 = var_3.sorting_function
    var_6 = callable(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'native'
    var_5 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.src_paths
    var_3 = bool(var_1.src_paths is not None)
    assert var_3 is True
    var_4 = var_1.src_paths



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_with_exact_skip_path. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_skip_folder_in_path. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_glob_pattern. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_nonexistent_path. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_with_git_file. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_with_normal_file. Retrieved 3/11 statements.
# Partially parsed test_is_skipped_with_extended_skip. Retrieved 7/9 statements.
# Partially parsed test_is_skipped_with_skip_glob_pattern_match. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_with_directory_context. Retrieved 2/11 statements.
# Partially parsed test_is_skipped_with_extend_skip_glob. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/skip'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_folder'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '/some/path/skip_folder/file.py'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.pyc'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/file.py'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '.git'
    var_5 = [var_4]

import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = var_2.is_skipped(var_0)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = 'extended_skip'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'skip'
    var_6 = 'extend_skip'
    var_7 = {var_5: var_1, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = '/some/path/extended_skip/file.py'
    var_10 = [var_9]

import isort.settings as module_0

def test_case_0():
    var_0 = '**/test_*.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_glob'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_file.py'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file.py'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = frozenset(var_0)
    var_2 = '*.bak'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'skip_glob'
    var_6 = 'extend_skip_glob'
    var_7 = {var_5: var_1, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = 'file.bak'
    var_10 = [var_9]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_data_toml_file. Retrieved 5/10 statements.
# Partially parsed test_get_config_data_ini_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_file. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_editorconfig_tab_indent. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_boolean_value. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_frozenset_value. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_known_prefix. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_glob_pattern. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_nested_toml_sections. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'test.toml'
    var_1 = '\n[tool.isort]\nline_length = 88\nmulti_line_mode = 3\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nline_length = 100\nskip = file1.py,file2.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '\nroot = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '\n[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_2 = '*.py'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '\n[*.py]\nmax_line_length = off\n'
    var_2 = '*.py'
    var_3 = (var_2,)
    var_4 = 'inf'
    var_5 = float(var_4)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nforce_alphabetical_sort = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nsections = FUTURE,STDLIB,THIRDPARTY\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'sections'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nknown_django = django\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'known_django'
    var_5 = 'django'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nforce_grid_wrap = false\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\nforce_grid_wrap = true\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '\n[isort]\ncomment_prefix = "# "\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '\n[*.{py,pyi}]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '\n[tool.isort]\nline_length = 120\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_constructor_with_known_sections. Retrieved 4/5 statements.


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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
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
    var_0 = True
    var_1 = 88
    var_2 = 'black'
    var_3 = 3
    var_4 = 'quiet'
    var_5 = 'line_length'
    var_6 = 'profile'
    var_7 = 'multi_line_mode'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = var_9.quiet
    assert var_10 is True
    var_11 = var_9.line_length
    assert var_11 == 88
    var_12 = var_9.multi_line_mode
    assert var_12 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = set()
    var_7 = 'django'

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
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.src_paths
    var_5 = bool(var_3.src_paths is not None)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 28/36 statements.


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
    var_12 = 'Config'
    var_13 = ()
    var_14 = '__init__'
    var_15 = 'py_version'
    var_16 = '_known_patterns'
    var_17 = '_section_comments'
    var_18 = '_section_comments_end'
    var_19 = '_skips'
    var_20 = '_skip_globs'
    var_21 = '_sorting_function'
    var_22 = 'other_field'
    var_23 = None
    var_24 = lambda self, **kwargs: var_23
    var_25 = 'py310'
    var_26 = 'value'
    var_27 = {var_14: var_24, var_15: var_25, var_16: var_23, var_17: var_23, var_18: var_23, var_19: var_23, var_20: var_23, var_21: var_23, var_22: var_26}
    var_28 = [var_12, var_13, var_27]



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'some_setting'
    var_8 = 'py311'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_abspaths_relative_path_with_trailing_sep. Retrieved 2/10 statements.
# Partially parsed test_abspaths_multiple_values. Retrieved 5/14 statements.
# Partially parsed test_abspaths_single_dot_relative_path. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'subdir/'
    var_1 = [var_0]

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = bool(var_3 == {'/absolute/path/'})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'relative/path'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = bool(var_3 == {'relative/path'})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = bool(var_3 == {'/absolute/path'})
    assert var_4 is True

def test_case_0():
    var_0 = 'subdir1/'
    var_1 = 'subdir2/'
    var_2 = '/absolute/'
    var_3 = 'relative'
    var_4 = [var_0, var_1, var_2, var_3]

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
    var_1 = './subdir/'
    var_2 = [var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = [var_1]



# Parsed testcases at query #8
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
    var_2 = 'test.xyz'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/path/file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_skipped_predicate_line_3_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_line_123_evaluates_to_false. Retrieved 15/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'known_custom'
    var_1 = 'sections'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'CUSTOM'
    var_9 = 'FIRSTPARTY'
    var_10 = 'LOCALFOLDER'
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {var_0: var_4, var_1: var_11}
    var_13 = 'known_custom'
    var_14 = 'sections'
    var_15 = {var_13: var_4, var_14: var_11}
    var_16 = module_0.Config(**var_15)
    var_17 = 'setting is defined'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path/'
    var_2 = [var_1]



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_config_init_with_settings_path.
# Partially parsed test_config_init_sources_tuple. Retrieved 2/3 statements.


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
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.quiet
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
    var_0 = '  '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'sections'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'known_standard_library'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.quiet
    assert var_6 is True
    var_7 = var_5.line_length
    assert var_7 == 100

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
    var_0 = 'src'
    var_1 = 'lib'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths
    var_7 = bool(var_5.src_paths is not None)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.sources

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'known_other'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_config_data_toml_basic. Retrieved 5/11 statements.
# Partially parsed test_get_config_data_toml_nested. Retrieved 5/9 statements.
# Partially parsed test_get_config_data_ini_basic. Retrieved 4/9 statements.
# Partially parsed test_get_config_data_ini_skip_list. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_space. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_tab. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_max_line_length_off. Retrieved 6/10 statements.
# Partially parsed test_get_config_data_editorconfig_with_section_header. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_bool_value. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_number. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_false. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_force_grid_wrap_true. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_comment_prefix. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_empty_file. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_ini_multiline_list. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_space_default. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_editorconfig_indent_style_tab_default. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_ini_extension_pattern. Retrieved 4/8 statements.
# Partially parsed test_get_config_data_ini_extension_pattern_multiple. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool.isort]\nline_length = 88\nskip = ["file1.py", "file2.py"]\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'config.toml'
    var_1 = '[tool]\n[tool.isort]\nprofile = "black"\n'
    var_2 = 'tool'
    var_3 = 'isort'
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length = 88\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = file1.py,file2.py\n'
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
    var_1 = '[*]\nmax_line_length = 120\n'
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
    var_0 = '.editorconfig'
    var_1 = 'root = true\n\n[*]\nindent_style = space\nindent_size = 4\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile = black\nmulti_line_mode = 3\n'
    var_2 = 'isort'
    var_3 = (var_2,)
    var_4 = 'profile'

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
    var_1 = ''
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nskip = \n    file1.py\n    file2.py\n'
    var_2 = 'isort'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = space\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*]\nindent_style = tab\n'
    var_2 = '*'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[*.{py,pyi}]\nline_length = 100\n'
    var_2 = '*.{py,pyi}'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[*.{py,pyi}]\nline_length = 100\n'
    var_2 = '*.{py}'
    var_3 = (var_2,)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_config_constructor_with_settings_path.
# Partially parsed test_config_constructor_known_patterns_lazy_initialization. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_section_comments_lazy_initialization. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_skips_lazy_initialization. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_skip_globs_lazy_initialization. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_section_comments_end_lazy_initialization. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_directory_initialization. Retrieved 2/3 statements.


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
    var_1 = 88
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.quiet
    assert var_7 is True
    var_8 = var_5.line_length
    assert var_8 == 88

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

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

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
    var_5 = var_3.line_length
    assert var_5 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1._section_comments_end
    assert var_2 is None
    var_3 = var_1.section_comments_end
    var_4 = var_1._section_comments_end
    var_5 = bool(var_1._section_comments_end is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'skip_gitignore'
    var_5 = 'quiet'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.line_length
    assert var_8 == 100
    var_9 = var_7.skip_gitignore
    assert var_9 is True
    var_10 = var_7.quiet
    assert var_10 is False

import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 100
    var_5 = 'line_length'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(config=var_3, **var_6)
    var_8 = var_7.line_length
    assert var_8 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = bool(var_1.directory is not None)
    assert var_3 is True
    var_4 = var_1.directory



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_import_footer_prefix_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'import_footer_'
    var_1 = 'import_footer_section1'



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_98_evaluates_to_false. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 98 evaluates to False for known_standard_library.'
    var_1 = 'known_'
    var_2 = 'known_standard_library'
    var_3 = 'known_standard_library'
    var_4 = 'known_future_library'
    var_5 = 'known_third_party'
    var_6 = 'known_first_party'
    var_7 = 'known_local_folder'
    var_8 = (var_3, var_4, var_5, var_6, var_7)
    var_9 = var_2 not in var_8



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_line_14_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\n'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_constructor_with_config_object. Retrieved 13/20 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = 'quiet'
    var_2 = '_known_patterns'
    var_3 = '_section_comments'
    var_4 = '_section_comments_end'
    var_5 = '_skips'
    var_6 = '_skip_globs'
    var_7 = '_sorting_function'
    var_8 = 'py310'
    var_9 = False
    var_10 = None
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_10, var_4: var_10, var_5: var_10, var_6: var_10, var_7: var_10}
    var_12 = 'black'

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/.isort.cfg'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2._known_patterns
    assert var_3 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2._known_patterns
    assert var_3 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = True
    var_2 = 'profile'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5._known_patterns
    assert var_6 is None
    var_7 = var_5._skips
    assert var_7 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._known_patterns
    assert var_4 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3._known_patterns
    assert var_4 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

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



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = "[tool]\nkey = 'value'\n"
    var_2 = '.toml'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_formatter_in_combined_config. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}
    var_3 = 'formatter'
    var_4 = bool('formatter' in var_2)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_config_init_preserves_known_other. Retrieved 4/5 statements.
# Failed to parse test_config_init_with_custom_directory.
# Partially parsed test_config_init_sources_tuple. Retrieved 2/3 statements.


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
    var_0 = "'    '"
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '

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
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'known_django'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = frozenset()
    var_6 = 'django'

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'import_heading_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings
    var_5 = bool(var_3.import_headings is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of future imports'
    var_1 = 'import_footer_future'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers
    var_5 = bool(var_3.import_footers is not None)
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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.sources

import isort.settings as module_0

def test_case_0():
    var_0 = 88
    var_1 = 3
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_mode'
    var_5 = 'use_parentheses'
    var_6 = 'quiet'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = var_8.line_length
    assert var_9 == 88
    var_10 = var_8.use_parentheses
    assert var_10 is True
    var_11 = var_8.quiet
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = 'profile'
    var_1 = 'line_length'
    var_2 = 'black'
    var_3 = 88
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #25
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
    var_7 = 'some_other_attr'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_path_root_predicate_evaluates_to_false.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_159_predicate_evaluates_to_true. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = '/path/to/config/file.cfg'
    var_2 = {var_0: var_1}
    var_3 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_config_post_init_py_version_auto. Retrieved 3/4 statements.
# Partially parsed test_config_post_init_vertical_grid_grouped_no_comma. Retrieved 1/3 statements.


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
    var_3 = 'python version'
    var_4 = bool('python version' in str(e).lower())
    assert var_4 is True

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
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = (var_0, var_1)
    var_3 = frozenset(var_2)
    var_4 = '3'
    var_5 = module_0._Config(var_4, known_standard_library=var_3)
    var_6 = var_5.known_standard_library
    var_7 = bool(var_5.known_standard_library == var_3)
    assert var_7 is True

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
    var_1 = 79
    var_2 = 100
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'wrap_length'

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = 79
    var_2 = module_0._Config(var_0, line_length=var_1, wrap_length=var_1)
    var_3 = var_2.wrap_length
    assert var_3 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = 100
    var_2 = 79
    var_3 = module_0._Config(var_0, line_length=var_1, wrap_length=var_2)
    var_4 = var_3.wrap_length
    assert var_4 == 79

def test_case_0():
    var_0 = '3'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_skipped_predicate_line_3_false. Retrieved 2/21 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/some/test/file.py'
    var_3 = [var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_config_init_with_settings_file. Retrieved 2/9 statements.
# Partially parsed test_config_init_with_settings_path. Retrieved 2/7 statements.
# Partially parsed test_config_init_with_known_sections. Retrieved 7/10 statements.
# Partially parsed test_config_init_with_import_headings. Retrieved 5/8 statements.
# Partially parsed test_config_init_with_import_footers. Retrieved 5/8 statements.
# Partially parsed test_config_init_with_src_paths. Retrieved 1/10 statements.
# Partially parsed test_config_init_default_directory. Retrieved 1/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py39'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 100
    var_5 = 'line_length'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(config=var_3, **var_6)
    var_8 = var_7.line_length
    assert var_8 == 100
    var_9 = var_7.py_version
    assert var_9 == '39'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 3
    var_2 = 4
    var_3 = 'line_length'
    var_4 = 'multi_line_mode'
    var_5 = 'indent'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.line_length
    assert var_8 == 100
    var_9 = var_7.multi_line_mode
    assert var_9 == 3
    var_10 = var_7.indent
    assert var_10 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'
    var_5 = var_3.line_length
    assert var_5 == 88

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
    var_0 = 2
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
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'numpy'
    var_3 = [var_2]
    var_4 = 'known_django'
    var_5 = 'known_numpy'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = frozenset()
    var_9 = 'django'
    var_10 = frozenset()
    var_11 = 'numpy'

import isort.settings as module_0

def test_case_0():
    var_0 = 'Future imports'
    var_1 = 'Stdlib imports'
    var_2 = 'import_heading_future'
    var_3 = 'import_heading_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'future'
    var_7 = 'stdlib'

import isort.settings as module_0

def test_case_0():
    var_0 = 'End future'
    var_1 = 'End stdlib'
    var_2 = 'import_footer_future'
    var_3 = 'import_footer_stdlib'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'future'
    var_7 = 'stdlib'

def test_case_0():
    var_0 = 'src'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.line_length
    assert var_6 == 100

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory

import isort.settings as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'unsupported_option_xyz'
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
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.skips
    var_5 = var_3.skips
    var_6 = bool(var_4 is var_5)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_is_supported_filetype_fifo_returns_false. Retrieved 5/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = 'os.stat'
    var_4 = 'stat.S_ISFIFO'
    var_5 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_line_197_predicate_evaluates_to_true. Retrieved 6/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'force_single_line'
    var_2 = '.'
    var_3 = {}
    var_4 = True
    var_5 = module_0.Config(settings_path=var_2, **var_1)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_line_66_predicate_evaluates_to_true. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 'MockPlugin'
    var_1 = 'name'
    var_2 = 'load'
    var_3 = [var_1, var_2]
    var_4 = 'black'
    var_5 = 'line_length'
    var_6 = 88
    var_7 = {var_5: var_6}
    var_8 = lambda : var_7
    var_9 = 'isort.profiles'
    var_10 = bool(var_1 > 0)
    assert var_10 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_line_98_evaluates_to_false. Retrieved 24/29 statements.


def test_case_0():
    var_0 = 'known_standard_library'
    var_1 = 'known_future_library'
    var_2 = 'known_third_party'
    var_3 = 'known_first_party'
    var_4 = 'known_local_folder'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = '__future__'
    var_9 = [var_8]
    var_10 = 'numpy'
    var_11 = [var_10]
    var_12 = 'mymodule'
    var_13 = [var_12]
    var_14 = 'local'
    var_15 = [var_14]
    var_16 = {var_0: var_7, var_1: var_9, var_2: var_11, var_3: var_13, var_4: var_15}
    var_17 = 'known_'
    var_18 = 'known_standard_library'
    var_19 = 'known_future_library'
    var_20 = 'known_third_party'
    var_21 = 'known_first_party'
    var_22 = 'known_local_folder'
    var_23 = (var_18, var_19, var_20, var_21, var_22)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_line_123_predicate_evaluates_to_true. Retrieved 18/25 statements.


def test_case_0():
    var_0 = 'known_custom'
    var_1 = 'sections'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = 'FUTURE'
    var_6 = 'STDLIB'
    var_7 = 'THIRDPARTY'
    var_8 = 'FIRSTPARTY'
    var_9 = 'LOCALFOLDER'
    var_10 = (var_5, var_6, var_7, var_8, var_9)
    var_11 = {var_0: var_4, var_1: var_10}
    var_12 = 'known_custom'
    var_13 = [var_2, var_3]
    var_14 = 'known_'
    var_15 = len(var_14)
    var_16 = var_12[var_15:]
    var_17 = ()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_config_init_with_config_parameter. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'other_attr'
    var_8 = 'py310'
    var_9 = None
    var_10 = 'value'
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '*.{py,txt}'
    var_1 = '*.{'
    var_2 = '}'



# Parsed testcases at query #38
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'a,b,c'
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
    var_0 = 'a,b\nc,d'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c', 'd'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ' a , b , c '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ' a '
    var_1 = ' b '
    var_2 = ' c '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a,,b,,c'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '   ,   ,   '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['single'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = [var_0]
    var_2 = module_0._as_list(var_1)
    var_3 = bool(var_2 == ['single'])
    assert var_3 is True



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    var_0 = 'comment_prefix'
    assert var_0 == 'comment_prefix'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_is_supported_filetype_opens_file_when_extension_not_supported. Retrieved 5/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = b'#!/usr/bin/env python\n'
    var_1 = b'print("hello")\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.supported_extensions
    var_5 = var_3.blocked_extensions
    var_6 = bool(var_0)
    assert var_6 is True



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_config_init_with_settings_path.


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
    var_0 = True
    var_1 = 100
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5._known_patterns
    assert var_6 is None
    var_7 = var_5._section_comments
    assert var_7 is None
    var_8 = var_5._section_comments_end
    assert var_8 is None
    var_9 = var_5._skips
    assert var_9 is None
    var_10 = var_5._skip_globs
    assert var_10 is None
    var_11 = var_5._sorting_function
    assert var_11 is None
    var_12 = var_5.line_length
    assert var_12 == 100

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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_deprecated_options_used_predicate_evaluates_to_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'some_deprecated_option'
    var_1 = [var_0]
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_find_config_returns_path_and_empty_dict_when_no_config_found. Retrieved 6/12 statements.
# Partially parsed test_find_config_returns_config_data_when_file_found. Retrieved 9/21 statements.
# Partially parsed test_find_config_stops_at_stop_directory. Retrieved 6/20 statements.
# Partially parsed test_find_config_searches_parent_directories. Retrieved 11/27 statements.
# Partially parsed test_find_config_handles_exception_and_continues_search. Retrieved 9/25 statements.
# Partially parsed test_find_config_respects_max_search_depth. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'os.path.isfile'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = 'os.path.isdir'
    var_4 = lambda x: var_1
    var_5 = {}

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'os.path.isfile'
    var_3 = 'os.path.isdir'
    var_4 = '__main__._get_config_data'
    var_5 = 'line_length'
    var_6 = 'source'
    var_7 = 88
    var_8 = lambda path, sections: {var_5: var_7, var_6: path}
    var_9 = 'line_length'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = '.git'
    var_3 = 'os.path.isfile'
    var_4 = 'os.path.isdir'
    var_5 = {}

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nline_length=88\n'
    var_4 = 'os.path.isfile'
    var_5 = 'os.path.isdir'
    var_6 = '__main__._get_config_data'
    var_7 = 'line_length'
    var_8 = 'source'
    var_9 = 88
    var_10 = lambda path, sections: {var_7: var_9, var_8: path}
    var_11 = 'line_length'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'os.path.isfile'
    var_3 = 'os.path.isdir'
    var_4 = '__main__._get_config_data'
    var_5 = '__main__.warn'
    var_6 = None
    var_7 = lambda msg, stacklevel: var_6
    var_8 = {}

def test_case_0():
    var_0 = 'os.path.isfile'
    var_1 = 'os.path.isdir'
    var_2 = {}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_line_23_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'stop_marker'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_line_123_evaluates_to_true. Retrieved 21/27 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py310'
    var_8 = None
    var_9 = {var_0: var_7, var_1: var_8, var_2: var_8, var_3: var_8, var_4: var_8, var_5: var_8, var_6: var_8}
    var_10 = 'sections'
    var_11 = 'FUTURE'
    var_12 = 'STDLIB'
    var_13 = 'THIRDPARTY'
    var_14 = 'FIRSTPARTY'
    var_15 = 'LOCALFOLDER'
    var_16 = (var_11, var_12, var_13, var_14, var_15)
    var_17 = {var_10: var_16}
    var_18 = 'CUSTOMSECTION'
    var_19 = False
    var_20 = ()



