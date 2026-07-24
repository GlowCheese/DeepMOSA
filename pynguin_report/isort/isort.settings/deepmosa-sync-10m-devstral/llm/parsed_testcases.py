####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_config_post_init_with_valid_py_version. Retrieved 3/6 statements.
# Partially parsed test_config_post_init_with_auto_py_version. Retrieved 3/6 statements.
# Failed to parse test_config_post_init_with_vertical_grid_grouped_no_comma.


import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py3.8'
    var_3 = 'py3.8'
    var_4 = var_1.known_standard_library

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    var_3 = var_1.py_version
    var_4 = var_1.known_standard_library

import isort.settings as module_0

def test_case_0():
    var_0 = '4.0'
    var_1 = module_0._Config(var_0)

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

import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.py_version
    assert var_1 == 'py3'
    var_2 = var_0.line_length
    assert var_2 == 79
    var_3 = var_0.wrap_length
    assert var_3 == 0
    var_4 = var_0.multi_line_output
    var_5 = '__future__'
    var_6 = (var_5,)
    var_7 = frozenset(var_6)
    var_8 = var_0.known_future_library
    var_9 = bool(var_0.known_future_library == var_7)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_config. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_profile_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_formatter_does_not_exist. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_sort_order_does_not_exist. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_settings.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.settings_file
    assert var_3 == 'test_settings.ini'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.settings_path
    assert var_3 == 'test_path'

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
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
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
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid_path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(not var_1._known_patterns)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}
    var_3 = 'indent'
    var_4 = bool('indent' in var_2)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir1/'
    var_2 = 'dir2/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = bool(var_4 == {'/home/user/dir1/', '/home/user/dir2/'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/abs/dir1/'
    var_2 = '/abs/dir2/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = bool(var_4 == {'/abs/dir1/', '/abs/dir2/'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'dir1/'
    var_2 = '/abs/dir2/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = bool(var_4 == {'/home/user/dir1/', '/abs/dir2/'})
    assert var_5 is True

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
    var_1 = 'dir1/'
    var_2 = [var_1, var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = bool(var_3 == {'/home/user/dir1/'})
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_init_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_init_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_skips. Retrieved 4/5 statements.
# Partially parsed test_config_init_with_skip_globs. Retrieved 4/5 statements.
# Partially parsed test_config_init_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_unsupported_config. Retrieved 3/5 statements.


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
    var_1 = 'bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'import_footer_foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'skip'
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'foo'

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'foo'

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'foo'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'foo'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'indent'
    var_5 = bool('indent' in var_3.__dict__)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_constructor_default. Retrieved 2/9 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/8 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = 'src'
    var_4 = var_1.src_paths

def test_case_0():
    var_0 = '[isort]\nline_length=120\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length=100\n'

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'quiet'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.quiet
    assert var_6 is True
    var_7 = var_5.line_length
    assert var_7 == 80

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
    var_4 = var_3.line_length
    assert var_4 == 88
    var_5 = var_3.multi_line_output
    assert var_5 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'include_trailing_comma'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = 'nonexistent'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test___post_init___with_py_version_auto. Retrieved 3/4 statements.
# Partially parsed test___post_init___with_known_standard_library_empty. Retrieved 4/7 statements.
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
    var_3 = 'py38'
    var_4 = var_2.known_standard_library

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/path/'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_config_constructor_with_config_object. Retrieved 5/6 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 3/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 5/10 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 8/11 statements.


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
    var_0 = '[isort]\nline_length = 88\n'
    var_1 = 'test_settings.cfg'
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = var_3.line_length
    assert var_4 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_project'
    var_1 = True
    var_2 = '[isort]\nline_length = 100\n'
    var_3 = {}
    var_4 = module_0.Config(settings_path=var_2, **var_3)
    var_5 = var_4.line_length
    assert var_5 == 100
    var_6 = 'test_project/.isort.cfg'

import isort.settings as module_0

def test_case_0():
    var_0 = 120
    var_1 = '    '
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.line_length
    assert var_6 == 120
    var_7 = var_5.indent
    assert var_7 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 88
    var_5 = var_3.multi_line_output
    assert var_5 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'baz'
    var_2 = [var_0, var_1]
    var_3 = 'STDLIB'
    var_4 = 'FOO'
    var_5 = [var_3, var_4]
    var_6 = 'known_foo'
    var_7 = 'sections'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.Config(**var_8)
    var_10 = 'foo'
    var_11 = [var_0, var_1]
    var_12 = frozenset(var_11)
    var_13 = {var_10: var_12}
    var_14 = var_9.known_other
    var_15 = bool(var_9.known_other == var_13)
    assert var_15 is True
    var_16 = 'FOO'
    var_17 = bool('FOO' in var_9.sections)
    assert var_17 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Bar Imports'
    var_1 = 'STDLIB'
    var_2 = 'FOO'
    var_3 = [var_1, var_2]
    var_4 = 'import_heading_foo'
    var_5 = 'sections'
    var_6 = {var_4: var_0, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.import_headings
    var_9 = bool(var_7.import_headings == {'foo': 'Bar Imports'})
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'End of Bar Imports'
    var_1 = 'STDLIB'
    var_2 = 'FOO'
    var_3 = [var_1, var_2]
    var_4 = 'import_footer_foo'
    var_5 = 'sections'
    var_6 = {var_4: var_0, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.import_footers
    var_9 = bool(var_7.import_footers == {'foo': 'End of Bar Imports'})
    assert var_9 is True

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
    var_0 = False
    var_1 = True
    var_2 = 'force_single_line'
    var_3 = 'quiet'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.force_single_line
    assert var_6 is False

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
    var_0 = 'test_project/src'
    var_1 = True
    var_2 = 'test_project'
    var_3 = 'src'
    var_4 = [var_3]
    var_5 = 'src_paths'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(settings_path=var_2, **var_6)
    var_8 = var_7.src_paths
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7.src_paths[0].name
    assert var_10 == 'src'

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
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = var_3._known_patterns
    assert var_4 is None
    var_5 = {}
    var_6 = module_0.Config(config=var_1, **var_5)
    var_7 = var_6._section_comments
    assert var_7 is None
    var_8 = {}
    var_9 = module_0.Config(config=var_1, **var_8)
    var_10 = var_9._section_comments_end
    assert var_10 is None
    var_11 = {}
    var_12 = module_0.Config(config=var_1, **var_11)
    var_13 = var_12._skips
    assert var_13 is None
    var_14 = {}
    var_15 = module_0.Config(config=var_1, **var_14)
    var_16 = var_15._skip_globs
    assert var_16 is None
    var_17 = {}
    var_18 = module_0.Config(config=var_1, **var_17)
    var_19 = var_18._sorting_function
    assert var_19 is None



# Parsed testcases at query #13
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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.min.js'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

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
    var_2 = '/dev/null'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_tab_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 4/8 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_unsupported_config. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_nonexistent_sort_order. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
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
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
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
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_skipped_with_skipped_file. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_with_non_skipped_file. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_skipped_directory. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_non_skipped_directory. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_non_matching_skip_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_with_skip_gitignore. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_with_non_existent_file. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_with_directory. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_with_symlink. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_with_git_ls_files. Retrieved 4/7 statements.
# Partially parsed test_is_skipped_with_git_ls_files_included. Retrieved 3/6 statements.


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
    var_0 = 'other'
    var_1 = {var_0}
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'tests/test.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = {var_0}
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.pyc'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = '*.pyc'
    var_1 = {var_0}
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.py'
    var_6 = [var_5]

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'non_existent.py'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some_directory'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some_symlink'
    var_3 = [var_2]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/test/file.py'
    var_5 = '/test/other.py'
    var_6 = [var_5]

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/test/file.py'
    var_5 = [var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__find_config_finds_and_returns_config_data. Retrieved 7/10 statements.
# Partially parsed test__find_config_stops_search_on_stop_dir. Retrieved 6/12 statements.
# Partially parsed test__find_config_returns_config_from_parent_dir. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/nonexistent/path', {}))
    assert var_2 is True

import posixpath as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '[section]\nkey = value'
    var_1 = 'test_config.ini'
    var_2 = module_0.abspath(var_1)
    var_3 = module_0.dirname(var_2)
    var_4 = module_1._find_config(var_3)
    var_5 = module_0.abspath(var_1)
    var_6 = module_0.dirname(var_5)
    var_7 = var_4[0]
    var_8 = bool(var_4[0] == var_6)
    assert var_8 is True
    var_9 = 'key'
    var_10 = bool('key' in var_4[1])
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir/.git'
    var_1 = True
    var_2 = '[section]\nkey = value'
    var_3 = 'test_dir'
    var_4 = module_0._find_config(var_3)
    var_5 = bool(var_4 == ('test_dir', {}))
    assert var_5 is True
    var_6 = 'test_dir/test_config.ini'

import isort.settings as module_0

def test_case_0():
    var_0 = 'parent/child'
    var_1 = True
    var_2 = '[section]\nkey = value'
    var_3 = module_0._find_config(var_2)
    var_4 = var_3[0]
    assert var_4 == 'parent'
    var_5 = 'key'
    var_6 = bool('key' in var_3[1])
    assert var_6 is True
    var_7 = 'parent/config.ini'
    var_8 = 'parent'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__get_config_data_with_toml_file. Retrieved 6/9 statements.
# Partially parsed test__get_config_data_with_editorconfig_file. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_with_ini_file. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_with_empty_sections. Retrieved 4/7 statements.
# Partially parsed test__get_config_data_with_known_prefix. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_with_force_grid_wrap. Retrieved 5/8 statements.
# Partially parsed test__get_config_data_with_comment_prefix. Retrieved 5/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = (var_1, var_2)
    var_4 = "[section1]\nkey1 = 'value1'\nkey2 = 123\n[section2]\nkey3 = true"
    var_5 = module_0._get_config_data(var_0, var_3)
    var_6 = bool(var_5 == {'key1': 'value1', 'key2': 123, 'key3': True, 'source': var_0})
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = bool(var_4 == {'indent': '    ', 'line_length': 88, 'source': var_0})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nkey1 = value1\nkey2 = 123'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = bool(var_4 == {'key1': 'value1', 'key2': 123, 'source': var_0})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = ()
    var_2 = '[section1]\nkey1 = value1'
    var_3 = module_0._get_config_data(var_0, var_1)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nknown_key = value1,value2'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = bool(var_4 == {'known_key': {'value1', 'value2'}, 'source': var_0})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '[section1]\nforce_grid_wrap = false'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = bool(var_4 == {'force_grid_wrap': 0, 'source': var_0})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = "[section1]\ncomment_prefix = '#'"
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = bool(var_4 == {'comment_prefix': '#', 'source': var_0})
    assert var_5 is True



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

# Partially parsed test_predicate_at_line_80_evaluates_to_true. Retrieved 21/28 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = 'indent_style'
    var_4 = 'indent_size'
    var_5 = 'max_line_length'
    var_6 = 'force_grid_wrap'
    var_7 = 'comment_prefix'
    var_8 = 'space'
    var_9 = '4'
    var_10 = '88'
    var_11 = 'true'
    var_12 = "'# '"
    var_13 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_12}
    var_14 = '[*.{py}]\n'
    var_15 = 'indent_style = space\n'
    var_16 = 'indent_size = 4\n'
    var_17 = 'max_line_length = 88\n'
    var_18 = 'force_grid_wrap = true\n'
    var_19 = "comment_prefix = '# '\n"
    var_20 = module_0._get_config_data(var_0, var_2)
    var_21 = var_20['force_grid_wrap']
    assert var_21 == 2



# Parsed testcases at query #20
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
    var_0 = 'a,,b'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b'])
    assert var_2 is True

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
    var_0 = '   '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a'])
    assert var_2 is True



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 3/6 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 5/10 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_profile. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 4/7 statements.
# Partially parsed test_config_constructor_with_unsupported_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_formatter_plugin. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_invalid_sort_order. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 120
    var_2 = {var_0: var_1}

import isort.settings as module_0

def test_case_0():
    var_0 = '[isort]\nline_length = 88\n'
    var_1 = 'test_settings.cfg'
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = var_3.line_length
    assert var_4 == 88

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_project'
    var_1 = True
    var_2 = '[isort]\nline_length = 100\n'
    var_3 = {}
    var_4 = module_0.Config(settings_path=var_2, **var_3)
    var_5 = var_4.line_length
    assert var_5 == 100
    var_6 = 'test_project/.isort.cfg'

def test_case_0():
    var_0 = 'profile'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'profile'
    var_1 = 'invalid_profile'
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
    var_0 = 'virtual_env'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'virtual_env'
    var_4 = bool(not var_1)
    assert var_4 is True

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



# Parsed testcases at query #23
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
    var_0 = '4'
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
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

import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'quiet'
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'known_foo'
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'import_heading_foo'
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'import_footer_foo'
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_tab_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 4/8 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_config_object. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
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
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
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
    var_0 = 'force_single_line'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = 120
    var_4 = {var_0: var_3}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_known_other_section_not_in_sections. Retrieved 18/30 statements.


def test_case_0():
    var_0 = 'sections'
    var_1 = 'known_custom'
    var_2 = 'quiet'
    var_3 = 'STANDARD_LIB'
    var_4 = 'THIRD_PARTY'
    var_5 = (var_3, var_4)
    var_6 = 'custom_module'
    var_7 = {var_6}
    var_8 = False
    var_9 = {var_0: var_5, var_1: var_7, var_2: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = {}
    var_13 = 'known_'
    var_14 = len(var_13)
    var_15 = f'known_{maps_to_section.lower()}'
    var_16 = 'sections'
    var_17 = ()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True
    var_4 = 'file.pyi'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'txt'
    var_1 = [var_0]
    var_2 = 'blocked_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'file.txt'
    var_6 = var_4.is_supported_filetype(var_5)
    assert var_6 is False

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
    var_2 = 'script.sh'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_with_no_arguments. Retrieved 2/9 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/8 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.
# Failed to parse test_config_constructor_with_src_paths.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = 'src'
    var_4 = var_1.src_paths

def test_case_0():
    var_0 = '[isort]\nline_length=79\n'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.isort]\nline_length=88\n'

import isort.settings as module_0

def test_case_0():
    var_0 = 79
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
    var_0 = 79
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.line_length
    assert var_4 == 88
    var_5 = var_3.multi_line_output
    assert var_5 == 3

import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

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
    var_0 = 'nonexistent'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'force_grid_wrap'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'force_grid_wrap'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)

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
    var_0 = 'Bar'
    var_1 = 'import_footer_foo'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers
    var_5 = bool(var_3.import_footers == {'foo': 'Bar'})
    assert var_5 is True

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
    var_0 = 'natural'
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_supported_filetype_fifo. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_shebang. Retrieved 4/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'txt'
    var_1 = [var_0]
    var_2 = 'blocked_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test.txt'
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
    var_2 = 'test_fifo'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent_file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = '#!/usr/bin/env python3\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test_script'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_indent_tab. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_known_sections. Retrieved 8/9 statements.
# Partially parsed test_config_constructor_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_deprecated_options. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_unsupported_options. Retrieved 3/5 statements.
# Partially parsed test_config_constructor_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_skip_gitignore. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 4/7 statements.
# Partially parsed test_config_constructor_with_directory. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_quiet. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 100
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
    var_0 = 'indent'
    var_1 = '4'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
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
    var_1 = 'venv'
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
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'skip_gitignore'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = [var_1]

def test_case_0():
    var_0 = 'directory'
    var_1 = '.'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'quiet'
    var_1 = True
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_indent_lower_equals_tab. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'indent'
    var_1 = 'tab'
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = str(var_3)
    var_5 = "'"
    var_6 = '"'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_123_predicate_true. Retrieved 18/23 statements.


def test_case_0():
    var_0 = 'quiet'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'sections'
    var_4 = 'known_custom'
    var_5 = 'CUSTOM'
    var_6 = (var_5,)
    var_7 = 'module'
    var_8 = {var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_3: var_6, var_4: var_9}
    var_11 = 'known_custom'
    var_12 = [var_7]
    var_13 = 'known_'
    var_14 = len(var_13)
    var_15 = var_11[var_14:]
    var_16 = {}
    var_17 = ()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_supported_filetype_fifo. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_with_shebang. Retrieved 4/7 statements.
# Partially parsed test_is_supported_filetype_without_shebang. Retrieved 4/7 statements.


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
    var_2 = 'test_fifo'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = b'#!/usr/bin/env python\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test_file'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = b"print('hello')\n"
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test_file'
    var_4 = var_2.is_supported_filetype(var_3)
    assert var_4 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_with_config_parameter. Retrieved 4/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = 'py'
    var_5 = ''
    var_6 = var_3.py_version
    var_7 = var_3._known_patterns
    assert var_7 is None
    var_8 = var_3._section_comments
    assert var_8 is None
    var_9 = var_3._section_comments_end
    assert var_9 is None
    var_10 = var_3._skips
    assert var_10 is None
    var_11 = var_3._skip_globs
    assert var_11 is None
    var_12 = var_3._sorting_function
    assert var_12 is None

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
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '
    var_5 = var_3._known_patterns
    assert var_5 is None
    var_6 = var_3._section_comments
    assert var_6 is None
    var_7 = var_3._section_comments_end
    assert var_7 is None
    var_8 = var_3._skips
    assert var_8 is None
    var_9 = var_3._skip_globs
    assert var_9 is None
    var_10 = var_3._sorting_function
    assert var_10 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
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

import isort.settings as module_0

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'known_future_library'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4._known_patterns
    assert var_5 is None
    var_6 = var_4._section_comments
    assert var_6 is None
    var_7 = var_4._section_comments_end
    assert var_7 is None
    var_8 = var_4._skips
    assert var_8 is None
    var_9 = var_4._skip_globs
    assert var_9 is None
    var_10 = var_4._sorting_function
    assert var_10 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '__future__'
    var_1 = 'import_heading_future'
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

import isort.settings as module_0

def test_case_0():
    var_0 = '__future__'
    var_1 = 'import_footer_future'
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
    var_9 = var_8._known_patterns
    assert var_9 is None
    var_10 = var_8._section_comments
    assert var_10 is None
    var_11 = var_8._section_comments_end
    assert var_11 is None
    var_12 = var_8._skips
    assert var_12 is None
    var_13 = var_8._skip_globs
    assert var_13 is None
    var_14 = var_8._sorting_function
    assert var_14 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4._known_patterns
    assert var_5 is None
    var_6 = var_4._section_comments
    assert var_6 is None
    var_7 = var_4._section_comments_end
    assert var_7 is None
    var_8 = var_4._skips
    assert var_8 is None
    var_9 = var_4._skip_globs
    assert var_9 is None
    var_10 = var_4._sorting_function
    assert var_10 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'black'
    var_1 = 'formatter'
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

import isort.settings as module_0

def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_initialization_with_config_parameter. Retrieved 2/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '38'
    var_4 = var_2.indent
    assert var_4 == '    '
    var_5 = var_2.line_length
    assert var_5 == 88
    var_6 = var_2._known_patterns
    assert var_6 is None
    var_7 = var_2._section_comments
    assert var_7 is None
    var_8 = var_2._section_comments_end
    assert var_8 is None
    var_9 = var_2._skips
    assert var_9 is None
    var_10 = var_2._skip_globs
    assert var_10 is None
    var_11 = var_2._sorting_function
    assert var_11 is None



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'custom_module'
    var_1 = [var_0]
    var_2 = 'known_custom_section'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'custom_section'
    var_6 = [var_0]
    var_7 = frozenset(var_6)
    var_8 = {var_5: var_7}
    var_9 = var_4.known_other
    var_10 = bool(var_4.known_other == var_8)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = './file1'
    var_2 = 'dir1/'
    var_3 = 'file2'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = bool(var_5 == {'/home/user/./file1', '/home/user/dir1/', '/home/user/file2'})
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = '/absolute/file1'
    var_2 = '/absolute/dir1/'
    var_3 = [var_1, var_2]
    var_4 = module_0._abspaths(var_0, var_3)
    var_5 = bool(var_4 == {'/absolute/file1', '/absolute/dir1/'})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = './file1'
    var_2 = '/absolute/file2'
    var_3 = 'dir1/'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0._abspaths(var_0, var_4)
    var_6 = bool(var_5 == {'/home/user/./file1', '/absolute/file2', '/home/user/dir1/'})
    assert var_6 is True

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
    var_1 = './file1'
    var_2 = [var_1, var_1]
    var_3 = module_0._abspaths(var_0, var_2)
    var_4 = bool(var_3 == {'/home/user/./file1'})
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__find_config_returns_config_data_when_config_file_exists. Retrieved 1/5 statements.
# Partially parsed test__find_config_stops_search_on_stop_dir. Retrieved 2/8 statements.
# Partially parsed test__find_config_returns_config_data_for_valid_config_file. Retrieved 2/7 statements.
# Partially parsed test__find_config_handles_exception_during_config_parsing. Retrieved 2/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/non/existent/path', {}))
    assert var_2 is True

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 'stop_dir'
    var_1 = {}

def test_case_0():
    var_0 = 'valid_config_file'
    var_1 = 1

def test_case_0():
    var_0 = 'malformed_config_file'
    var_1 = {}



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'line_length': 88, 'indent': '    '})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'indent': '    ', 'line_length': 88})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test_config.ini'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'line_length': 88, 'indent': '    '})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'empty_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'multi_section_config.toml'
    var_1 = 'tool.black'
    var_2 = 'tool.isort'
    var_3 = (var_1, var_2)
    var_4 = module_0._get_config_data(var_0, var_3)
    var_5 = bool(var_4 == {'source': var_0, 'line_length': 88, 'indent': '    ', 'multi_line_output': 3})
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'bool_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'skip_string_normalization': True, 'verbose': False})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'list_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'include': ('tests/', 'src/'), 'exclude': ('build/', 'dist/')})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'force_grid_wrap_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'force_grid_wrap': 2})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'comment_prefix_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'comment_prefix': '#'})
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'abspaths_config.toml'
    var_1 = 'tool.black'
    var_2 = (var_1,)
    var_3 = module_0._get_config_data(var_0, var_2)
    var_4 = bool(var_3 == {'source': var_0, 'src_paths': {'/absolute/path1', '/absolute/path2'}})
    assert var_4 is True



# Parsed testcases at query #8
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
    var_0 = 'a, , b'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b'])
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '  a  ,  b  '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b'])
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_supported_filetype_returns_true_for_supported_extension. Retrieved 4/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'test.py'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'empty_file.cfg'
    var_1 = False
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)
    var_5 = var_4._known_patterns
    assert var_5 is None
    var_6 = var_4._section_comments
    assert var_6 is None
    var_7 = var_4._section_comments_end
    assert var_7 is None
    var_8 = var_4._skips
    assert var_8 is None
    var_9 = var_4._skip_globs
    assert var_9 is None
    var_10 = var_4._sorting_function
    assert var_10 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'known_prefix_key'
    var_1 = 'known_'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test__post_init__vertical_grid_grouped_no_comma_converted.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.py_version
    assert var_1 == 'py3'

import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0._Config(var_0)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = var_0.known_standard_library
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

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



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(not var_1._known_patterns)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_80_evaluates_to_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'force_grid_wrap'
    var_1 = 'true'
    var_2 = 'false'
    var_3 = 0
    var_4 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_config_init_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_profile. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_quiet. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_indent. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_known_sections. Retrieved 7/8 statements.
# Partially parsed test_config_init_with_import_headings. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_import_footers. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_skips. Retrieved 9/10 statements.
# Partially parsed test_config_init_with_skip_globs. Retrieved 9/10 statements.
# Partially parsed test_config_init_with_sort_order. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_formatter. Retrieved 3/4 statements.
# Partially parsed test_config_init_with_deprecated_options. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_unsupported_config. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_nonexistent_profile. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_nonexistent_formatter. Retrieved 3/5 statements.
# Partially parsed test_config_init_with_nonexistent_sort_order. Retrieved 3/5 statements.


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
    var_0 = 'skip'
    var_1 = 'extend_skip'
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = 'test2.py'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_2, var_4]
    var_8 = frozenset(var_7)

def test_case_0():
    var_0 = 'skip_glob'
    var_1 = 'extend_skip_glob'
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = 'test2.py'
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_2, var_4]
    var_8 = frozenset(var_7)

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'natural'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'black'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'deprecated_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'profile'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'formatter'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'sort_order'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = bool(not var_1._known_patterns)
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_config_returns_empty_dict_when_config_file_is_invalid. Retrieved 3/5 statements.
# Partially parsed test_find_config_returns_config_data_when_valid_config_file_exists. Retrieved 3/5 statements.
# Partially parsed test_find_config_stops_search_on_stop_directory. Retrieved 3/4 statements.
# Partially parsed test_find_config_searches_parent_directories. Retrieved 3/5 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/non/existent/path', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'invalid content'
    var_1 = '/tmp'
    var_2 = module_0._find_config(var_1)
    var_3 = bool(var_2 == ('/tmp', {}))
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '[section]\nkey = value'
    var_1 = '/tmp'
    var_2 = module_0._find_config(var_1)
    var_3 = var_2[0]
    assert var_3 == '/tmp'
    var_4 = 'key'
    var_5 = bool('key' in var_2[1])
    assert var_5 is True
    var_6 = var_2[1]['key']
    assert var_6 == 'value'

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/stop_dir'
    var_1 = True
    var_2 = module_0._find_config(var_0)
    var_3 = bool(var_2 == ('/tmp/stop_dir', {}))
    assert var_3 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '[section]\nkey = value'
    var_1 = '/tmp/child'
    var_2 = module_0._find_config(var_1)
    var_3 = var_2[0]
    assert var_3 == '/tmp'
    var_4 = 'key'
    var_5 = bool('key' in var_2[1])
    assert var_5 is True
    var_6 = var_2[1]['key']
    assert var_6 == 'value'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_directory_defaults_to_config_source_directory. Retrieved 6/10 statements.


import posixpath as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = '/path/to/config/file'
    var_2 = {var_0: var_1}
    var_3 = 'directory'
    var_4 = None
    var_5 = var_2[var_0]
    var_6 = module_0.dirname(var_5)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_section'
    var_2 = 'non_existent_section'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'test.ini'
    var_1 = 'section1'
    var_2 = (var_1,)
    var_3 = '.toml'



# Parsed testcases at query #22
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
    var_0 = 'test.cfg'
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
    var_0 = '4'
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



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent_file.py'
    var_3 = var_1.is_supported_filetype(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



