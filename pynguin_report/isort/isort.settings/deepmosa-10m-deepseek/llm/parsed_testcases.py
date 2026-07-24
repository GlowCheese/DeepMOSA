####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory
    var_4 = bool(var_2.directory is not None)
    assert var_4 is True


def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.directory
    var_4 = bool(var_2.directory is not None)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = var_3.directory
    var_5 = bool(var_3.directory == var_1.directory)
    assert var_5 is True


def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'


def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '


def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'


def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'known_mysection'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'mysection'
    var_6 = bool('mysection' in var_4.known_other)
    assert var_6 is True


def test_case_0():
    var_0 = 'My Section'
    var_1 = 'import_heading_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['mysection']
    assert var_4 == 'My Section'


def test_case_0():
    var_0 = 'End of My Section'
    var_1 = 'import_footer_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['mysection']
    assert var_4 == 'End of My Section'


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


def test_case_0():
    var_0 = 'color_output'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(var_3.formatting_function is not None)
    assert var_5 is True


def test_case_0():
    var_0 = False
    var_1 = 'force_alphabetical_sort'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.force_alphabetical_sort
    assert var_4 is False


def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'py310'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = {}
    var_5 = module_0.Config(config=var_3, **var_4)
    var_6 = var_5.py_version
    assert var_6 == '310'


def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.skip_gitignore
    assert var_4 is True


def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sort_order
    assert var_4 == 'natural'


def test_case_0():
    var_0 = 'nonexistent.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory
    var_4 = bool(var_2.directory is not None)
    assert var_4 is True


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'invalid_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_config_with_valid_toml_file. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_valid_editorconfig_file. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_valid_ini_file. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_no_config_file. Retrieved 2/5 statements.
# Partially parsed test_find_config_with_stop_directory. Retrieved 5/15 statements.
# Partially parsed test_find_config_with_max_search_depth. Retrieved 10/23 statements.
# Partially parsed test_find_config_with_invalid_config_file. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_extension_specific_editorconfig. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_multiple_extension_editorconfig. Retrieved 4/11 statements.
# Partially parsed test_find_config_with_off_max_line_length. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '/tmp/test_config'
    var_1 = True
    var_2 = b'[tool.isort]\nline_length = 100\n'
    var_3 = 'pyproject.toml'
    var_4 = [var_3]

def test_case_0():
    var_0 = '/tmp/test_config'
    var_1 = True
    var_2 = '[*.py]\nindent_style = space\nindent_size = 2\n'
    var_3 = '.editorconfig'
    var_4 = [var_3]

def test_case_0():
    var_0 = '/tmp/test_config'
    var_1 = True
    var_2 = '[isort]\nline_length = 120\n'
    var_3 = '.isort.cfg'
    var_4 = [var_3]

def test_case_0():
    var_0 = '/tmp/test_no_config'
    var_1 = True

def test_case_0():
    var_0 = '/tmp/test_stop'
    var_1 = True
    var_2 = '.git'
    var_3 = [var_2]
    var_4 = '.isort.cfg'
    var_5 = [var_4]
    var_6 = '[isort]\nline_length = 80\n'

def test_case_0():
    var_0 = '/tmp/test_depth'
    var_1 = True
    var_2 = 'sub1'
    var_3 = 'sub2'
    var_4 = 'sub3'
    var_5 = 'sub4'
    var_6 = 'sub5'
    var_7 = 'sub6'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '.isort.cfg'
    var_10 = [var_9]
    var_11 = '[isort]\nline_length = 90\n'

def test_case_0():
    var_0 = '/tmp/test_invalid'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = [var_2]
    var_4 = 'invalid toml content'

def test_case_0():
    var_0 = '/tmp/test_ext'
    var_1 = True
    var_2 = '[*.{py}]\nindent_style = tab\nindent_size = 4\n'
    var_3 = '.editorconfig'
    var_4 = [var_3]

def test_case_0():
    var_0 = '/tmp/test_multi_ext'
    var_1 = True
    var_2 = '[*.{py,js}]\nmax_line_length = 120\n'
    var_3 = '.editorconfig'
    var_4 = [var_3]

def test_case_0():
    var_0 = '/tmp/test_off'
    var_1 = True
    var_2 = '[*.py]\nmax_line_length = off\n'
    var_3 = '.editorconfig'
    var_4 = [var_3]
    var_5 = 'inf'
    var_6 = float(var_5)



# Parsed testcases at query #3
#--------------------------





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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__get_config_data_with_toml_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_ini_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_editorconfig_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_editorconfig_file_tab_indent. Retrieved 5/13 statements.
# Partially parsed test__get_config_data_with_editorconfig_file_extension_section. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_force_grid_wrap_string. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_force_grid_wrap_boolean_backwards_compat. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_comment_prefix. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_known_prefix. Retrieved 4/14 statements.
# Partially parsed test__get_config_data_with_tuple_type. Retrieved 4/14 statements.
# Partially parsed test__get_config_data_with_empty_settings. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '\n[tool.black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[black]\nline_length = 100\nskip_string_normalization = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\nroot = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 80\n'
    var_1 = '*'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*]\nindent_style = tab\ntab_width = 4\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '\n[*.{py,pyi}]\nindent_size = 4\nline_length = 79\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[black]\nforce_grid_wrap = 3\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[black]\nforce_grid_wrap = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[black]\ncomment_prefix = "# "\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[black]\nknown_third_party = requests,django\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'known_third_party'
    var_4 = 'requests'
    var_5 = 'django'

def test_case_0():
    var_0 = '\n[black]\npyproject_include = a,b,c\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'pyproject_include'

def test_case_0():
    var_0 = '\n[other_section]\nkey = value\n'
    var_1 = 'black'
    var_2 = (var_1,)



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = ' b '
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['hello'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a,b,c'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = ' a , b , c '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a\nb\nc'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a,b\nc'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True


def test_case_0():
    var_0 = ' , , '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True


def test_case_0():
    var_0 = []
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True


def test_case_0():
    var_0 = ''
    var_1 = 'a'
    var_2 = ' '
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._as_list(var_4)
    var_6 = bool(var_5 == ['a', 'b'])
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_27_true_for_editorconfig_with_wildcard_extension. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '[*.{py,js}]\nindent_style = space\nindent_size = 4\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)
    var_3 = 'indent_style'
    var_4 = 'indent'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_config_constructor_with_config_parameter. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_with_config_overrides. Retrieved 3/4 statements.
# Partially parsed test_config_constructor_with_settings_file. Retrieved 1/7 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/7 statements.



def test_case_0():
    var_0 = module_0._Config()
    var_1 = {}
    var_2 = module_0.Config(config=var_0, **var_1)
    var_3 = var_2.py_version
    assert var_3 == '310'


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


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '


def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'


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


def test_case_0():
    var_0 = 'My Section'
    var_1 = 'import_heading_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['mysection']
    assert var_4 == 'My Section'


def test_case_0():
    var_0 = 'Footer'
    var_1 = 'import_footer_mysection'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['mysection']
    assert var_4 == 'Footer'


def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_setting'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = 'force_sort_within_sections'
    var_2 = 'quiet'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = dir(var_4)
    var_6 = 'force_sort_within_sections'
    var_7 = bool('force_sort_within_sections' not in var_5)
    assert var_7 is True


def test_case_0():
    var_0 = 'example'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(var_3.formatting_function is not None)
    assert var_5 is True


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


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


def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function


def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function


def test_case_0():
    var_0 = 'custom'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.skip_gitignore
    assert var_4 is True


def test_case_0():
    var_0 = 'venv'
    var_1 = [var_0]
    var_2 = 'tests'
    var_3 = [var_2]
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'venv'
    var_9 = bool('venv' in var_7.skips)
    assert var_9 is True
    var_10 = 'tests'
    var_11 = bool('tests' in var_7.skips)
    assert var_11 is True


def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = '__pycache__'
    var_3 = [var_2]
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '*.pyc'
    var_9 = bool('*.pyc' in var_7.skip_globs)
    assert var_9 is True
    var_10 = '__pycache__'
    var_11 = bool('__pycache__' in var_7.skip_globs)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_profile_name_not_in_profiles_and_plugin_exists. Retrieved 12/22 statements.



def test_case_0():
    var_0 = {}
    var_1 = 'test_profile'
    var_2 = 'MockPlugin'
    var_3 = ()
    var_4 = 'load'
    var_5 = {}
    var_6 = lambda : var_5
    var_7 = {var_4: var_6}
    var_8 = [var_2, var_3, var_7]
    var_9 = 'MockEntryPoint'
    var_10 = ()
    var_11 = 'name'
    var_12 = 'profile'
    var_13 = {var_12: var_1}
    var_14 = module_0.Config(**var_13)



# Parsed testcases at query #9
#--------------------------





def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = module_0.Config(var_0, var_0, var_1, **var_2)
    var_4 = 'directory'
    var_5 = bool('directory' not in var_3._combined_config)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = 'test.editorconfig'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = '[*.py]\nindent_style = space\nindent_size = 4\n'
    var_4 = module_0._get_config_data(var_0, var_2)
    var_5 = 'indent'
    var_6 = bool('indent' in var_4)
    assert var_6 is True
    var_7 = var_4['indent']
    assert var_7 == '    '



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__get_config_data_toml. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_ini. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_indent_spaces. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_indent_tabs. Retrieved 5/12 statements.
# Partially parsed test__get_config_data_editorconfig_wildcard_extension. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_numeric. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_true. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_false. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_comment_prefix_stripping. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_known_prefix_abspaths. Retrieved 6/20 statements.
# Partially parsed test__get_config_data_bool_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_tuple_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_frozenset_conversion. Retrieved 7/14 statements.
# Partially parsed test__get_config_data_empty_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_multiple_sections. Retrieved 4/11 statements.
# Partially parsed test__get_config_data_nested_toml. Retrieved 3/10 statements.


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
    var_0 = '[*]\nindent_style = tab\ntab_width = 4\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*{py,pyi}]\nline_length = 120\n'
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
    var_0 = '[black]\nextend-exclude = foo/, bar/\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'foo/'
    var_4 = 'bar/'
    var_5 = 'extend-exclude'

def test_case_0():
    var_0 = '[black]\nskip_magic_trailing_comma = yes\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nknown_third_party = requests, pytest\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\npyink_extensions = foo, bar\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3, var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '[black]\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\n\n[pyink]\nline_length = 120\n'
    var_1 = 'black'
    var_2 = 'pyink'
    var_3 = (var_1, var_2)

def test_case_0():
    var_0 = b'[tool.black.format]\nline_length = 88\n'
    var_1 = 'tool.black.format'
    var_2 = (var_1,)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_skipped_skips_exact_path_match. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_skips_exact_normalized_path. Retrieved 5/9 statements.
# Partially parsed test_is_skipped_skips_parent_folder. Retrieved 5/9 statements.
# Partially parsed test_is_skipped_skips_by_glob. Retrieved 4/6 statements.
# Partially parsed test_is_skipped_skips_by_glob_with_leading_slash. Retrieved 5/8 statements.
# Partially parsed test_is_skipped_does_not_skip_when_no_match. Retrieved 6/8 statements.
# Partially parsed test_is_skipped_skips_non_existent_path. Retrieved 2/4 statements.
# Partially parsed test_is_skipped_skips_gitignored_file_when_skip_gitignore_true. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_does_not_skip_gitignored_file_when_skip_gitignore_false. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_skips_dot_git_folder. Retrieved 3/5 statements.
# Partially parsed test_is_skipped_uses_extend_skip. Retrieved 5/7 statements.
# Partially parsed test_is_skipped_uses_extend_skip_glob. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'my_file.py'
    var_1 = [var_0]
    var_2 = 'skips'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_0]


def test_case_0():
    var_0 = 'my_file.py'
    var_1 = [var_0]
    var_2 = 'skips'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'some_folder'
    var_6 = [var_5]
    var_7 = '..'


def test_case_0():
    var_0 = 'my_folder'
    var_1 = [var_0]
    var_2 = 'skips'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_0]
    var_6 = 'nested'
    var_7 = 'file.py'


def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'module.pyc'
    var_6 = [var_5]


def test_case_0():
    var_0 = '*.pyc'
    var_1 = [var_0]
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'subdir'
    var_6 = [var_5]
    var_7 = 'module.pyc'


def test_case_0():
    var_0 = 'other.py'
    var_1 = [var_0]
    var_2 = '*.pyc'
    var_3 = [var_2]
    var_4 = 'skips'
    var_5 = 'skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'my_file.py'
    var_9 = [var_8]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'non_existent_file.py'
    var_3 = [var_2]


def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'ignored.py'
    var_5 = [var_4]


def test_case_0():
    var_0 = False
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'ignored.py'
    var_5 = [var_4]


def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '.git'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'skip1'
    var_1 = {var_0}
    var_2 = 'skip2'
    var_3 = {var_2}
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2]


def test_case_0():
    var_0 = '*.tmp'
    var_1 = {var_0}
    var_2 = '*.log'
    var_3 = {var_2}
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'file.log'
    var_9 = [var_8]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 2/3 statements.
# Failed to parse test_config_constructor_with_settings_path.
# Partially parsed test_config_constructor_with_src_paths. Retrieved 5/7 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 2/3 statements.



def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True


def test_case_0():
    var_0 = 'test.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = {}
    var_3 = module_0.Config(config=var_1, **var_2)
    var_4 = var_3.py_version
    var_5 = bool(var_3.py_version == var_1.py_version)
    assert var_5 is True


def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sources
    var_5 = str(var_4)
    var_6 = 'black'
    var_7 = bool('black' in var_5)
    assert var_7 is True


def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '


def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'


def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'known_custom'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'custom'
    var_6 = bool('custom' in var_4.known_other)
    assert var_6 is True


def test_case_0():
    var_0 = 'Standard Library'
    var_1 = 'import_heading_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['stdlib']
    assert var_4 == 'Standard Library'


def test_case_0():
    var_0 = 'End Standard Library'
    var_1 = 'import_footer_stdlib'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['stdlib']
    assert var_4 == 'End Standard Library'


def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.src_paths


def test_case_0():
    var_0 = 'color_output'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = bool(var_3.formatting_function is not None)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = 'force_alphabetical_sort'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'force_alphabetical_sort'
    var_5 = bool('force_alphabetical_sort' not in var_3.__dict__)
    assert var_5 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'requests'
    var_1 = [var_0]
    var_2 = 'mylib'
    var_3 = [var_2]
    var_4 = 'known_third_party'
    var_5 = 'known_first_party'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'requests'
    var_9 = bool('requests' in var_7.known_third_party)
    assert var_9 is True
    var_10 = 'mylib'
    var_11 = bool('mylib' in var_7.known_first_party)
    assert var_11 is True


def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'known_custom'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'CUSTOM'
    var_8 = bool('CUSTOM' in var_6.sections)
    assert var_8 is True


def test_case_0():
    var_0 = '/tmp'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    assert var_4 == '/tmp'


def test_case_0():
    var_0 = False
    var_1 = 'test'
    var_2 = [var_1]
    var_3 = 'quiet'
    var_4 = 'known_custom'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = var_6.quiet
    assert var_7 is False


def test_case_0():
    var_0 = 'empty.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_supported_filetype_with_supported_extension. Retrieved 5/6 statements.
# Partially parsed test_is_supported_filetype_with_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_shebang. Retrieved 4/8 statements.
# Partially parsed test_is_supported_filetype_with_unknown_extension_and_no_shebang. Retrieved 4/8 statements.
# Partially parsed test_is_supported_filetype_with_fifo_file. Retrieved 3/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = 'test.py'
    var_5 = var_1.is_supported_filetype(var_4)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'log'
    var_3 = 'error.log'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b'#!/usr/bin/env python\n'
    var_3 = 'script'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = b"print('Hello')"
    var_3 = 'script'
    var_4 = var_1.is_supported_filetype(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py~'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'fifo'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'missing.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_with_settings_path. Retrieved 2/3 statements.
# Partially parsed test_config_constructor_with_empty_settings_file. Retrieved 2/3 statements.



def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True


def test_case_0():
    var_0 = 'test.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = '.'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'quiet'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(config=var_1, **var_4)
    var_6 = var_5.quiet
    assert var_6 is True


def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    assert var_4 == 'black'


def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '


def test_case_0():
    var_0 = '2'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '  '


def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'


def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = 'known_custom'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'custom'
    var_6 = bool('custom' in var_4.known_other)
    assert var_6 is True


def test_case_0():
    var_0 = 'Custom Imports'
    var_1 = 'import_heading_custom'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_headings['custom']
    assert var_4 == 'Custom Imports'


def test_case_0():
    var_0 = 'End Custom Imports'
    var_1 = 'import_footer_custom'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.import_footers['custom']
    assert var_4 == 'End Custom Imports'


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


def test_case_0():
    var_0 = 'color'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatting_function
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = False
    var_1 = 'force_alphabetical_sort'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.force_alphabetical_sort
    assert var_4 is False


def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'empty.ini'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = '/invalid/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'invalid_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_sort'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'known_standard_library'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'os'
    var_6 = bool('os' in var_4.known_standard_library)
    assert var_6 is True


def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = (var_0,)
    var_2 = 'mypackage'
    var_3 = [var_2]
    var_4 = 'sections'
    var_5 = 'known_custom'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'custom'
    var_9 = bool('custom' in var_7.known_other)
    assert var_9 is True


def test_case_0():
    var_0 = 'test.ini'
    var_1 = True
    var_2 = 'black'
    var_3 = 'quiet'
    var_4 = 'profile'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(var_0, **var_5)
    var_7 = var_6.quiet
    assert var_7 is True
    var_8 = var_6.profile
    assert var_8 == 'black'


def test_case_0():
    var_0 = 'py310'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = {}
    var_5 = module_0.Config(config=var_3, **var_4)
    var_6 = var_5.py_version
    assert var_6 == '310'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_profile_name_not_in_profiles_and_plugin_entry_points_exist. Retrieved 4/11 statements.



def test_case_0():
    var_0 = 'some'
    var_1 = 'config'
    var_2 = 'test_profile'
    var_3 = 'profile'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_profile'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test___post_init___valid_py_version_auto. Retrieved 9/14 statements.
# Failed to parse test___post_init___multi_line_output_vertical_grid_grouped_no_comma_converted.



def test_case_0():
    var_0 = 'version_info'
    var_1 = ()
    var_2 = 'major'
    var_3 = 'minor'
    var_4 = 3
    var_5 = 8
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'auto'
    var_9 = module_0._Config(var_8)
    var_10 = var_9.py_version
    assert var_10 == 'py38'


def test_case_0():
    var_0 = 'all'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'all'


def test_case_0():
    var_0 = '310'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py310'


def test_case_0():
    var_0 = '99'
    var_1 = module_0._Config(var_0)
    var_2 = 'The python version 99 is not supported'


def test_case_0():
    var_0 = 'py38'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True


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


def test_case_0():
    var_0 = 100
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = 'wrap_length must be set lower than or equal to line_length'


def test_case_0():
    var_0 = 79
    var_1 = module_0._Config(line_length=var_0, wrap_length=var_0)
    var_2 = var_1.wrap_length
    assert var_2 == 79
    var_3 = var_1.line_length
    assert var_3 == 79


def test_case_0():
    var_0 = 50
    var_1 = 79
    var_2 = module_0._Config(line_length=var_1, wrap_length=var_0)
    var_3 = var_2.wrap_length
    assert var_3 == 50
    var_4 = var_2.line_length
    assert var_4 == 79



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_supported_filetype_os_error_occurs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.txt'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_config_finds_toml_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_finds_editorconfig_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_finds_cfg_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_finds_ini_file. Retrieved 2/9 statements.
# Partially parsed test_find_config_stops_at_stop_dir. Retrieved 4/15 statements.
# Partially parsed test_find_config_searches_upwards. Retrieved 5/14 statements.
# Failed to parse test_find_config_limits_search_depth.
# Partially parsed test_find_config_handles_invalid_config_gracefully. Retrieved 2/9 statements.
# Failed to parse test_find_config_returns_empty_dict_when_no_config.
# Partially parsed test_find_config_prefers_closer_config. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nline_length = 100'

def test_case_0():
    var_0 = '.editorconfig'
    var_1 = '[*.py]\nindent_style = space\nindent_size = 2'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[tool:black]\nline_length = 88'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.black]\nskip_string_normalization = true'

def test_case_0():
    var_0 = 'sub'
    var_1 = '.git'
    var_2 = 'pyproject.toml'
    var_3 = '[tool.black]\nline_length = 100'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'pyproject.toml'
    var_4 = '[tool.black]\nline_length = 120'

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'invalid toml content !@#$%'

def test_case_0():
    var_0 = 'sub'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.black]\nline_length = 100'
    var_3 = '[tool.black]\nline_length = 80'



# Parsed testcases at query #9
#--------------------------





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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_config_constructor_with_settings_file. Retrieved 2/3 statements.
# Failed to parse test_config_constructor_with_settings_path.
# Partially parsed test_config_constructor_without_parameters. Retrieved 2/9 statements.



def test_case_0():
    var_0 = 'quiet'
    var_1 = 'py_version'
    var_2 = True
    var_3 = 'py310'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._Config(var_3, quiet=var_2)
    var_6 = {}
    var_7 = module_0.Config(config=var_5, **var_6)
    var_8 = var_7.quiet
    assert var_8 is True
    var_9 = var_7.py_version
    assert var_9 == '310'


def test_case_0():
    var_0 = 'quiet'
    var_1 = 'py_version'
    var_2 = False
    var_3 = 'py39'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._Config(var_3, quiet=var_2)
    var_6 = True
    var_7 = 'quiet'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(config=var_5, **var_8)
    var_10 = var_9.quiet
    assert var_10 is True
    var_11 = var_9.py_version
    assert var_11 == '39'


def test_case_0():
    var_0 = 'test.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.directory


def test_case_0():
    var_0 = 'black'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.profile
    var_5 = bool(var_3.profile == var_0)
    assert var_5 is True


def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '


def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '\t'


def test_case_0():
    var_0 = 'custom_section'
    var_1 = 'mypackage'
    var_2 = [var_1]
    var_3 = f'known_{var_0}'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = frozenset(var_2)
    var_7 = var_5.known_other[var_0]
    var_8 = bool(var_5.known_other[var_0] == var_6)
    assert var_8 is True


def test_case_0():
    var_0 = 'custom_section'
    var_1 = 'Custom Section'
    var_2 = f'import_heading_{var_0}'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4.import_headings[var_0]
    var_6 = bool(var_4.import_headings[var_0] == var_1)
    assert var_6 is True


def test_case_0():
    var_0 = 'custom_section'
    var_1 = 'End of Custom Section'
    var_2 = f'import_footer_{var_0}'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = var_4.import_footers[var_0]
    var_6 = bool(var_4.import_footers[var_0] == var_1)
    assert var_6 is True


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


def test_case_0():
    var_0 = 'custom_formatter'
    var_1 = 'formatter'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.formatter
    var_5 = bool(var_3.formatter == var_0)
    assert var_5 is True


def test_case_0():
    var_0 = 'force_single_line'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'force_single_line'
    var_4 = {var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_0 not in var_5.__dict__)
    assert var_6 is True


def test_case_0():
    var_0 = 'unsupported_option'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'unsupported_option'
    var_4 = {var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.directory
    var_3 = 'src'
    var_4 = var_1.src_paths


def test_case_0():
    var_0 = True
    var_1 = 'quiet'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.quiet
    assert var_4 is True


def test_case_0():
    var_0 = 'py38'
    var_1 = 'py_version'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.py_version
    assert var_4 == '38'


def test_case_0():
    var_0 = '/tmp'
    var_1 = 'directory'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.directory
    var_5 = bool(var_3.directory == var_0)
    assert var_5 is True


def test_case_0():
    var_0 = 'skip1'
    var_1 = {var_0}
    var_2 = 'skip2'
    var_3 = {var_2}
    var_4 = 'skip'
    var_5 = 'extend_skip'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = {var_0, var_2}
    var_9 = frozenset(var_8)
    var_10 = var_7.skips
    var_11 = bool(var_7.skips == var_9)
    assert var_11 is True


def test_case_0():
    var_0 = '*.pyc'
    var_1 = {var_0}
    var_2 = '*.pyo'
    var_3 = {var_2}
    var_4 = 'skip_glob'
    var_5 = 'extend_skip_glob'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = {var_0, var_2}
    var_9 = frozenset(var_8)
    var_10 = var_7.skip_globs
    var_11 = bool(var_7.skip_globs == var_9)
    assert var_11 is True


def test_case_0():
    var_0 = 'natural'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function


def test_case_0():
    var_0 = 'native'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.sorting_function


def test_case_0():
    var_0 = 'custom'
    var_1 = 'sort_order'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_formatter_plugin_found. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'MockPlugin'
    var_1 = ()
    var_2 = 'name'
    var_3 = 'load'
    var_4 = 'test_formatter'
    var_5 = 'formatting_function'
    var_6 = lambda : var_5
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'isort.formatters'
    var_10 = []
    var_11 = 'formatter'
    var_12 = {var_11: var_4}
    var_13 = 'Plugin not found'
    var_14 = [var_13]
    var_15 = var_12['formatting_function']
    assert var_15 == 'formatting_function'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__get_config_data_toml. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_ini. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_indent_spaces. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_indent_tabs. Retrieved 5/12 statements.
# Partially parsed test__get_config_data_editorconfig_extension_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_numeric. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_false. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_boolean_true. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_comment_prefix. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_known_prefix. Retrieved 4/13 statements.
# Partially parsed test__get_config_data_bool_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_empty_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_multiple_sections. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '\n[tool.black]\nline_length = 88\ntarget_version = ["py37", "py38"]\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nindent = "    "\nline_length = 100\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\nroot = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 80\n'
    var_1 = '*'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\nroot = true\n\n[*]\nindent_style = tab\ntab_width = 4\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '\nroot = true\n\n[*.{py,pyi}]\nindent_style = space\nindent_size = 4\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nforce_grid_wrap = 3\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nforce_grid_wrap = false\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nforce_grid_wrap = true\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\ncomment_prefix = "# "\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nknown_third_party = requests,django\n'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'known_third_party'
    var_4 = 'requests'
    var_5 = 'django'

def test_case_0():
    var_0 = '\n[*.py]\nskip_gitignore = true\nmulti_line_output = 3\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nline_length = 120\n'
    var_1 = '*.c'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '\n[*.py]\nline_length = 100\n\n[*.js]\nline_length = 80\n'
    var_1 = '*.py'
    var_2 = '*.js'
    var_3 = (var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__get_config_data_toml. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_ini. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_tab. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_editorconfig_line_length_off. Retrieved 5/12 statements.
# Partially parsed test__get_config_data_editorconfig_wildcard_extension. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_backwards_compat_false. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_force_grid_wrap_backwards_compat_true. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_comment_prefix. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_known_prefix. Retrieved 6/19 statements.
# Partially parsed test__get_config_data_bool_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_tuple_conversion. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_frozenset_conversion. Retrieved 4/13 statements.
# Partially parsed test__get_config_data_empty_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_nonexistent_section. Retrieved 3/10 statements.
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
    var_0 = 'root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nmax_line_length = off\n'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*.{py,pyi}]\nindent_size = 4\n'
    var_1 = '*.{py}'
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
    var_0 = '[black]\nextend-exclude = foo, bar\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'extend-exclude'
    var_4 = 'foo'
    var_5 = 'bar'

def test_case_0():
    var_0 = '[black]\nskip_magic_trailing_comma = yes\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\npyproject_include = a, b, c\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nrequired_version = 1.0, 2.0\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'required_version'
    var_4 = '1.0'
    var_5 = '2.0'

def test_case_0():
    var_0 = '[black]\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[other]\nline_length = 100\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\n[pyproject]\nline_length = 100\n'
    var_1 = 'black'
    var_2 = 'pyproject'
    var_3 = (var_1, var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_comment_prefix_strips_quotes. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'comment_prefix'
    var_1 = "'test'"
    var_2 = {var_0: var_1}
    var_3 = 'comment_prefix'
    var_4 = var_2[var_3]
    var_5 = str(var_4)
    var_6 = "'"
    var_7 = '"'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_maps_to_section_in_known_section_mapping. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'known_foo'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'known_foo'



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['a', 'b', 'c'])
    assert var_5 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['hello'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a,b,c'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a\nb\nc'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = 'a,b\nc'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = '  a , b ,  c  '
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == ['a', 'b', 'c'])
    assert var_2 is True


def test_case_0():
    var_0 = ''
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True


def test_case_0():
    var_0 = ',\n,,'
    var_1 = module_0._as_list(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True


def test_case_0():
    var_0 = '  x  '
    var_1 = ' y '
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._as_list(var_3)
    var_5 = bool(var_4 == ['x', 'y', 'z'])
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_formatter_plugin_found. Retrieved 11/18 statements.



def test_case_0():
    var_0 = 'MockPlugin'
    var_1 = ()
    var_2 = 'name'
    var_3 = 'custom_formatter'
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'MockEntryPoints'
    var_7 = ()
    var_8 = 'group'
    var_9 = 'isort.formatters'
    var_10 = []
    var_11 = 'formatter'
    var_12 = {var_11: var_3}
    var_13 = module_0.Config(**var_12)
    var_14 = var_13.formatting_function
    var_15 = bool(var_13.formatting_function is not None)
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_warning_when_settings_file_empty_and_not_quiet. Retrieved 2/11 statements.


def test_case_0():
    var_0 = ''
    var_1 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__get_config_data_with_toml_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_ini_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_editorconfig_file. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_editorconfig_file_tab_indent. Retrieved 5/13 statements.
# Partially parsed test__get_config_data_with_editorconfig_file_extension_section. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_unknown_setting. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_force_grid_wrap_string. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_comment_prefix. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_frozenset_type. Retrieved 4/14 statements.
# Partially parsed test__get_config_data_with_tuple_type. Retrieved 4/14 statements.
# Partially parsed test__get_config_data_with_known_prefix. Retrieved 6/22 statements.
# Partially parsed test__get_config_data_with_empty_section. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_nonexistent_section. Retrieved 3/11 statements.
# Partially parsed test__get_config_data_with_multiple_sections. Retrieved 4/12 statements.
# Partially parsed test__get_config_data_with_nested_toml_section. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '[tool.black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 100\nskip_string_normalization = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'indent_style = space\nindent_size = 2\nmax_line_length = 80\n'
    var_1 = '*'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'indent_style = tab\ntab_width = 4\nmax_line_length = off\n'
    var_1 = '*'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*]\nindent_style = space\nindent_size = 4\n\n[*.{py,pyi}]\nmax_line_length = 120\n'
    var_1 = '*.{py}'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 100\nunknown_setting = value\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'unknown_setting'

def test_case_0():
    var_0 = '[black]\nforce_grid_wrap = false\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\ncomment_prefix = "# "\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nknown_third_party = requests, pytest\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'known_third_party'
    var_4 = 'requests'
    var_5 = 'pytest'

def test_case_0():
    var_0 = '[black]\ninclude = *.py, *.pyi\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'include'
    var_4 = '*.py'
    var_5 = '*.pyi'

def test_case_0():
    var_0 = '[black]\nextra_standard_library = lib1, lib2\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'extra_standard_library'
    var_4 = 'lib1'
    var_5 = 'lib2'

def test_case_0():
    var_0 = '[black]\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[other]\nline_length = 100\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[tool.black]\nline_length = 88\n[tool.isort]\nprofile = black\n'
    var_1 = 'tool.black'
    var_2 = 'tool.isort'
    var_3 = (var_1, var_2)

def test_case_0():
    var_0 = '[tool.black.format]\nline_length = 88\n'
    var_1 = 'tool.black.format'
    var_2 = (var_1,)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__get_config_data_with_toml. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_ini. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_editorconfig. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_editorconfig_tab_indent. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_editorconfig_off_line_length. Retrieved 5/12 statements.
# Partially parsed test__get_config_data_with_editorconfig_extension_wildcard. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_force_grid_wrap_string. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_comment_prefix. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_known_prefix. Retrieved 4/13 statements.
# Partially parsed test__get_config_data_with_tuple_type. Retrieved 4/13 statements.
# Partially parsed test__get_config_data_with_empty_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_nonexistent_section. Retrieved 3/10 statements.
# Partially parsed test__get_config_data_with_multiple_sections. Retrieved 4/11 statements.
# Partially parsed test__get_config_data_with_toml_nested_section. Retrieved 3/10 statements.


def test_case_0():
    var_0 = b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'tool.black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\nskip_string_normalization = true\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 79\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nindent_style = tab\nindent_size = 2\n'
    var_1 = '*.py'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[*.py]\nmax_line_length = off\n'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'inf'
    var_4 = float(var_3)

def test_case_0():
    var_0 = '[*.{py,pyi}]\nindent_size = 2\n'
    var_1 = '*.{py}'
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
    var_0 = '[black]\nknown_third_party = requests,django\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'known_third_party'
    var_4 = 'requests'
    var_5 = 'django'

def test_case_0():
    var_0 = '[black]\ninclude = "*.py,*.pyi"\n'
    var_1 = 'black'
    var_2 = (var_1,)
    var_3 = 'include'
    var_4 = '*.py'
    var_5 = '*.pyi'

def test_case_0():
    var_0 = '[black]\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[other]\nline_length = 100\n'
    var_1 = 'black'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '[black]\nline_length = 88\n[pycodestyle]\nmax_line_length = 79\n'
    var_1 = 'black'
    var_2 = 'pycodestyle'
    var_3 = (var_1, var_2)

def test_case_0():
    var_0 = b'[tool.isort]\nprofile = "black"\n'
    var_1 = 'tool.isort'
    var_2 = (var_1,)



