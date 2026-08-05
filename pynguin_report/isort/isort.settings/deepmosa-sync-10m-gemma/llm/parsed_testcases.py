####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_config_post_init_multi_line_output_transformation.


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
    var_0 = '310'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.py_version
    assert var_2 == 'py310'

import isort.settings as module_0

def test_case_0():
    var_0 = '99'
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

import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = var_1.known_standard_library
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '3.9'
    var_1 = 4
    var_2 = 'black'
    var_3 = 'py_version'
    var_4 = 'indent'
    var_5 = 'profile'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = var_7.py_version
    assert var_8 == '3.9'
    var_9 = var_7.indent
    assert var_9 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = '3.8'
    var_1 = 2
    var_2 = 'py_version'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '3.10'
    var_7 = 'tab'
    var_8 = 'py_version'
    var_9 = 'indent'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.Config(config=var_5, **var_10)
    var_12 = var_11.py_version
    assert var_12 == '3.10'
    var_13 = var_11.indent
    assert var_13 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = var_3.indent
    assert var_4 == '    '
    var_5 = '2'
    var_6 = 'indent'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = var_8.indent
    assert var_9 == '  '
    var_10 = 'tab'
    var_11 = 'indent'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = var_13.indent
    assert var_14 == '\t'

import isort.settings as module_0

def test_case_0():
    var_0 = 'django_module'
    var_1 = 'django'
    var_2 = 'known_django'
    var_3 = 'sections'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'django'
    var_7 = bool('django' in var_5.sections)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'module_a'
    var_1 = 'end_a'
    var_2 = 'import_heading_custom'
    var_3 = 'import_footer_custom'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.import_headings['custom']
    assert var_6 == 'module_a'
    var_7 = var_5.import_footers['custom']
    assert var_7 == 'end_a'

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = (var_0, var_1)
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = bool(var_8 in var_5.src_paths)
    assert var_9 is True
    var_10 = [var_1]
    var_11 = {}
    var_12 = module_1.Path(*var_10, **var_11)
    var_13 = bool(var_12 in var_5.src_paths)
    assert var_13 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path/to/config'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'some_value'
    var_1 = 'non_existent_key'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_skipped_skips_exact_path. Retrieved 3/25 statements.
# Partially parsed test_is_skipped_skips_parent_folder. Retrieved 4/18 statements.
# Partially parsed test_is_skipped_matches_glob. Retrieved 4/18 statements.
# Partially parsed test_is_skipped_does_not_skip_regular_file. Retrieved 4/13 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)

import pathlib as module_0

def test_case_0():
    var_0 = 'ignored_dir'
    var_1 = [var_0]
    var_2 = '/home/user/ignored_dir/file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)

import pathlib as module_0

def test_case_0():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = '/home/user/data.tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)

import pathlib as module_0

def test_case_0():
    var_0 = '/other/path'
    var_1 = [var_0]
    var_2 = '/home/user/actual_file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_supported_filetype_returns_true_for_py_extension. Retrieved 3/14 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_blocked_extension. Retrieved 3/10 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_editor_backup_files. Retrieved 2/8 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_fifo_files. Retrieved 2/13 statements.
# Partially parsed test_is_supported_filetype_returns_false_when_file_cannot_be_read. Retrieved 2/13 statements.
# Partially parsed test_is_supported_filetype_returns_true_for_shebang_file. Retrieved 2/13 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = '.txt'
    var_2 = 'test.py'

def test_case_0():
    var_0 = '.py'
    var_1 = '.txt'
    var_2 = 'test.txt'

def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py~'

def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py'

def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py'

def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_src_paths_with_glob. Retrieved 4/17 statements.
# Partially parsed test_config_src_paths_no_glob. Retrieved 4/17 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'match.py'
    var_1 = '*.py'
    var_2 = (var_1,)
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(True)
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'subfolder'
    var_1 = 'subfolder'
    var_2 = (var_1,)
    var_3 = 'src_paths'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_config_init_with_existing_config. Retrieved 1/7 statements.


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
    var_0 = 2

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
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
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'some_old_option'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'unsupported_key'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'unsupported_key'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_init_with_config_object. Retrieved 15/21 statements.


def test_case_0():
    var_0 = 'py_version'
    var_1 = '_known_patterns'
    var_2 = '_section_comments'
    var_3 = '_section_comments_end'
    var_4 = '_skips'
    var_5 = '_skip_globs'
    var_6 = '_sorting_function'
    var_7 = 'py39'
    var_8 = []
    var_9 = ()
    var_10 = ()
    var_11 = frozenset()
    var_12 = frozenset()
    var_13 = None
    var_14 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13}



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/path/to/file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_5.is_skipped(var_8)
    assert var_9 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/path/to/ignored_dir'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '/path/to/ignored_dir/some_file.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = var_5.is_skipped(var_9)
    assert var_10 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_globs'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_file.tmp'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = var_5.is_skipped(var_9)
    assert var_10 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/non/existent/path/at/all'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = var_1.is_skipped(var_5)
    assert var_6 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/dev/null'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = var_1.is_skipped(var_5)
    assert var_6 is False

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/project/.git/config'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = var_3.is_skipped(var_7)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_supported_filetype_py_extension. Retrieved 3/10 statements.
# Partially parsed test_is_supported_filetype_blocked_extension. Retrieved 4/5 statements.
# Partially parsed test_is_supported_filetype_fifo_pipe. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_unreadable_file. Retrieved 3/7 statements.
# Partially parsed test_is_supported_filetype_no_shebang. Retrieved 3/7 statements.


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
    var_2 = '.txt'
    var_3 = 'test.txt'
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
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_config_formatter_plugin_exists. Retrieved 3/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'black'
    var_2 = 'formatter'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(settings_path=var_0, **var_3)
    var_5 = var_4._Config__dataclass_fields__['py_version']
    var_6 = bool(var_4._Config__dataclass_fields__['py_version'] is not None)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_config_init_predicate_false_via_settings_file_with_data. Retrieved 6/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'some_key'
    var_2 = 'black'
    var_3 = 'some_value'
    var_4 = 'test.ini'
    var_5 = {}
    var_6 = module_0.Config(var_4, **var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_abspaths_joins_cwd_when_value_is_relative_and_ends_with_sep. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'file.txt'
    var_2 = '/absolute/path'
    var_3 = 'dir/'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_1, var_2, var_3}
    var_6 = module_0._abspaths(var_0, var_4)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import posixpath as module_0

def test_case_0():
    var_0 = '/home/user'
    var_1 = 'subdir/'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = module_0.join(var_0, *var_3)
    var_5 = {var_4}

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
    var_1 = '/root/dir/'
    var_2 = [var_1]
    var_3 = {var_1}
    var_4 = module_0._abspaths(var_0, var_2)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

def test_case_0():
    var_0 = '/base'
    var_1 = 'rel/'
    var_2 = 'abs'
    var_3 = '/abs/'
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_config_init_with_overrides. Retrieved 5/17 statements.
# Partially parsed test_config_init_with_settings_file. Retrieved 18/37 statements.


def test_case_0():
    var_0 = 'indent'
    var_1 = 'known_first_party'
    var_2 = 4
    var_3 = 'my_module'
    var_4 = {var_0: var_2, var_1: var_3}

import isort.settings as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'indent'
    var_2 = 'black'
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'isort.config._get_config_data'
    var_6 = 'os.path.basename'
    var_7 = 'pyproject.toml'
    var_8 = 'os.path.exists'
    var_9 = True
    var_10 = 'os.path.abspath'
    var_11 = '/fake/path/pyproject.toml'
    var_12 = 'isort.config._find_config'
    var_13 = '/fake/path'
    var_14 = (var_13, var_4)
    var_15 = '/fake/path/pyproject.toml'
    var_16 = 4
    var_17 = 'indent'
    var_18 = {var_17: var_16}
    var_19 = module_0.Config(var_15, **var_18)
    var_20 = bool(var_19 is not None)
    assert var_20 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = '4'
    var_2 = 'indent'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = 'tab'
    var_2 = 'indent'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'some_pkg'
    var_1 = [var_0]
    var_2 = 'known_my_section'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_non_existent_path. Retrieved 3/21 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/this_path_should_not_exist_12345'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_explicit_skip_path. Retrieved 13/17 statements.
# Partially parsed test_is_skipped_returns_true_for_glob_match. Retrieved 12/16 statements.
# Partially parsed test_is_skipped_returns_true_for_directory_in_skips. Retrieved 12/16 statements.
# Partially parsed test_is_skipped_returns_false_for_normal_file. Retrieved 16/20 statements.
# Partially parsed test_is_skipped_returns_true_for_non_existent_path. Retrieved 10/14 statements.


import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/tmp/skip_me.py'
    var_1 = [var_0]
    var_2 = frozenset()
    var_3 = frozenset(var_1, var_2)
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = frozenset()
    var_7 = frozenset()
    var_8 = frozenset()
    var_9 = 'skips'
    var_10 = 'skip_globs'
    var_11 = 'extend_skip'
    var_12 = 'extend_skip_glob'
    var_13 = 'skip'
    var_14 = 'skip_glob'
    var_15 = {var_9: var_3, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7, var_14: var_8}
    var_16 = module_0.Config(**var_15)
    var_17 = [var_0]
    var_18 = [var_0]
    var_19 = {}
    var_20 = module_1.Path(*var_18, **var_19)
    var_21 = var_16.is_skipped(var_20)
    assert var_21 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = frozenset()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = 'skips'
    var_7 = 'skip_globs'
    var_8 = 'extend_skip'
    var_9 = 'extend_skip_glob'
    var_10 = 'skip'
    var_11 = 'skip_glob'
    var_12 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_0.Config(**var_12)
    var_14 = '*.tmp'
    var_15 = [var_14]
    var_16 = '/tmp/test_file.tmp'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Path(*var_17, **var_18)
    var_20 = var_13.is_skipped(var_19)
    assert var_20 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = frozenset()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = 'skips'
    var_7 = 'skip_globs'
    var_8 = 'extend_skip'
    var_9 = 'extend_skip_glob'
    var_10 = 'skip'
    var_11 = 'skip_glob'
    var_12 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_0.Config(**var_12)
    var_14 = '/tmp/ignored_dir'
    var_15 = [var_14]
    var_16 = '/tmp/ignored_dir/file.py'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Path(*var_17, **var_18)
    var_20 = var_13.is_skipped(var_19)
    assert var_20 is True

import pathlib as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'test_not_skipped.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = frozenset()
    var_6 = frozenset()
    var_7 = frozenset()
    var_8 = frozenset()
    var_9 = frozenset()
    var_10 = frozenset()
    var_11 = 'skips'
    var_12 = 'skip_globs'
    var_13 = 'extend_skip'
    var_14 = 'extend_skip_glob'
    var_15 = 'skip'
    var_16 = 'skip_glob'
    var_17 = {var_11: var_5, var_12: var_6, var_13: var_7, var_14: var_8, var_15: var_9, var_16: var_10}
    var_18 = module_1.Config(**var_17)
    var_19 = '/not_in_skips'
    var_20 = [var_19]
    var_21 = '*.not_matching'
    var_22 = [var_21]
    var_23 = var_18.is_skipped(var_3)
    assert var_23 is False
    var_24 = var_3.unlink()

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = frozenset()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = 'skoi'
    var_7 = 'skip_globs'
    var_8 = 'extend_skip'
    var_9 = 'extend_skip_glob'
    var_10 = 'skip'
    var_11 = 'skip_glob'
    var_12 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_0.Config(**var_12)
    var_14 = '/non/existent/path/to/nowhere'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Path(*var_15, **var_16)
    var_18 = var_13.is_skipped(var_17)
    assert var_18 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_skipped_while_loop_condition_false. Retrieved 2/38 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_constructor_with_overrides. Retrieved 6/11 statements.
# Partially parsed test_config_constructor_with_existing_config. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 'include'
    var_2 = 88
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = 'source'

def test_case_0():
    var_0 = 100

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.line_length
    assert var_3 == 79

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.indent
    assert var_3 == '    '

import isort.settings as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = var_2.indent
    assert var_3 == '\t'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_supported_filetype_returns_true_for_py_file. Retrieved 2/6 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_blocked_extension. Retrieved 3/8 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_editor_backup_file. Retrieved 2/6 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_fifo_file. Retrieved 1/6 statements.
# Partially parsed test_is_supported_filetype_returns_false_for_non_existent_file. Retrieved 1/2 statements.
# Partially parsed test_is_supported_filetype_returns_true_for_supported_extension_without_shebang. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '#!/usr/bin/python\nimport os'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'hello'
    var_2 = '.txt'

def test_case_0():
    var_0 = 'test.py~'
    var_1 = '#!/usr/bin/python\nimport os'

def test_case_0():
    var_0 = 'test_fifo'

def test_case_0():
    var_0 = 'non_existent_file.py'

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = '.py'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_git_ignored_file_when_skip_gitignore_is_enabled. Retrieved 9/10 statements.
# Partially parsed test_is_skipped_returns_false_for_git_tracked_file_when_skip_gitignore_is_enabled. Retrieved 8/9 statements.
# Partially parsed test_is_skipped_returns_true_for_git_dot_folder. Retrieved 9/10 statements.


import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = var_4.is_skipped(var_7)
    assert var_8 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'ignored_folder'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'ignored_folder/some_file.py'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_4.is_skipped(var_8)
    assert var_9 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'data.tmp'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_4.is_skipped(var_8)
    assert var_9 is True
    var_10 = 'subdir/temp_file.tmp'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Path(*var_11, **var_12)
    var_14 = var_4.is_skipped(var_13)
    assert var_14 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'other.py'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'keep_me.py'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_4.is_skipped(var_8)
    assert var_9 is False

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'non_existent_file_12345.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = var_1.is_skipped(var_5)
    assert var_6 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/tmp'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = '/tmp/tracked_file.py'
    var_9 = {var_8}
    var_10 = '/tmp/ignored.py'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Path(*var_11, **var_12)
    var_14 = var_3.is_skipped(var_13)
    assert var_14 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '/tmp'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = '/tmp/tracked_file.py'
    var_9 = {var_8}
    var_10 = [var_8]
    var_11 = {}
    var_12 = module_1.Path(*var_10, **var_11)
    var_13 = var_3.is_skipped(var_12)
    assert var_13 is False

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '.'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = './file.py'
    var_9 = {var_8}
    var_10 = '.git/config'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Path(*var_11, **var_12)
    var_14 = var_3.is_skipped(var_13)
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_config_init_with_overrides. Retrieved 5/9 statements.
# Partially parsed test_config_init_with_existing_config_object. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'line_length'
    var_1 = 'indent'
    var_2 = 88
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}

import isort.settings as module_0

def test_case_0():
    var_0 = '/tmp/pyproject.toml'
    var_1 = False
    var_2 = 'quiet'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(var_0, **var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'non_existent_profile'
    var_1 = 'profile'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'ProfileDoesNotExist'

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

def test_case_0():
    var_0 = 100



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_skipped_while_loop_condition_false. Retrieved 2/42 statements.


def test_case_0():
    var_0 = 'empty_logic.py'
    var_1 = 'content'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/fake/project/pyproject.toml'
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'known_custom'
    var_1 = 'some_module'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'module_a'
    var_5 = [var_4]
    var_6 = 'known_custom'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_skipped_returns_true_when_path_does_not_exist. Retrieved 3/10 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '\n    Tests that is_skipped returns True when the file_path does not exist \n    on the filesystem (triggering the predicate at line 30).\n    '
    var_1 = '/tmp/this_file_should_not_exist_12345'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_config_handles_get_config_data_exception. Retrieved 2/4 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user/project'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/home/user/project', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user/project'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/home/user/project', {'key': 'value'}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user/project'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/home/user/project', {}))
    assert var_2 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/home/user/project'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/home/user/project', {}))
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '/root'
    var_1 = module_0._find_config(var_0)
    var_2 = bool(var_1 == ('/root', {}))
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_skipped_loop_terminates_when_no_parent_matches_skips. Retrieved 2/37 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_find_all_configs_returns_trie_instance.
# Partially parsed test_find_all_configs_with_nested_structure. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'subdir'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_supported_filetype_oserror_on_stat. Retrieved 3/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_skipped_directory_not_in_parents. Retrieved 4/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/home/user/my_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.resolve()
    var_5 = var_4.parents



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'known_custom'
    var_1 = 'sections'
    var_2 = 'quiet'
    var_3 = 'some_pkg'
    var_4 = [var_3]
    var_5 = 'standard'
    var_6 = (var_5,)
    var_7 = False
    var_8 = {var_0: var_4, var_1: var_6, var_2: var_7}
    var_9 = 'known_unknown'
    var_10 = 'sections'
    var_11 = 'quiet'
    var_12 = 'val'
    var_13 = [var_12]
    var_14 = 'standard'
    var_15 = (var_14,)
    var_16 = False
    var_17 = {var_9: var_13, var_10: var_15, var_11: var_16}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_config_returns_empty_dict_when_no_config_exists. Retrieved 1/5 statements.
# Partially parsed test_find_config_returns_config_when_found. Retrieved 1/5 statements.
# Partially parsed test_find_config_stops_searching_on_stop_dirs. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_path_at_all'

def test_case_0():
    var_0 = '/mock/project'

def test_case_0():
    var_0 = '/mock/project/stop_dir'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_config_path_root_is_directory. Retrieved 5/21 statements.


import posixpath as module_0

def test_case_0():
    var_0 = 'dummy content'
    var_1 = 'directory'
    var_2 = 'test_file.txt'
    var_3 = module_0.abspath(var_2)
    var_4 = {var_1: var_3}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_config_predicate_true. Retrieved 15/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'os.path.join'
    var_1 = '/fake/path/config.yaml'
    var_2 = 'os.path.isfile'
    var_3 = True
    var_4 = 'builtins.CONFIG_SOURCES'
    var_5 = 'config.yaml'
    var_6 = [var_5]
    var_7 = 'builtins.MAX_CONFIG_SEARCH_DEPTH'
    var_8 = 5
    var_9 = 'builtins._get_config_data'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = '/fake/path'
    var_14 = module_0._find_config(var_13)
    var_15 = bool(var_14 == ('/fake/path', {'key': 'value'}))
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_skipped_returns_true_for_git_ignored_files_when_enabled. Retrieved 9/10 statements.
# Partially parsed test_is_skipped_returns_false_for_valid_tracked_git_file. Retrieved 8/10 statements.


import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/tmp/ignored_file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_5.is_skipped(var_8)
    assert var_9 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/tmp/ignored_dir'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '/tmp/skip_folder/sub/file.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = '/tmp/ignored_dir/file.py'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Path(*var_11, **var_12)
    var_14 = var_5.is_skipped(var_13)
    assert var_14 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skip_globs'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test_file.tmp'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = var_5.is_skipped(var_9)
    assert var_10 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = '/tmp/ignored'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '/home/user/project/main.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/non/existent/path/to/file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = var_1.is_skipped(var_5)
    assert var_6 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'C:/ignored/file.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'skips'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'C:\\ignored\\file.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = var_5.is_skipped(var_9)
    assert var_10 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = frozenset()
    var_2 = 'skip_gitignore'
    var_3 = 'skips'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '/repo'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = '/repo/ignored_file.py'
    var_11 = {var_10}
    var_12 = [var_10]
    var_13 = {}
    var_14 = module_1.Path(*var_12, **var_13)
    var_15 = var_5.is_skipped(var_14)
    assert var_15 is True

import isort.settings as module_0
import pathlib as module_1

def test_case_0():
    var_0 = True
    var_1 = frozenset()
    var_2 = 'skip_gitignore'
    var_3 = 'skips'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '/repo'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = '/repo/tracked_file.py'
    var_11 = {var_10}
    var_12 = [var_10]
    var_13 = {}
    var_14 = module_1.Path(*var_12, **var_13)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_config_constructor_with_existing_config. Retrieved 1/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'py310'
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
    var_0 = '4'

import isort.settings as module_0

def test_case_0():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = {}
    var_2 = module_0.Config(settings_path=var_0, **var_1)



