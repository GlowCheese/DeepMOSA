####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/8 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 2/10 statements.
# Partially parsed test_identify_imports_main_with_unique_flag. Retrieved 3/11 statements.
# Partially parsed test_identify_imports_main_with_packages_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_modules_flag. Retrieved 3/11 statements.
# Partially parsed test_identify_imports_main_with_attributes_flag. Retrieved 3/11 statements.
# Partially parsed test_identify_imports_main_with_follow_links_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_top_only_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_multiple_files. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--top-only'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\n'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\nimport os\nimport sys\n'
    var_2 = '--unique'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os.path\nimport sys.argv\n'
    var_2 = '--packages'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\nfrom pathlib import Path\n'
    var_2 = '--modules'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'from pathlib import Path\nfrom os import path\n'
    var_2 = '--attributes'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\n'
    var_2 = '--follow-links'
    var_3 = 'os'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\n\ndef func():\n    import sys\n'
    var_2 = '--top-only'
    var_3 = 'os'

def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'import os\n'
    var_2 = 'test2.py'
    var_3 = 'import sys\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is True
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is True
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_5)
    assert var_8 is True
    var_9 = var_7.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_2, var_1, var_3, write_to_stdout=var_4, **var_5)
    var_7 = bool(var_2)
    assert var_7 is True
    var_8 = var_6.incorrectly_sorted
    assert var_8 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_2, var_1, var_3, var_4, **var_5)
    var_7 = bool(var_2)
    assert var_7 is True
    var_8 = var_6.incorrectly_sorted
    assert var_8 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_during_check. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = True
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 4/10 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_multi_line_output_numeric. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_string. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_src_paths. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'prog'
    var_1 = '--check'
    var_2 = None
    var_3 = module_0.parse_args(var_2)
    var_4 = 'check'
    var_5 = bool('check' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'check'
    var_4 = bool('check' in var_2)
    assert var_4 is True
    var_5 = var_2['check']
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = '--diff'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'check'
    var_5 = bool('check' in var_3)
    assert var_5 is True
    var_6 = 'diff'
    var_7 = bool('diff' in var_3)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = bool('order_by_type' in var_2)
    assert var_4 is True
    var_5 = var_2['order_by_type']
    assert var_5 is False
    var_6 = 'dont_order_by_type'
    var_7 = bool('dont_order_by_type' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = bool('follow_links' in var_2)
    assert var_4 is True
    var_5 = var_2['follow_links']
    assert var_5 is False
    var_6 = 'dont_follow_links'
    var_7 = bool('dont_follow_links' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = bool('float_to_top' in var_2)
    assert var_4 is True
    var_5 = var_2['float_to_top']
    assert var_5 is False
    var_6 = 'dont_float_to_top'
    var_7 = bool('dont_float_to_top' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--settings-path'
    var_1 = '/path/to/config'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'settings_path'
    var_5 = bool('settings_path' in var_3)
    assert var_5 is True
    var_6 = var_3['settings_path']
    assert var_6 == '/path/to/config'

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = 'force_single_line'
    var_6 = bool('force_single_line' in var_2['remapped_deprecated_args'])
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sort_imports_exception_handler_line_40. Retrieved 4/12 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = 1
    var_6 = 'offending_file'
    var_7 = 'message'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_print_hard_fail_with_custom_message. Retrieved 4/7 statements.
# Partially parsed test_print_hard_fail_with_default_message. Retrieved 4/7 statements.
# Partially parsed test_print_hard_fail_with_custom_format_error. Retrieved 5/8 statements.
# Partially parsed test_print_hard_fail_without_offending_file. Retrieved 4/7 statements.
# Partially parsed test_print_hard_fail_uses_stderr. Retrieved 4/9 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'Custom error message'
    var_5 = module_1._print_hard_fail(var_3, message=var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_file.py'
    var_5 = module_1._print_hard_fail(var_3, var_4)
    var_6 = 'Unrecoverable exception'
    var_7 = 'https://github.com/PyCQA/isort/issues/new'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '[CUSTOM] {error}: {message}'
    var_1 = False
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = {var_2: var_1, var_3: var_0}
    var_5 = module_0.Config(**var_4)
    var_6 = 'Test message'
    var_7 = module_1._print_hard_fail(var_5, message=var_6)
    var_8 = '[CUSTOM]'
    var_9 = 'ERROR'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = None
    var_5 = module_1._print_hard_fail(var_3, var_4)
    var_6 = 'Unrecoverable exception thrown when parsing'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'Error output test'
    var_5 = module_1._print_hard_fail(var_3, message=var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_exception. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test.py'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 6/9 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_arguments_without_paths. Retrieved 4/7 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/16 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 7/12 statements.
# Partially parsed test_main_check_mode. Retrieved 5/10 statements.
# Partially parsed test_main_show_files. Retrieved 4/17 statements.
# Partially parsed test_main_stdin_show_files_conflict. Retrieved 5/8 statements.
# Partially parsed test_main_stream_filename_without_stdin. Retrieved 6/9 statements.
# Partially parsed test_main_recursive_root_without_allow_root. Retrieved 4/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 == 'Error: either specify show-config or show-files not both.'
    var_1 = '--show-config'
    var_2 = '--show-files'
    var_3 = 'test.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 == 'Error: arguments passed in without any paths or content.'
    var_1 = '--check'
    var_2 = [var_1]
    var_3 = module_0.main(var_2)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-files'
    var_3 = 'test.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = '--check'
    var_3 = [var_2, var_1]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '--show-files'
    var_3 = 0

import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 == "Error: can't show files for streaming input."
    var_1 = '--show-files'
    var_2 = '-'
    var_3 = [var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 == 'Filename override is intended only for stream (-) sorting.'
    var_1 = '--filename'
    var_2 = 'test.py'
    var_3 = 'somefile.py'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 == 1
    var_1 = '/'
    var_2 = [var_1]
    var_3 = module_0.main(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_resolve_all_configs_predicate. Retrieved 6/13 statements.


def test_case_0():
    var_0 = True
    assert var_0 is True
    var_1 = 'config_root'
    var_2 = '.'
    var_3 = {var_1: var_2}
    var_4 = 'config_root'
    var_5 = '.'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_args_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 7/9 statements.
# Partially parsed test_parse_args_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multiple_arguments. Retrieved 6/8 statements.
# Partially parsed test_parse_args_filters_empty_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = 'verbose'
    var_6 = bool('verbose' in var_2['remapped_deprecated_args'])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = 'dont_float_to_top'
    var_5 = bool('dont_float_to_top' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = 0
    var_9 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = '--dont-follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'
    var_5 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_parse_args_remapped_deprecated_args. Retrieved 4/13 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'old_arg'
    var_1 = 'deprecated'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = var_3['remapped_deprecated_args']
    var_7 = bool(var_3['remapped_deprecated_args'] == ['old_arg', 'deprecated'])
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_multi_line_output_predicate_evaluates_to_true. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'multi_line_output'
    var_3 = 'order_by_type'
    var_4 = 'follow_links'
    var_5 = 'float_to_top'
    var_6 = '1'
    var_7 = True
    var_8 = False
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_8}
    var_10 = '1'
    var_11 = bool(var_10)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_args_float_to_top_predicate_true. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = False



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_paths_in_config_dict_gets_resolved. Retrieved 14/24 statements.
# Partially parsed test_src_paths_not_in_config_dict. Retrieved 5/8 statements.
# Partially parsed test_src_paths_empty_list. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'settings_path'
    var_2 = './src'
    var_3 = '../other/path'
    var_4 = [var_2, var_3]
    var_5 = '/some/path'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'src_paths'
    var_8 = ()
    var_9 = 'src_paths'
    var_10 = bool('src_paths' in var_6)
    assert var_10 is True
    var_11 = var_6[var_7]
    var_12 = var_6[var_7]
    var_13 = var_6[var_7]
    var_14 = var_6[var_7]
    var_15 = len(var_14)
    assert var_15 == 2

def test_case_0():
    var_0 = 'settings_path'
    var_1 = '/some/path'
    var_2 = {var_0: var_1}
    var_3 = 'src_paths'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = bool('src_paths' not in var_2)
    assert var_6 is True

def test_case_0():
    var_0 = 'src_paths'
    var_1 = 'settings_path'
    var_2 = []
    var_3 = '/some/path'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'src_paths'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = bool('src_paths' in var_4)
    assert var_8 is True
    var_9 = var_4[var_5]
    var_10 = var_4[var_5]
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_115. Retrieved 3/9 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'settings_path'
    var_1 = '/some/path'
    var_2 = {var_0: var_1}
    var_3 = var_0 not in var_2
    assert var_3 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_main_show_version. Retrieved 6/9 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/4 statements.
# Partially parsed test_main_arguments_without_paths. Retrieved 5/13 statements.
# Partially parsed test_main_settings_path_file. Retrieved 8/24 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 7/15 statements.
# Partially parsed test_main_stdin_check_mode. Retrieved 3/7 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 5/13 statements.
# Partially parsed test_main_stream_filename_without_stdin. Retrieved 7/15 statements.
# Partially parsed test_main_show_files_with_stdin. Retrieved 6/14 statements.
# Partially parsed test_main_with_file_names. Retrieved 3/10 statements.
# Partially parsed test_main_deprecated_single_dash_args. Retrieved 6/13 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-config'
    var_3 = '--show-files'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--check'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black'
    var_4 = False
    var_5 = 'sys.exit'
    var_6 = '--settings-path'
    var_7 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--virtual-env'
    var_3 = '/nonexistent/path'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--stream-filename'
    var_3 = 'test.py'
    var_4 = 'other.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-files'
    var_3 = '-'
    var_4 = [var_2, var_3]
    var_5 = module_0.main(var_4)
    var_6 = bool(var_0)
    assert var_6 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = '--show-files'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = 'dont_order_by_type'
    var_3 = 'test.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.main(var_4)



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 'some_arg'
    var_1 = 'other_arg'
    var_2 = 'value'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/12 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/9 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 5/9 statements.
# Partially parsed test_main_with_files_argument. Retrieved 4/12 statements.
# Partially parsed test_main_with_settings_path_file. Retrieved 7/18 statements.
# Partially parsed test_main_with_settings_path_directory. Retrieved 5/14 statements.
# Partially parsed test_main_with_virtual_env_invalid. Retrieved 6/14 statements.
# Partially parsed test_main_recursive_on_root_without_allow_root. Retrieved 6/8 statements.
# Partially parsed test_main_show_files. Retrieved 6/21 statements.
# Partially parsed test_main_stdin_mode_check. Retrieved 7/11 statements.
# Partially parsed test_main_filename_override_without_stdin. Retrieved 8/15 statements.
# Partially parsed test_main_check_mode. Retrieved 5/13 statements.
# Partially parsed test_main_with_jobs. Retrieved 6/14 statements.
# Partially parsed test_main_verbose_mode. Retrieved 5/13 statements.
# Partially parsed test_main_quiet_mode. Retrieved 5/14 statements.
# Partially parsed test_main_show_config. Retrieved 6/18 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = [var_1]
    var_3 = '--show-config'
    var_4 = '--show-files'
    var_5 = [var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'either specify show-config or show-files not both'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = 'sys.argv'
    var_5 = 'isort'
    var_6 = '--settings-path'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--settings-path'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--virtual-env'
    var_5 = '/nonexistent/path'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = [var_1]
    var_3 = '/'
    var_4 = [var_3]
    var_5 = module_0.main(var_4)
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--show-files'
    var_5 = 0

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = [var_3]
    var_5 = '-'
    var_6 = '--check'
    var_7 = [var_5, var_6]

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--filename'
    var_5 = 'override.py'
    var_6 = [var_0, var_4, var_5]
    var_7 = module_0.main(var_6)
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--check'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--jobs'
    var_5 = '2'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--verbose'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--quiet'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.argv'
    var_3 = 'isort'
    var_4 = '--show-config'
    var_5 = 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false_supported_encoding. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = True
    var_2 = bool(var_0)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_line_9_true. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'show_version'
    var_1 = 'show_config'
    var_2 = 'show_files'
    var_3 = False
    var_4 = True
    var_5 = {var_0: var_3, var_1: var_4, var_2: var_4}
    var_6 = []
    var_7 = None
    var_8 = 'Error: either specify show-config or show-files not both.'



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = 'some_file.py'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]
    var_4 = var_1 == var_3
    assert var_4 is False
    var_5 = []
    var_6 = [var_2]
    var_7 = var_5 == var_6
    assert var_7 is False
    var_8 = 'file1.py'
    var_9 = 'file2.py'
    var_10 = [var_8, var_9]
    var_11 = [var_2]
    var_12 = var_10 == var_11
    assert var_12 is False
    var_13 = 'other.py'
    var_14 = [var_2, var_13]
    var_15 = [var_2]
    var_16 = var_14 == var_15
    assert var_16 is False



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 12/20 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = {var_0}
    var_2 = 'obj'
    var_3 = '__dict__'
    var_4 = f'-{var_0}'
    var_5 = True
    var_6 = False
    var_7 = {var_0: var_5, var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = 'other_arg'
    var_10 = [var_0, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = bool(f'-{var_0}' in var_10 or 'remapped_deprecated_args' in var_11)
    assert var_12 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/7 statements.
# Partially parsed test_identify_imports_main_with_file. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_unique_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_packages_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_modules_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_attributes_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_top_only_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_with_follow_links_flag. Retrieved 3/9 statements.
# Partially parsed test_identify_imports_main_multiple_files. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'from os import path\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\nfrom sys import path\n'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\nimport os\nimport sys\n'
    var_2 = '--unique'
    var_3 = 'os'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'from os.path import join\nimport sys\n'
    var_2 = '--packages'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'from os.path import join\nimport sys\n'
    var_2 = '--modules'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'from os import path\n'
    var_2 = '--attributes'
    var_3 = 'os.path'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\n\ndef foo():\n    import sys\n'
    var_2 = '--top-only'
    var_3 = 'os'

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = 'import os\n'
    var_2 = '--follow-links'
    var_3 = 'os'

def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'import os\n'
    var_2 = 'test2.py'
    var_3 = 'import sys\n'



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 8/17 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 11/27 statements.
# Partially parsed test_main_settings_path_is_directory. Retrieved 11/27 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 7/14 statements.
# Partially parsed test_main_stream_input_check_mode. Retrieved 7/13 statements.
# Partially parsed test_main_stream_input_sort_mode. Retrieved 6/11 statements.
# Partially parsed test_main_recursive_on_root_without_allow_root. Retrieved 5/12 statements.
# Partially parsed test_main_stream_filename_without_stream. Retrieved 7/14 statements.
# Partially parsed test_main_show_files_with_stream. Retrieved 6/13 statements.
# Partially parsed test_main_with_files. Retrieved 6/17 statements.
# Partially parsed test_main_wrong_sorted_files_exit. Retrieved 10/22 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '--show-config'
    var_4 = '--show-files'
    var_5 = 'test.py'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.main(var_6)
    var_8 = bool(var_0)
    assert var_8 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = False
    var_5 = 'sys.exit'
    var_6 = 'isort.main.api.sort_file'
    var_7 = True
    var_8 = lambda *args, **kwargs: var_7
    var_9 = '--settings-path'
    var_10 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = False
    var_5 = 'sys.exit'
    var_6 = 'isort.main.api.sort_file'
    var_7 = True
    var_8 = lambda *args, **kwargs: var_7
    var_9 = '--settings-path'
    var_10 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--virtual-env'
    var_3 = '/nonexistent/path'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = 'isort.main.api.check_stream'
    var_3 = True
    var_4 = lambda **kwargs: var_3
    var_5 = '--check-only'
    var_6 = '-'
    var_7 = [var_5, var_6]

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = 'isort.main.api.sort_stream'
    var_3 = None
    var_4 = lambda **kwargs: var_3
    var_5 = '-'
    var_6 = [var_5]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'somefile.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-files'
    var_3 = '-'
    var_4 = [var_2, var_3]
    var_5 = module_0.main(var_4)
    var_6 = bool(var_0)
    assert var_6 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'isort.main.api.sort_file'
    var_3 = True
    var_4 = lambda *args, **kwargs: var_3
    var_5 = 'isort.main.files.find'

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = 'isort.main.api.check_stream'
    var_4 = False
    var_5 = lambda **kwargs: var_4
    var_6 = 'import os\n'
    var_7 = [var_6]
    var_8 = '--check-only'
    var_9 = '-'
    var_10 = [var_8, var_9]
    var_11 = bool(var_0 and var_1 == 1)
    assert var_11 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_error. Retrieved 8/17 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 5/17 statements.
# Partially parsed test_main_show_config_with_file. Retrieved 3/14 statements.
# Partially parsed test_main_with_stdin_check. Retrieved 4/8 statements.
# Partially parsed test_main_dangerous_root_operation. Retrieved 6/15 statements.
# Partially parsed test_main_stream_filename_without_stdin. Retrieved 7/15 statements.
# Partially parsed test_main_with_settings_path_file. Retrieved 6/14 statements.
# Partially parsed test_main_with_virtual_env. Retrieved 5/17 statements.
# Partially parsed test_main_with_nonexistent_virtual_env. Retrieved 6/8 statements.
# Partially parsed test_main_parse_args_deprecated_single_dash. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_only. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_multi_line_output_digit. Retrieved 6/8 statements.
# Partially parsed test_main_parse_args_no_argv. Retrieved 5/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '--show-config'
    var_4 = '--show-files'
    var_5 = 'test.py'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.main(var_6)
    var_8 = bool(var_0)
    assert var_8 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = []
    var_3 = module_0.main(var_2)
    var_4 = 0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-config'

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '/'
    var_4 = [var_3]
    var_5 = module_0.main(var_4)
    var_6 = bool(var_0)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'file.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = '--settings-path'
    var_5 = '--show-config'

def test_case_0():
    var_0 = 'venv'
    var_1 = 'test.py'
    var_2 = 'import os\n'
    var_3 = '--virtual-env'
    var_4 = '--show-config'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = '.'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'
    var_5 = 'dont_order_by_type'
    var_6 = bool('dont_order_by_type' not in var_3)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'
    var_5 = 'dont_follow_links'
    var_6 = bool('dont_follow_links' not in var_3)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--dont-float-to-top'
    var_3 = '--float-to-top'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'float_to_top'
    var_5 = 'dont_float_to_top'
    var_6 = bool('dont_float_to_top' not in var_3)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = [var_1]
    var_3 = None
    var_4 = module_0.parse_args(var_3)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_parse_args_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_multi_line_output_numeric. Retrieved 7/9 statements.
# Partially parsed test_parse_args_multi_line_output_named. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_empty_values. Retrieved 2/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'verbose'
    var_4 = bool('verbose' in var_2)
    assert var_4 is True
    var_5 = var_2['verbose']
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--quiet'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'verbose'
    var_5 = bool('verbose' in var_3)
    assert var_5 is True
    var_6 = 'quiet'
    var_7 = bool('quiet' in var_3)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--src'
    var_1 = 'src_path'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'src'
    var_5 = bool('src' in var_3)
    assert var_5 is True
    var_6 = var_3['src']
    assert var_6 == 'src_path'

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = 'isort'
    var_6 = bool('isort' in var_2['remapped_deprecated_args'])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = bool('order_by_type' in var_2)
    assert var_4 is True
    var_5 = var_2['order_by_type']
    assert var_5 is False
    var_6 = 'dont_order_by_type'
    var_7 = bool('dont_order_by_type' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = bool('follow_links' in var_2)
    assert var_4 is True
    var_5 = var_2['follow_links']
    assert var_5 is False
    var_6 = 'dont_follow_links'
    var_7 = bool('dont_follow_links' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = bool('float_to_top' in var_2)
    assert var_4 is True
    var_5 = var_2['float_to_top']
    assert var_5 is False
    var_6 = 'dont_float_to_top'
    var_7 = bool('dont_float_to_top' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = 0
    var_9 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = 'remapped_deprecated_args'
    var_7 = var_3[var_6]
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'isort'
    var_10 = bool('isort' in var_3['remapped_deprecated_args'])
    assert var_10 is True
    var_11 = 'black'
    var_12 = bool('black' in var_3['remapped_deprecated_args'])
    assert var_12 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--src'
    var_2 = 'src'
    var_3 = '--dont-follow-links'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = 'verbose'
    var_7 = bool('verbose' in var_5)
    assert var_7 is True
    var_8 = 'src'
    var_9 = bool('src' in var_5)
    assert var_9 is True
    var_10 = var_5['src']
    assert var_10 == 'src'
    var_11 = 'follow_links'
    var_12 = bool('follow_links' in var_5)
    assert var_12 is True
    var_13 = var_5['follow_links']
    assert var_13 is False



# Parsed testcases at query #32
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.SortAttempt(var_0, var_0, var_0)
    var_2 = var_1.incorrectly_sorted
    assert var_2 is False
    var_3 = var_1.skipped
    assert var_3 is False
    var_4 = var_1.supported_encoding
    assert var_4 is False



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_resolve_all_configs_true. Retrieved 20/37 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'show_version'
    var_1 = 'show_config'
    var_2 = 'show_files'
    var_3 = 'resolve_all_configs'
    var_4 = 'config_root'
    var_5 = 'files'
    var_6 = 'check'
    var_7 = 'ask_to_apply'
    var_8 = 'jobs'
    var_9 = 'show_diff'
    var_10 = 'write_to_stdout'
    var_11 = 'deprecated_flags'
    var_12 = 'remapped_deprecated_args'
    var_13 = False
    var_14 = True
    var_15 = '.'
    var_16 = 'test.py'
    var_17 = [var_16]
    var_18 = None
    var_19 = module_0.main()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/12 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/4 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/15 statements.
# Partially parsed test_main_virtual_env_invalid. Retrieved 6/10 statements.
# Partially parsed test_main_show_config_with_file. Retrieved 3/10 statements.
# Partially parsed test_main_check_mode_with_stdin. Retrieved 4/8 statements.
# Partially parsed test_main_sort_mode_with_stdin. Retrieved 3/8 statements.
# Partially parsed test_main_with_valid_file. Retrieved 2/8 statements.
# Partially parsed test_main_with_check_flag. Retrieved 3/9 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_only. Retrieved 5/7 statements.
# Partially parsed test_main_with_show_files. Retrieved 3/10 statements.
# Partially parsed test_main_with_verbose. Retrieved 3/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = '--show-version'
    var_5 = [var_4]
    var_6 = module_0.main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '--show-files'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = '--settings-path'
    var_5 = 'settings_file'
    var_6 = 'settings_path'

import isort.main as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = '--virtual-env'
    var_2 = '/nonexistent/path'
    var_3 = '--show-config'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-config'
    var_3 = '{'

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = '--check'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'test.py'
    var_2 = 'somefile.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = '--check'

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line=0'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = "Can't set both"

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'float_to_top'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'
    var_3 = 'test.py'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--verbose'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_line_218_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = 0
    var_3 = var_0 > var_2



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_basic_args. Retrieved 4/6 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top_alone. Retrieved 4/5 statements.
# Partially parsed test_parse_args_both_float_to_top_options_exits. Retrieved 4/8 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 10/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'check'

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = 'isort'
    var_6 = bool('isort' in var_2['remapped_deprecated_args'])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = 'dont_float_to_top'
    var_5 = bool('dont_float_to_top' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = module_1.str(var_7)
    var_9 = var_3[var_6]
    var_10 = 'value'
    var_11 = hasattr(var_9, var_10)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = 'skip'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = 'remapped_deprecated_args'
    var_7 = var_3[var_6]
    var_8 = len(var_7)
    assert var_8 == 2

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = 'isort'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = 'isort'
    var_7 = bool('isort' in var_3['remapped_deprecated_args'])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_with_invalid_virtual_env. Retrieved 5/7 statements.
# Partially parsed test_main_check_mode_with_correctly_sorted_stream. Retrieved 4/10 statements.
# Partially parsed test_main_parse_args_with_deprecated_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_with_deprecated_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_with_dont_float_to_top. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_empty. Retrieved 2/4 statements.
# Partially parsed test_main_show_config_flag. Retrieved 4/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '--show-files'
    var_2 = '.'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--settings-path'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/venv'
    var_2 = '--show-config'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'test.py'
    var_2 = 'somefile.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'float_to_top'

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = "Can't set both"

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '.'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_error. Retrieved 9/18 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_with_settings_path_file. Retrieved 4/11 statements.
# Partially parsed test_main_with_virtual_env_invalid. Retrieved 5/10 statements.
# Partially parsed test_main_with_stream_input. Retrieved 3/7 statements.
# Partially parsed test_main_with_dangerous_root_path. Retrieved 5/13 statements.
# Partially parsed test_main_with_stream_filename_override. Retrieved 7/15 statements.
# Partially parsed test_main_parse_args_with_dont_order_by_type. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_with_dont_follow_links. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_with_dont_float_to_top. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_float_to_top_conflict. Retrieved 6/14 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '--show-config'
    var_4 = '--show-files'
    var_5 = 'test.py'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.main(var_6)
    var_8 = bool(var_0)
    assert var_8 is True
    var_9 = module_1.str(var_1)
    var_10 = 'either specify show-config or show-files not both'
    var_11 = bool('either specify show-config or show-files not both' in var_9)
    assert var_11 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = '--settings-path'
    var_3 = '--show-config'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)
    var_4 = 0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'somefile.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--float-to-top'
    var_3 = '--dont-float-to-top'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = bool(var_0)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_args_multi_line_output_predicate. Retrieved 6/25 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_22_evaluates_to_true. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'non_existent_venv'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'virtual_env'
    var_5 = 'files'
    var_6 = False
    var_7 = []
    var_8 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multiple_options. Retrieved 8/10 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--file-path'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'file_path'
    var_5 = bool('file_path' in var_3)
    assert var_5 is True
    var_6 = var_3['file_path']
    assert var_6 == 'test.py'

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = 'dont_float_to_top'
    var_5 = bool('dont_float_to_top' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = '--file-path'
    var_1 = 'test.py'
    var_2 = '--dont-order-by-type'
    var_3 = '--dont-follow-links'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = 'file_path'
    var_7 = bool('file_path' in var_5)
    assert var_7 is True
    var_8 = 'order_by_type'
    var_9 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'profile'
    var_5 = bool('profile' in var_3)
    assert var_5 is True
    var_6 = var_3['profile']
    assert var_6 == 'black'

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = 'force_single_line'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'remapped_deprecated_args'
    var_6 = bool('remapped_deprecated_args' in var_4)
    assert var_6 is True
    var_7 = 'force_single_line'
    var_8 = bool('force_single_line' in var_4['remapped_deprecated_args'])
    assert var_8 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = bool('order_by_type' in var_2)
    assert var_4 is True
    var_5 = var_2['order_by_type']
    assert var_5 is False
    var_6 = 'dont_order_by_type'
    var_7 = bool('dont_order_by_type' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = bool('follow_links' in var_2)
    assert var_4 is True
    var_5 = var_2['follow_links']
    assert var_5 is False
    var_6 = 'dont_follow_links'
    var_7 = bool('dont_follow_links' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = bool('float_to_top' in var_2)
    assert var_4 is True
    var_5 = var_2['float_to_top']
    assert var_5 is False
    var_6 = 'dont_float_to_top'
    var_7 = bool('dont_float_to_top' not in var_2)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 0
    var_7 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = '--line-length'
    var_3 = '88'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = 'profile'
    var_7 = bool('profile' in var_5)
    assert var_7 is True
    var_8 = 'line_length'
    var_9 = bool('line_length' in var_5)
    assert var_9 is True
    var_10 = var_5['profile']
    assert var_10 == 'black'
    var_11 = var_5['line_length']
    assert var_11 == '88'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 3/8 statements.
# Partially parsed test_main_with_settings_path_file. Retrieved 9/21 statements.
# Partially parsed test_main_with_virtual_env_valid. Retrieved 8/19 statements.
# Partially parsed test_main_with_virtual_env_invalid. Retrieved 9/15 statements.
# Partially parsed test_main_stdin_input. Retrieved 6/12 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 5/12 statements.
# Partially parsed test_main_parse_args_with_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_with_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_main_check_mode_with_unsorted_file. Retrieved 7/18 statements.
# Partially parsed test_main_show_files_flag. Retrieved 8/14 statements.
# Partially parsed test_main_with_src_paths. Retrieved 7/16 statements.
# Partially parsed test_main_show_config_output. Retrieved 6/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-config'
    var_3 = '--show-files'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 0

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = 'sys.exit'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = '--settings-path'
    var_8 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'venv'
    var_1 = 'test.py'
    var_2 = 'import os\n'
    var_3 = 'sys.exit'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = '--virtual-env'
    var_7 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '/nonexistent/venv/path'
    var_1 = 'test.py'
    var_2 = 'import os\n'
    var_3 = 'sys.exit'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = '--virtual-env'
    var_7 = [var_6, var_0, var_2]
    var_8 = module_0.main(var_7)

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = 'sys.exit'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = '-'
    var_6 = [var_5]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '--check'
    var_5 = [var_4, var_1]
    var_6 = module_0.main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.exit'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = '--show-files'
    var_6 = [var_5, var_1]
    var_7 = module_0.main(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool('remapped_deprecated_args' in var_3 or 'order_by_type' in var_3 or True)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.exit'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = '--src'
    var_6 = module_0.main(var_2)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'sys.exit'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = '--show-config'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_main_show_version. Retrieved 6/9 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_and_no_show_config. Retrieved 4/13 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/15 statements.
# Partially parsed test_main_virtual_env_invalid. Retrieved 8/13 statements.
# Partially parsed test_main_stream_input_check_mode. Retrieved 4/9 statements.
# Partially parsed test_main_recursive_root_without_allow_root. Retrieved 5/13 statements.
# Partially parsed test_main_stream_filename_override_error. Retrieved 7/15 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_conflict. Retrieved 7/15 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 3/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-config'
    var_3 = '--show-files'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = []
    var_3 = module_0.main(var_2)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = 'test.py'
    var_3 = 'import os\nimport sys\n'
    var_4 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    var_5 = len(var_0)
    var_6 = 0
    var_7 = var_5 >= var_6

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = 'test.py'
    var_3 = '--filename'
    var_4 = 'override.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'VERTICAL'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--float-to-top'
    var_3 = '--dont-float-to-top'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/nonexistent/file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = b'\xff\xfe'
    var_2 = {}
    var_3 = module_0.Config(**var_2)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = var_1 == var_2
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_show_config_predicate_evaluates_to_true. Retrieved 13/27 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'show_config'
    var_1 = 'show_files'
    var_2 = 'files'
    var_3 = 'settings_path'
    var_4 = True
    var_5 = False
    var_6 = []
    var_7 = '/tmp'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'key'
    var_10 = 'value'
    var_11 = []
    var_12 = module_0.main(var_11)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding_not_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 6/13 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 6/13 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is True
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is True
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_5)
    assert var_8 is True
    var_9 = var_7.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_5)
    assert var_8 is True
    var_9 = var_7.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_2, var_1, var_3, var_4, **var_5)
    var_7 = bool(var_2)
    assert var_7 is True
    var_8 = 1

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_2, var_1, var_3, write_to_stdout=var_4, **var_5)
    var_7 = bool(var_2)
    assert var_7 is True
    var_8 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_line_9_evaluates_to_true. Retrieved 12/20 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'show_version'
    var_1 = 'show_config'
    var_2 = 'show_files'
    var_3 = 'files'
    var_4 = False
    var_5 = True
    var_6 = []
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_5, var_3: var_6}
    var_8 = 'sys.exit'
    var_9 = '__main__.parse_args'
    var_10 = module_0.main()
    var_11 = 'Error: either specify show-config or show-files not both.'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_main_show_version. Retrieved 3/4 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 3/7 statements.
# Partially parsed test_main_settings_path_file. Retrieved 6/14 statements.
# Partially parsed test_main_stdin_check. Retrieved 4/7 statements.
# Partially parsed test_main_show_files. Retrieved 3/10 statements.
# Partially parsed test_main_dont_order_by_type. Retrieved 5/6 statements.
# Partially parsed test_main_dont_follow_links. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 7/14 statements.
# Partially parsed test_sort_imports_returns_none_on_os_error. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_returns_none_on_value_error. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '--show-files'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = 'test.py'
    var_3 = 'import os\nimport sys\n'
    var_4 = '--settings-path'
    var_5 = '--check'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'test.py'
    var_2 = '-'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'

import isort.main as module_0

def test_case_0():
    var_0 = 'order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = '--float-to-top'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'multi_line_output'
    var_6 = bool('multi_line_output' in var_4)
    assert var_6 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'isort.settings'
    var_3 = 'Config'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = True

def test_case_0():
    var_0 = 'isort.settings'
    var_1 = 'Config'
    var_2 = [var_1]
    var_3 = __import__(var_0, fromlist=var_2)
    var_4 = '/nonexistent/path/file.py'

def test_case_0():
    var_0 = 'isort.settings'
    var_1 = 'Config'
    var_2 = [var_1]
    var_3 = __import__(var_0, fromlist=var_2)
    var_4 = ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_sort_imports_isort_error_handling. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = 1
    var_3 = 'message'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_single_argument. Retrieved 4/6 statements.
# Partially parsed test_parse_args_with_multiple_arguments. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_file_input. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_combined_arguments. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'verbose'

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--check'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = 'dont_float_to_top'
    var_5 = bool('dont_float_to_top' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--file-path'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--check'
    var_2 = '--dont-order-by-type'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'order_by_type'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_main_show_version. Retrieved 6/9 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 4/13 statements.
# Partially parsed test_main_settings_path_file. Retrieved 8/20 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 5/18 statements.
# Partially parsed test_main_virtual_env_invalid. Retrieved 7/14 statements.
# Partially parsed test_main_stream_input_check. Retrieved 6/16 statements.
# Partially parsed test_main_stream_input_sort. Retrieved 5/15 statements.
# Partially parsed test_main_recursive_root_without_allow_root. Retrieved 5/12 statements.
# Partially parsed test_main_stream_filename_without_stream. Retrieved 7/14 statements.
# Partially parsed test_main_show_files_with_stream. Retrieved 6/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-config'
    var_3 = '--show-files'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = []
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '--settings-path'
    var_5 = '--show-config'
    var_6 = [var_4, var_1, var_5]
    var_7 = module_0.main(var_6)

def test_case_0():
    var_0 = 'test_dir'
    var_1 = False
    var_2 = 'sys.exit'
    var_3 = '--settings-path'
    var_4 = '--show-config'

import isort.main as module_0

def test_case_0():
    var_0 = '/nonexistent/venv'
    var_1 = False
    var_2 = 'sys.exit'
    var_3 = '--virtual-env'
    var_4 = '--show-config'
    var_5 = [var_3, var_0, var_4]
    var_6 = module_0.main(var_5)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '-'
    var_5 = '--check'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '-'
    var_5 = [var_4]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'somefile.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(var_0)
    assert var_7 is True

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = 'import os\n'
    var_3 = [var_2]
    var_4 = '-'
    var_5 = '--show-files'
    var_6 = [var_4, var_5]
    var_7 = bool(var_0)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_help_flag. Retrieved 2/3 statements.
# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 5/6 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multiple_arguments. Retrieved 6/8 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool('src' in var_2 or var_2 == {})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--force-single-line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 0
    var_7 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = 'dont_float_to_top'
    var_5 = bool('dont_float_to_top' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = '--dont-follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'
    var_5 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 12/17 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 (show_config and show_files) evaluates to True.'
    var_1 = 'sys.exit'
    var_2 = 'isort.main.parse_args'
    var_3 = 'show_version'
    var_4 = 'show_config'
    var_5 = 'show_files'
    var_6 = False
    var_7 = True
    var_8 = {var_3: var_6, var_4: var_7, var_5: var_7}
    var_9 = []
    var_10 = module_0.main(var_9)
    var_11 = 'Error: either specify show-config or show-files not both.'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_main_show_version. Retrieved 4/7 statements.
# Partially parsed test_main_no_arguments_no_files. Retrieved 4/10 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/13 statements.
# Partially parsed test_main_show_config_flag. Retrieved 3/9 statements.
# Partially parsed test_main_show_files_flag. Retrieved 4/16 statements.
# Partially parsed test_main_with_stream_check. Retrieved 4/7 statements.
# Partially parsed test_main_with_stream_no_check. Retrieved 3/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '--show-files'
    var_2 = '.'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 'isort'
    var_3 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = 'arguments passed in without any paths or content'

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black\n'
    var_4 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--help'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    assert var_4 is None

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'test.py'
    var_2 = 'somefile.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    var_5 = 'Filename override is intended only for stream'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-config'
    var_3 = '{'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'
    var_3 = 0

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]
    var_4 = bool(True)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--show-files'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)
    var_4 = "can't show files for streaming input"



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 14/20 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'show_version'
    var_1 = 'show_config'
    var_2 = 'show_files'
    var_3 = 'settings_path'
    var_4 = 'files'
    var_5 = False
    var_6 = True
    var_7 = '/test/path'
    var_8 = 'test.py'
    var_9 = [var_8]
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_6, var_3: var_7, var_4: var_9}
    var_11 = None
    var_12 = module_0.main(var_11, var_11)
    var_13 = 'Error: either specify show-config or show-files not both.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parse_args_multi_line_output_predicate_true. Retrieved 16/21 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = 'order_by_type'
    var_2 = 'follow_links'
    var_3 = 'float_to_top'
    var_4 = 'dont_order_by_type'
    var_5 = 'dont_follow_links'
    var_6 = 'dont_float_to_top'
    var_7 = 'remapped_deprecated_args'
    var_8 = '1'
    var_9 = None
    var_10 = False
    var_11 = {var_0: var_8, var_1: var_9, var_2: var_9, var_3: var_9, var_4: var_10, var_5: var_10, var_6: var_10, var_7: var_9}
    var_12 = '--multi-line-output'
    var_13 = '1'
    var_14 = [var_12, var_13]
    var_15 = module_0.parse_args(var_14)
    var_16 = 'multi_line_output'
    var_17 = bool('multi_line_output' in var_15)
    assert var_17 is True
    var_18 = var_15['multi_line_output']
    var_19 = bool(var_15['multi_line_output'] is not None)
    assert var_19 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = True
    var_2 = var_0 and var_1
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_parse_args_argv_none_uses_sys_argv. Retrieved 5/11 statements.
# Partially parsed test_parse_args_argv_provided_converts_to_list. Retrieved 4/5 statements.
# Partially parsed test_parse_args_argv_none_predicate. Retrieved 6/13 statements.
# Partially parsed test_parse_args_argv_provided_predicate. Retrieved 7/10 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--profile'
    var_2 = 'black'
    var_3 = None
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

def test_case_0():
    var_0 = 'script.py'
    var_1 = None
    var_2 = None
    var_3 = var_1 is var_2
    var_4 = 1
    var_5 = list(var_1)

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = var_2 is var_3
    var_5 = 1
    var_6 = list(var_2)



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_98_evaluates_to_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = False
    var_3 = var_0 in var_1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_parse_args_multi_line_output_truthy. Retrieved 12/20 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = 'dont_order_by_type'
    var_2 = 'dont_follow_links'
    var_3 = 'dont_float_to_top'
    var_4 = 'float_to_top'
    var_5 = '1'
    var_6 = False
    var_7 = {var_0: var_5, var_1: var_6, var_2: var_6, var_3: var_6, var_4: var_6}
    var_8 = '--multi-line-output'
    var_9 = '1'
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = 'multi_line_output'
    var_13 = bool('multi_line_output' in var_11)
    assert var_13 is True
    var_14 = var_11['multi_line_output']
    var_15 = bool(var_11['multi_line_output'] is not None)
    assert var_15 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/19 statements.
# Partially parsed test_identify_imports_main_with_file. Retrieved 4/16 statements.
# Partially parsed test_identify_imports_main_with_unique_package. Retrieved 7/18 statements.
# Partially parsed test_identify_imports_main_with_unique_module. Retrieved 7/18 statements.
# Partially parsed test_identify_imports_main_with_unique_attribute. Retrieved 7/18 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 5/15 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = 'api.find_imports_in_stream'
    var_3 = '-'
    var_4 = [var_3]
    var_5 = 'os'
    var_6 = 'sys'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = 'os'
    var_5 = 'sys'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'api.ImportKey.PACKAGE'
    var_2 = 'package'
    var_3 = 'test.py'
    var_4 = '--packages'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'os'
    var_8 = 'sys'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'api.ImportKey.MODULE'
    var_2 = 'module'
    var_3 = 'test.py'
    var_4 = '--modules'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'os'
    var_8 = 'sys'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'api.ImportKey.ATTRIBUTE'
    var_2 = 'attribute'
    var_3 = 'test.py'
    var_4 = '--attributes'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)
    var_7 = 'os.path'
    var_8 = 'sys.argv'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--top-only'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'os'

import isort.main as module_0

def test_case_0():
    var_0 = 'api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--follow-links'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)
    var_5 = 'os'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_22_evaluates_to_true. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'nonexistent_venv'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'virtual_env'
    var_5 = 'files'
    var_6 = False
    var_7 = []
    var_8 = 'virtual_env'



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = 'test_file.py'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_args_remapped_deprecated_args. Retrieved 5/15 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'some_arg'
    var_1 = 'value'
    var_2 = 'help'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'remapped_deprecated_args'
    var_6 = bool('remapped_deprecated_args' in var_4)
    assert var_6 is True
    var_7 = var_4['remapped_deprecated_args']
    var_8 = bool(var_4['remapped_deprecated_args'] == ['help'])
    assert var_8 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_preconvert_set. Retrieved 6/8 statements.
# Partially parsed test_preconvert_frozenset. Retrieved 7/9 statements.
# Partially parsed test_preconvert_path. Retrieved 3/5 statements.
# Failed to parse test_preconvert_callable_with_name.
# Partially parsed test_preconvert_lambda. Retrieved 2/4 statements.
# Failed to parse test_preconvert_builtin_function.
# Failed to parse test_preconvert_invalid_type.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)
    var_5 = set(var_4)
    var_6 = bool(var_5 == {1, 2, 3})
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)
    var_6 = set(var_5)
    var_7 = bool(var_6 == {1, 2, 3})
    assert var_7 is True

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/home/user/file.txt'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/home/user/file.txt'

import isort.main as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0._preconvert(var_0)
    assert var_1 == '<lambda>'

import isort.main as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0._preconvert(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unserializable object'

import isort.main as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0._preconvert(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unserializable object'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false_supported_encoding. Retrieved 4/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = var_6.supported_encoding
    assert var_7 is False
    var_8 = var_6.incorrectly_sorted
    assert var_8 is False
    var_9 = var_6.skipped
    assert var_9 is False



