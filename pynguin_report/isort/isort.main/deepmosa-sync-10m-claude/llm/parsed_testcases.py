####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose_off. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose_on. Retrieved 5/12 statements.
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_args_with_single_argument. Retrieved 5/6 statements.
# Partially parsed test_parse_args_with_multiple_arguments. Retrieved 8/10 statements.
# Partially parsed test_parse_args_dont_order_by_type_conversion. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links_conversion. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top_alone. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_with_digit. Retrieved 6/8 statements.
# Partially parsed test_parse_args_multi_line_output_with_name. Retrieved 5/6 statements.
# Partially parsed test_parse_args_filters_out_falsy_values. Retrieved 4/7 statements.
# Partially parsed test_parse_args_with_combined_deprecated_and_regular_args. Retrieved 6/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'line_length'

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = '--profile'
    var_3 = 'black'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = 'line_length'
    var_7 = 'profile'

import isort.main as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = 'settings'
    var_6 = bool('settings' in var_2['remapped_deprecated_args'])
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
    var_5 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = '--line-length'
    var_2 = '100'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'remapped_deprecated_args'
    var_6 = bool('remapped_deprecated_args' in var_4)
    assert var_6 is True
    var_7 = 'line_length'
    var_8 = 'settings'
    var_9 = bool('settings' in var_4['remapped_deprecated_args'])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_caught. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = bool(var_0)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 14/25 statements.


def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'line_length'
    var_2 = {var_0, var_1}
    var_3 = 'MockParser'
    var_4 = ()
    var_5 = 'parse_args'
    var_6 = 'Namespace'
    var_7 = ()
    var_8 = '__dict__'
    var_9 = {}
    var_10 = lambda x: var_9
    var_11 = property(var_10)
    var_12 = {var_8: var_11}
    var_13 = [var_6, var_7, var_12]
    var_14 = 'force_single_line'
    var_15 = bool(var_14 in var_2)
    assert var_15 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 6/25 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 6/23 statements.
# Partially parsed test_identify_imports_main_with_unique_package. Retrieved 6/24 statements.
# Partially parsed test_identify_imports_main_with_unique_module. Retrieved 6/24 statements.
# Partially parsed test_identify_imports_main_with_unique_attribute. Retrieved 7/25 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 6/21 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 6/21 statements.
# Partially parsed test_identify_imports_main_with_unique_flag. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = 'find_imports_in_stream'
    var_7 = 'os'
    var_8 = 'sys'

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'json'
    var_3 = 're'
    var_4 = 'find_imports_in_paths'
    var_5 = module_0.identify_imports_main(var_1)
    var_6 = 'json'
    var_7 = 're'

def test_case_0():
    var_0 = '-'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = 'from os.path import join\n'
    var_4 = [var_3]
    var_5 = 'os.path'
    var_6 = 'find_imports_in_stream'
    var_7 = 'os'

def test_case_0():
    var_0 = '-'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = 'from os.path import join\n'
    var_4 = [var_3]
    var_5 = 'os.path'
    var_6 = 'find_imports_in_stream'
    var_7 = 'os.path'

def test_case_0():
    var_0 = '-'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = 'from os.path import join\n'
    var_4 = [var_3]
    var_5 = 'os.path'
    var_6 = 'join'
    var_7 = 'find_imports_in_stream'
    var_8 = 'os.path.join'

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'find_imports_in_paths'
    var_5 = module_0.identify_imports_main(var_2)
    var_6 = 'os'

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'find_imports_in_paths'
    var_5 = module_0.identify_imports_main(var_2)
    var_6 = 'os'

def test_case_0():
    var_0 = '-'
    var_1 = '--unique'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'find_imports_in_stream'
    var_6 = 'import os\nimport sys\n'
    var_7 = [var_6]
    var_8 = 'os'
    var_9 = 'sys'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sort_imports_isort_error_handling. Retrieved 6/17 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'isort.api.check_file'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_0, var_3, var_5, **var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_76_evaluates_to_true. Retrieved 18/50 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = []
    var_2 = 'files'
    var_3 = '+'
    var_4 = '--top-only'
    var_5 = 'store_true'
    var_6 = False
    var_7 = '--follow-links'
    var_8 = '--unique'
    var_9 = '--packages'
    var_10 = 'unique'
    var_11 = 'store_const'
    var_12 = '--modules'
    var_13 = '--attributes'
    var_14 = 'test.py'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = False
    var_18 = True
    assert var_18 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_args_float_to_top_predicate_true. Retrieved 10/21 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = 'multi_line_output'
    var_3 = True
    var_4 = None
    var_5 = {var_0: var_3, var_1: var_3, var_2: var_4}
    var_6 = '--dont-float-to-top'
    var_7 = '--float-to-top'
    var_8 = [var_6, var_7]
    var_9 = module_0.parse_args(var_8)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_caught. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_multi_line_output_numeric. Retrieved 7/9 statements.
# Partially parsed test_parse_args_multi_line_output_string. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_out_falsy_values. Retrieved 4/7 statements.


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
    var_4 = var_3['profile']
    assert var_4 == 'black'

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = '--line-length'
    var_3 = '88'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = var_5['profile']
    assert var_6 == 'black'
    var_7 = var_5['line_length']
    assert var_7 == '88'

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

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = var_2['order_by_type']
    assert var_3 is False
    var_4 = 'dont_order_by_type'
    var_5 = bool('dont_order_by_type' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = var_2['follow_links']
    assert var_3 is False
    var_4 = 'dont_follow_links'
    var_5 = bool('dont_follow_links' not in var_2)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = var_2['float_to_top']
    assert var_3 is False
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
    var_5 = var_3[var_4]
    var_6 = 0
    var_7 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'force_sort_within_sections'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = 'remapped_deprecated_args'
    var_7 = var_3[var_6]
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'force_single_line'
    var_10 = bool('force_single_line' in var_3['remapped_deprecated_args'])
    assert var_10 is True
    var_11 = 'force_sort_within_sections'
    var_12 = bool('force_sort_within_sections' in var_3['remapped_deprecated_args'])
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_no_arguments. Retrieved 4/10 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 8/15 statements.
# Partially parsed test_main_settings_path_file. Retrieved 13/28 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 6/11 statements.
# Partially parsed test_main_stdin_check_mode. Retrieved 5/13 statements.
# Partially parsed test_main_stdin_sort_mode. Retrieved 4/11 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 6/13 statements.
# Partially parsed test_main_filename_override_without_stdin. Retrieved 8/15 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_float_to_top_conflict. Retrieved 7/14 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_write_to_stdout. Retrieved 4/10 statements.


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
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 'usage'
    var_3 = 'quick'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '--show-config'
    var_3 = '--show-files'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = len(var_0)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = 'setup.cfg'
    var_3 = '[isort]\nprofile=black'
    var_4 = 'sys.exit'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = '--settings-path'
    var_8 = '--show-config'
    var_9 = [var_7, var_1, var_8]
    var_10 = module_0.main(var_9)
    var_11 = 'settings'
    var_12 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '/nonexistent/venv/path'
    var_1 = '--virtual-env'
    var_2 = 'test.py'
    var_3 = [var_1, var_0, var_2]
    var_4 = module_0.main(var_3)
    var_5 = len(var_1)
    var_6 = bool(var_5 >= 0)
    assert var_6 is True

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--check'
    var_4 = [var_2, var_3]
    var_5 = len(var_3)
    var_6 = bool(var_5 >= 0)
    assert var_6 is True

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 0)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = len(var_0)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'somefile.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = len(var_0)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line'
    var_1 = 'grid'
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
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '--float-to-top'
    var_3 = '--dont-float-to-top'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = len(var_0)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = True

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
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

import isort.main as module_0

def test_case_0():
    var_0 = '-sp'
    var_1 = '/tmp'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 31 evaluates to True when 'settings_path' is not in arguments."
    var_1 = 'files'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = False
    var_7 = {var_1: var_5, var_2: var_6, var_3: var_6}
    var_8 = 'settings_path'
    var_9 = var_8 not in var_7
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_main_predicate_line_1. Retrieved 5/10 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.main(var_0, var_0)
    assert var_1 is None
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = module_0.main(var_3, var_0)
    assert var_4 is None
    var_5 = 'argv'
    var_6 = 'stdin'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_args_float_to_top_predicate_evaluates_to_true. Retrieved 11/22 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = 'other_arg'
    var_3 = True
    var_4 = False
    var_5 = {var_0: var_3, var_1: var_3, var_2: var_4}
    var_6 = '--dont-float-to-top'
    var_7 = '--float-to-top'
    var_8 = [var_6, var_7]
    var_9 = module_0.parse_args(var_8)
    var_10 = "Can't set both --float-to-top and --dont-float-to-top."



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_parse_args_argv_none_uses_sys_argv. Retrieved 4/10 statements.
# Partially parsed test_parse_args_argv_provided_converts_to_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_predicate_line_2_evaluates_true. Retrieved 6/14 statements.
# Partially parsed test_parse_args_predicate_with_sequence. Retrieved 8/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'program'
    var_1 = '--verbose'
    var_2 = None
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

def test_case_0():
    var_0 = 'program'
    var_1 = None
    var_2 = None
    var_3 = var_1 is var_2
    var_4 = 1
    var_5 = list(var_1)

def test_case_0():
    var_0 = None
    var_1 = '--verbose'
    var_2 = '--check'
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = var_0 is var_4
    var_6 = 1
    var_7 = list(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_no_valid_encodings_predicate_false. Retrieved 27/43 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = 'show_version'
    var_3 = 'show_config'
    var_4 = 'show_files'
    var_5 = 'files'
    var_6 = 'filename'
    var_7 = 'check'
    var_8 = 'ask_to_apply'
    var_9 = 'jobs'
    var_10 = 'show_diff'
    var_11 = 'write_to_stdout'
    var_12 = 'deprecated_flags'
    var_13 = 'remapped_deprecated_args'
    var_14 = 'ext_format'
    var_15 = 'allow_root'
    var_16 = 'resolve_all_configs'
    var_17 = 'settings_path'
    var_18 = False
    var_19 = '-'
    var_20 = [var_19]
    var_21 = None
    var_22 = '/tmp'
    var_23 = {var_2: var_18, var_3: var_18, var_4: var_18, var_5: var_20, var_6: var_21, var_7: var_18, var_8: var_18, var_9: var_21, var_10: var_18, var_11: var_18, var_12: var_18, var_13: var_18, var_14: var_21, var_15: var_21, var_16: var_18, var_17: var_22}
    var_24 = 0
    var_25 = 1
    var_26 = [call for call in var_2 if call[var_24][var_24] == var_25]
    var_27 = len(var_26)
    assert var_27 == 0



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_76_evaluates_to_true. Retrieved 20/44 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'files'
    var_1 = '+'
    var_2 = '--top-only'
    var_3 = 'store_true'
    var_4 = False
    var_5 = '--follow-links'
    var_6 = '--unique'
    var_7 = '--packages'
    var_8 = 'unique'
    var_9 = 'store_const'
    var_10 = 'package'
    var_11 = '--modules'
    var_12 = 'module'
    var_13 = '--attributes'
    var_14 = 'attribute'
    var_15 = 'test.py'
    var_16 = [var_15]
    var_17 = module_0.parse_args(var_16)
    var_18 = False
    var_19 = True
    assert var_19 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_28_evaluates_to_true. Retrieved 13/20 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'files'
    var_5 = 'some_argument'
    var_6 = False
    var_7 = []
    var_8 = 'value'
    var_9 = 'sys.exit'
    var_10 = 'builtins.print'
    var_11 = module_0.main()
    var_12 = 'Error: arguments passed in without any paths or content.'



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = 0
    var_3 = var_0 > var_2
    var_4 = var_3 and var_1
    assert var_4 is False



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 106 (if stream_filename:) evaluates to True.'
    var_1 = 'test_file.py'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/12 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.


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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_6 = bool(False)
    assert var_6 is True

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_98_evaluates_to_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = False
    var_3 = var_0 in var_1



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 11/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.exit'
    var_1 = '__main__.parse_args'
    var_2 = 'show_version'
    var_3 = 'show_config'
    var_4 = 'show_files'
    var_5 = 'files'
    var_6 = False
    var_7 = True
    var_8 = []
    var_9 = module_0.main()
    var_10 = 'Error: either specify show-config or show-files not both.'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 4/10 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 3/4 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--verbose'
    var_2 = None
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

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
    var_0 = '--src'
    var_1 = 'path/to/file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'src'
    var_5 = bool('src' in var_3)
    assert var_5 is True
    var_6 = var_3['src']
    assert var_6 == 'path/to/file.py'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]
    var_8 = 'value'
    var_9 = hasattr(var_7, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

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
    var_8 = 'value'
    var_9 = hasattr(var_7, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--src'
    var_2 = 'test.py'
    var_3 = '--line-length'
    var_4 = '88'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = 'verbose'
    var_8 = bool('verbose' in var_6)
    assert var_8 is True
    var_9 = 'src'
    var_10 = bool('src' in var_6)
    assert var_10 is True
    var_11 = 'line_length'
    var_12 = bool('line_length' in var_6)
    assert var_12 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'dont_float_to_top'
    var_4 = bool('dont_float_to_top' not in var_2)
    assert var_4 is True
    var_5 = 'float_to_top'
    var_6 = bool('float_to_top' in var_2)
    assert var_6 is True
    var_7 = var_2['float_to_top']
    assert var_7 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_main_show_version. Retrieved 6/11 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 8/12 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 5/9 statements.
# Partially parsed test_main_settings_path_file. Retrieved 12/30 statements.
# Partially parsed test_main_virtual_env_nonexistent. Retrieved 14/19 statements.
# Partially parsed test_main_stdin_check_mode. Retrieved 16/21 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 16/20 statements.
# Partially parsed test_main_filename_override_with_stdin. Retrieved 18/22 statements.
# Partially parsed test_main_show_files. Retrieved 22/38 statements.
# Partially parsed test_main_parse_args_deprecated_single_dash. Retrieved 3/5 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_file_skipped. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 6/10 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'exit'
    var_1 = None
    var_2 = lambda x: var_1
    var_3 = '--show-config'
    var_4 = '--show-files'
    var_5 = 'test.py'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.main(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = 'exit'
    var_1 = None
    var_2 = lambda x: var_1
    var_3 = []
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nprofile=black\n'
    var_2 = 'test.py'
    var_3 = 'import os\nimport sys\n'
    var_4 = 'isort.main.parse_args'
    var_5 = 'show_version'
    var_6 = 'show_config'
    var_7 = 'show_files'
    var_8 = 'settings_path'
    var_9 = 'files'
    var_10 = False
    var_11 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'virtual_env'
    var_5 = 'files'
    var_6 = False
    var_7 = '/nonexistent/venv'
    var_8 = []
    var_9 = {var_1: var_6, var_2: var_6, var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = lambda x: var_9
    var_11 = '--virtual-env'
    var_12 = [var_11, var_7]
    var_13 = module_0.main(var_12)

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = 'isort.main.parse_args'
    var_3 = 'show_version'
    var_4 = 'show_config'
    var_5 = 'show_files'
    var_6 = 'files'
    var_7 = 'check'
    var_8 = 'filename'
    var_9 = False
    var_10 = '-'
    var_11 = [var_10]
    var_12 = True
    var_13 = None
    var_14 = {var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_11, var_7: var_12, var_8: var_13}
    var_15 = lambda x: var_14
    var_16 = [var_10]

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'files'
    var_5 = 'allow_root'
    var_6 = False
    var_7 = '/'
    var_8 = [var_7]
    var_9 = None
    var_10 = {var_1: var_6, var_2: var_6, var_3: var_6, var_4: var_8, var_5: var_9}
    var_11 = lambda x: var_10
    var_12 = 'exit'
    var_13 = lambda x: var_9
    var_14 = [var_7]
    var_15 = module_0.main(var_14)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'files'
    var_5 = 'filename'
    var_6 = False
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = 'override.py'
    var_10 = {var_1: var_6, var_2: var_6, var_3: var_6, var_4: var_8, var_5: var_9}
    var_11 = lambda x: var_10
    var_12 = 'exit'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = '--filename'
    var_16 = [var_15, var_9, var_7]
    var_17 = module_0.main(var_16)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = 'isort.main.parse_args'
    var_3 = 'show_version'
    var_4 = 'show_config'
    var_5 = 'show_files'
    var_6 = 'files'
    var_7 = False
    var_8 = True
    var_9 = 'isort.main.Config'
    var_10 = 'Config'
    var_11 = ()
    var_12 = '__dict__'
    var_13 = 'verbose'
    var_14 = 'quiet'
    var_15 = 'color_output'
    var_16 = 'format_error'
    var_17 = 'format_success'
    var_18 = {}
    var_19 = ''
    var_20 = {var_12: var_18, var_13: var_7, var_14: var_7, var_15: var_7, var_16: var_19, var_17: var_19}
    var_21 = [var_10, var_11, var_20]
    var_22 = '--show-config'

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'isort.main.api.check_file'
    var_3 = True
    var_4 = lambda *args, **kwargs: var_3
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_1, var_3, **var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'isort.main.api.check_file'
    var_3 = 'test.py'
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6.skipped
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'isort.main.api.check_file'
    var_3 = 'File error'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = True
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_1, var_6, **var_7)
    assert var_8 is None



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = bool(var_1 == ['-'])
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_settings_path_file. Retrieved 6/16 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 4/11 statements.
# Partially parsed test_main_virtual_env_missing. Retrieved 6/8 statements.
# Partially parsed test_main_check_flag. Retrieved 3/9 statements.
# Partially parsed test_main_show_files. Retrieved 4/17 statements.
# Partially parsed test_main_recursive_on_root_with_allow_root. Retrieved 3/10 statements.
# Partially parsed test_main_config_show. Retrieved 3/10 statements.
# Partially parsed test_main_parse_args_multi_line_output_digit. Retrieved 3/8 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'argv'
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
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    var_5 = bool(False)
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'arguments passed in without any paths or content'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\n'
    var_4 = '--settings-path'
    var_5 = '--show-files'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys'
    var_2 = '--settings-path'
    var_3 = '--show-files'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-files'
    var_3 = '.'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os'
    var_2 = '--check'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '--show-files'
    var_3 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--filename'
    var_2 = 'test.py'
    var_3 = '--stream-filename'
    var_4 = 'other.py'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Filename override is intended only for stream'

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '--allow-root'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '--show-config'
    var_3 = '{'

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'test.py'
    var_3 = 'multi_line_output'

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Can't set both --float-to-top and --dont-float-to-top"



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_build_arg_parser. Retrieved 96/98 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0._build_arg_parser()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = '-h'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = '--version'
    var_7 = [var_6]
    var_8 = module_0.parse_args(var_7)
    var_9 = var_8.show_version
    assert var_9 is True
    var_10 = '-v'
    var_11 = [var_10]
    var_12 = module_0.parse_args(var_11)
    var_13 = var_12.verbose
    assert var_13 is True
    var_14 = '-q'
    var_15 = [var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = var_16.quiet
    assert var_17 is True
    var_18 = '-c'
    var_19 = [var_18]
    var_20 = module_0.parse_args(var_19)
    var_21 = var_20.check
    assert var_21 is True
    var_22 = '--df'
    var_23 = [var_22]
    var_24 = module_0.parse_args(var_23)
    var_25 = var_24.show_diff
    assert var_25 is True
    var_26 = '-d'
    var_27 = [var_26]
    var_28 = module_0.parse_args(var_27)
    var_29 = var_28.write_to_stdout
    assert var_29 is True
    var_30 = 'file1.py'
    var_31 = 'file2.py'
    var_32 = [var_30, var_31]
    var_33 = module_0.parse_args(var_32)
    var_34 = var_33.files
    var_35 = bool(var_33.files == ['file1.py', 'file2.py'])
    assert var_35 is True
    var_36 = '--skip'
    var_37 = 'file.py'
    var_38 = [var_36, var_37]
    var_39 = module_0.parse_args(var_38)
    var_40 = var_39.skip
    var_41 = bool(var_39.skip == ['file.py'])
    assert var_41 is True
    var_42 = '-a'
    var_43 = 'import os'
    var_44 = [var_42, var_43]
    var_45 = module_0.parse_args(var_44)
    var_46 = var_45.add_imports
    var_47 = bool(var_45.add_imports == ['import os'])
    assert var_47 is True
    var_48 = '--rm'
    var_49 = 'import sys'
    var_50 = [var_48, var_49]
    var_51 = module_0.parse_args(var_50)
    var_52 = var_51.remove_imports
    var_53 = bool(var_51.remove_imports == ['import sys'])
    assert var_53 is True
    var_54 = '-i'
    var_55 = '  '
    var_56 = [var_54, var_55]
    var_57 = module_0.parse_args(var_56)
    var_58 = var_57.indent
    assert var_58 == '  '
    var_59 = '-j'
    var_60 = '4'
    var_61 = [var_59, var_60]
    var_62 = module_0.parse_args(var_61)
    var_63 = var_62.jobs
    assert var_63 == 4
    var_64 = '--profile'
    var_65 = 'black'
    var_66 = [var_64, var_65]
    var_67 = module_0.parse_args(var_66)
    var_68 = var_67.profile
    assert var_68 == 'black'
    var_69 = '--sp'
    var_70 = '/path/to/config'
    var_71 = [var_69, var_70]
    var_72 = module_0.parse_args(var_71)
    var_73 = var_72.settings_path
    assert var_73 == '/path/to/config'
    var_74 = '--ac'
    var_75 = [var_74]
    var_76 = module_0.parse_args(var_75)
    var_77 = var_76.atomic
    assert var_77 is True
    var_78 = '--interactive'
    var_79 = [var_78]
    var_80 = module_0.parse_args(var_79)
    var_81 = var_80.ask_to_apply
    assert var_81 is True
    var_82 = '--ca'
    var_83 = [var_82]
    var_84 = module_0.parse_args(var_83)
    var_85 = var_84.combine_as_imports
    assert var_85 is True
    var_86 = '--fgw'
    var_87 = '3'
    var_88 = [var_86, var_87]
    var_89 = module_0.parse_args(var_88)
    var_90 = var_89.force_grid_wrap
    assert var_90 == 3
    var_91 = '-m'
    var_92 = [var_91, var_87]
    var_93 = module_0.parse_args(var_92)
    var_94 = var_93.multi_line_output
    assert var_94 == '3'
    var_95 = '--ls'
    var_96 = [var_95]
    var_97 = module_0.parse_args(var_96)
    var_98 = var_97.length_sort
    assert var_98 is True
    var_99 = '--reverse-sort'
    var_100 = [var_99]
    var_101 = module_0.parse_args(var_100)
    var_102 = var_101.reverse_sort
    assert var_102 is True
    var_103 = '--ot'
    var_104 = [var_103]
    var_105 = module_0.parse_args(var_104)
    var_106 = var_105.order_by_type
    assert var_106 is True
    var_107 = '--show-config'
    var_108 = [var_107]
    var_109 = module_0.parse_args(var_108)
    var_110 = var_109.show_config
    assert var_110 is True
    var_111 = '--show-files'
    var_112 = [var_111]
    var_113 = module_0.parse_args(var_112)
    var_114 = var_113.show_files
    assert var_114 is True
    var_115 = '--sg'
    var_116 = '*.pyc'
    var_117 = [var_115, var_116]
    var_118 = module_0.parse_args(var_117)
    var_119 = var_118.skip_glob
    var_120 = bool(var_118.skip_glob == ['*.pyc'])
    assert var_120 is True
    var_121 = '--gitignore'
    var_122 = [var_121]
    var_123 = module_0.parse_args(var_122)
    var_124 = var_123.skip_gitignore
    assert var_124 is True
    var_125 = '--filename'
    var_126 = 'test.py'
    var_127 = [var_125, var_126]
    var_128 = module_0.parse_args(var_127)
    var_129 = var_128.filename
    assert var_129 == 'test.py'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/12 statements.
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_6 = bool(False)
    assert var_6 is True

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

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 4/12 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.


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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is False

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
    var_7 = var_5.supported_encoding
    assert var_7 is False

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
    var_6 = bool(False)
    assert var_6 is True

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 4/13 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.


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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is True
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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_2)
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is False

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
    var_7 = var_5.supported_encoding
    assert var_7 is False

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error_exits. Retrieved 4/12 statements.
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
    var_6 = bool(False)
    assert var_6 is True

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
    var_8 = var_6.supported_encoding
    assert var_8 is True

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
    var_8 = var_6.supported_encoding
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_converts_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_converts_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_handles_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_converts_multi_line_output_digit. Retrieved 7/10 statements.
# Partially parsed test_parse_args_none_argv_uses_sys_argv. Retrieved 2/3 statements.


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
    var_1 = '--check'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'verbose'
    var_5 = bool('verbose' in var_3)
    assert var_5 is True
    var_6 = 'check'
    var_7 = bool('check' in var_3)
    assert var_7 is True

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
    var_0 = []
    var_1 = module_0.parse_args(var_0)

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
    var_0 = '--src'
    var_1 = '/path/to/file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'src'
    var_5 = bool('src' in var_3)
    assert var_5 is True
    var_6 = var_3['src']
    assert var_6 == '/path/to/file.py'

import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_remapped_deprecated_args_added_to_arguments. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = 'some_deprecated_arg'
    var_3 = [var_2]
    var_4 = []
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = len(var_4)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 'some_deprecated_arg'
    var_9 = bool('some_deprecated_arg' in var_4)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_check_file_skipped. Retrieved 2/13 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_sort_file_skipped. Retrieved 2/13 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 2/13 statements.
# Partially parsed test_sort_imports_value_error. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 2/14 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False
    var_2 = bool(var_0)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_config. Retrieved 2/6 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/14 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 3/10 statements.
# Partially parsed test_main_stream_input_check. Retrieved 4/8 statements.
# Partially parsed test_main_stream_input_show_files_error. Retrieved 6/17 statements.
# Partially parsed test_main_recursive_root_without_allow_root. Retrieved 5/13 statements.
# Partially parsed test_main_stream_filename_without_stream. Retrieved 7/15 statements.
# Partially parsed test_main_show_files. Retrieved 4/17 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 4/6 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_conflict. Retrieved 6/14 statements.
# Partially parsed test_main_parse_args_dont_float_to_top_only. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=80\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = '--settings-path'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--files'
    var_3 = 'dummy.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = [var_0]
    var_2 = '--check-only'
    var_3 = '-'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = 'import os\n'
    var_3 = [var_2]
    var_4 = '--show-files'
    var_5 = '-'
    var_6 = [var_4, var_5]
    var_7 = bool(var_0)
    assert var_7 is True

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
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'
    var_3 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--force-single-line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool('remapped_deprecated_args' in var_2 or var_2 is not None)
    assert var_3 is True

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
    var_0 = '--multi-line-output'
    var_1 = 'VERTICAL'
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
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--dont-float-to-top'
    var_3 = '--float-to-top'
    var_4 = [var_2, var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = bool(var_0)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_print_hard_fail_with_default_message. Retrieved 4/7 statements.
# Partially parsed test_print_hard_fail_with_custom_message. Retrieved 5/8 statements.
# Partially parsed test_print_hard_fail_without_offending_file. Retrieved 3/6 statements.
# Partially parsed test_print_hard_fail_with_format_error. Retrieved 5/8 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = module_1._print_hard_fail(var_3, var_4)
    var_6 = 'Unrecoverable exception thrown when parsing test.py!'
    var_7 = 'This should NEVER happen.'
    var_8 = 'https://github.com/PyCQA/isort/issues/new'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'Custom error message'
    var_5 = 'file.py'
    var_6 = module_1._print_hard_fail(var_3, var_5, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1._print_hard_fail(var_3)
    var_5 = 'Unrecoverable exception thrown when parsing'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error} - {message}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'Test error'
    var_7 = module_1._print_hard_fail(var_5, message=var_6)
    var_8 = 'Test error'



