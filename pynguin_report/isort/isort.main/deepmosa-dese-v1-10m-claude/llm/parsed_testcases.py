####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/17 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 4/14 statements.
# Partially parsed test_identify_imports_main_unique_package. Retrieved 5/16 statements.
# Partially parsed test_identify_imports_main_unique_module. Retrieved 5/16 statements.
# Partially parsed test_identify_imports_main_unique_attribute. Retrieved 5/16 statements.
# Partially parsed test_identify_imports_main_top_only. Retrieved 5/15 statements.
# Partially parsed test_identify_imports_main_follow_links. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'isort.api.find_imports_in_stream'
    var_2 = '-'
    var_3 = [var_2]

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--packages'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--modules'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--attributes'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--top-only'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.api.find_imports_in_paths'
    var_1 = 'test.py'
    var_2 = '--follow-links'
    var_3 = [var_1, var_2]
    var_4 = module_0.identify_imports_main(var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/12 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/12 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, write_to_stdout=var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_print_hard_fail. Retrieved 11/28 statements.


import _io as module_0
import isort.settings as module_1
import isort.main as module_2

def test_case_0():
    var_0 = module_0.StringIO()
    var_1 = False
    var_2 = 'ERROR: {message}'
    var_3 = 'SUCCESS: {message}'
    var_4 = module_1.Config()
    var_5 = module_2._print_hard_fail(var_4)
    var_6 = 'test.py'
    var_7 = 'Custom error message'
    var_8 = module_2._print_hard_fail(var_4, var_6, var_7)
    var_9 = 'broken.py'
    var_10 = module_2._print_hard_fail(var_4, var_9)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_in_check_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_numeric. Retrieved 7/9 statements.
# Partially parsed test_parse_args_multiple_arguments. Retrieved 6/8 statements.
# Partially parsed test_parse_args_preserves_truthy_values. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 2/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 > var_4

import isort.main as module_0

def test_case_0():
    var_0 = 'order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

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
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 > var_4

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/6 statements.
# Partially parsed test_parse_args_with_deprecated_single_dash_args. Retrieved 4/5 statements.
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
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 > var_4

import isort.main as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 4/12 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, write_to_stdout=var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false. Retrieved 5/11 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false_supported_encoding. Retrieved 5/11 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

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

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 16/26 statements.


def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'line_length'
    var_2 = {var_0, var_1}
    var_3 = 'MockParser'
    var_4 = ()
    var_5 = 'parse_args'
    var_6 = 'Args'
    var_7 = ()
    var_8 = True
    var_9 = None
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = 'force_single_line'
    var_12 = 'file.py'
    var_13 = [var_11, var_12]
    var_14 = 'force_single_line'
    var_15 = var_14 in var_2
    assert var_15 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parse_args_float_to_top_predicate_true. Retrieved 11/20 statements.


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
    var_10 = "Can't set both --float-to-top and --dont-float-to-top."



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_parse_args_no_arguments. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_value_argument. Retrieved 5/6 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_numeric. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_string. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 3/5 statements.
# Partially parsed test_parse_args_none_argv_uses_sys_argv. Retrieved 4/9 statements.
# Partially parsed test_parse_args_dont_float_to_top_sets_false. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--check'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--src'
    var_1 = 'mydir'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'src'

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
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--verbose'
    var_2 = None
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'force_alphabetical_sort'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = var_3[var_4]
    var_6 = len(var_5)
    assert var_6 == 2



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 5/12 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/12 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 4/11 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 4/11 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, ask_to_apply=var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, write_to_stdout=var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 3/9 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top_alone. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multiple_arguments. Retrieved 7/9 statements.
# Partially parsed test_parse_args_none_argv_uses_sys_argv. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = 'remapped_deprecated_args'

import isort.main as module_0

def test_case_0():
    var_0 = 'myfile.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 > var_4

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = '--dont-follow-links'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'order_by_type'
    var_6 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_sort_imports_isort_error_handling. Retrieved 9/26 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'isort.api.check_file'
    var_2 = 'isort.main._print_hard_fail'
    var_3 = False
    var_4 = None
    assert var_4 == 1
    var_5 = 'sys.exit'
    var_6 = 'test_file.py'
    var_7 = True
    var_8 = module_1.sort_imports(var_6, var_0, var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 7/10 statements.
# Partially parsed test_parse_args_with_multi_line_output_name. Retrieved 6/9 statements.
# Partially parsed test_parse_args_filters_empty_values. Retrieved 4/8 statements.
# Partially parsed test_parse_args_with_multiple_arguments. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 > var_4

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--dont-order-by-type'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'order_by_type'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 6/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'force_single_line'
    var_2 = True
    var_3 = [var_0]
    var_4 = module_0.parse_args(var_3)
    var_5 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_preconvert_set. Retrieved 8/22 statements.
# Partially parsed test_preconvert_frozenset. Retrieved 9/23 statements.
# Partially parsed test_preconvert_enum. Retrieved 2/17 statements.
# Partially parsed test_preconvert_callable. Retrieved 2/18 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = module_0._preconvert(var_5)
    var_7 = set(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = 5
    var_4 = 6
    var_5 = [var_2, var_3, var_4]
    var_6 = frozenset(var_5)
    var_7 = module_0._preconvert(var_6)
    var_8 = set(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '/tmp/test.txt'
    var_3 = module_0.Path(var_2)
    var_4 = module_1._preconvert(var_3)
    assert var_4 == '/tmp/test.txt'

def test_case_0():
    var_0 = 1
    var_1 = 2

import builtins as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.object()
    var_3 = module_1._preconvert(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_identify_imports_main_predicate_line_76. Retrieved 5/17 statements.


import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.StringIO()
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = module_1.identify_imports_main(var_2)
    var_4 = len(var_1)



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = 'script.py'
    var_1 = '--some-arg'
    var_2 = 'value'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = None
    var_6 = var_4 is var_5
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_parse_args_argv_none_uses_sys_argv. Retrieved 5/11 statements.
# Partially parsed test_parse_args_argv_provided_uses_argument. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--line-length'
    var_2 = '88'
    var_3 = None
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'line_length'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_with_multiple_arguments. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 >= var_4

import isort.main as module_0

def test_case_0():
    var_0 = 'check'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

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
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false. Retrieved 5/12 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test encoding error'
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'wrap'
    var_1 = 'nowrap'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_args_no_arguments. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_filters_empty_values. Retrieved 2/4 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import isort.main as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--src'
    var_1 = 'path/to/src'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 7/23 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 6/19 statements.
# Partially parsed test_identify_imports_main_with_unique_packages. Retrieved 7/21 statements.
# Partially parsed test_identify_imports_main_with_unique_modules. Retrieved 7/21 statements.
# Partially parsed test_identify_imports_main_with_unique_attributes. Retrieved 9/23 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 6/18 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'import os\nfrom sys import path\n'
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = 'isort.api.find_imports_in_stream'
    var_5 = '-'
    var_6 = [var_5]

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'isort.api.find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = 'sys.argv'
    var_2 = 'isort.api.find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = '--packages'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = 'sys.argv'
    var_2 = 'isort.api.find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = '--modules'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = 'argv'
    var_4 = 'isort.api.find_imports_in_paths'
    var_5 = 'test.py'
    var_6 = '--attributes'
    var_7 = [var_5, var_6]
    var_8 = module_0.identify_imports_main(var_7)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'isort.api.find_imports_in_paths'
    var_2 = 'test.py'
    var_3 = '--top-only'
    var_4 = [var_2, var_3]
    var_5 = module_0.identify_imports_main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'isort.api.find_imports_in_paths'
    var_2 = 'test.py'
    var_3 = '--follow-links'
    var_4 = [var_2, var_3]
    var_5 = module_0.identify_imports_main(var_4)



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_preconvert_wrap_modes_predicate.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_preconvert_path_object. Retrieved 2/4 statements.


import zipfile as module_0

def test_case_0():
    var_0 = '/home/user/file.txt'
    var_1 = module_0.Path(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_sort_imports_exception_handler_line_40. Retrieved 3/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.Config()
    var_2 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_callable_with_name_attribute. Retrieved 1/15 statements.


def test_case_0():
    var_0 = '__name__'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_callable_with_name_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '__name__'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_identify_imports_main_predicate_line_76. Retrieved 5/17 statements.


import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.StringIO()
    var_2 = 'test.py'
    var_3 = [var_2]
    assert var_3 == 'import os'
    var_4 = module_1.identify_imports_main(var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_isort_error_handling. Retrieved 5/15 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = 'Test error'
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_3)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_preconvert_path_evaluates_to_true. Retrieved 2/4 statements.


import zipfile as module_0

def test_case_0():
    var_0 = '/home/user/file.txt'
    var_1 = module_0.Path(var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_error. Retrieved 8/17 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_settings_path_file. Retrieved 5/15 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 3/11 statements.
# Partially parsed test_main_stream_input_check. Retrieved 3/8 statements.
# Partially parsed test_main_recursive_root_error. Retrieved 5/13 statements.
# Partially parsed test_main_stream_filename_with_files_error. Retrieved 7/15 statements.
# Partially parsed test_main_show_files_with_stream_error. Retrieved 6/14 statements.
# Partially parsed test_main_check_stream. Retrieved 4/8 statements.
# Partially parsed test_main_wrong_sorted_files_exit. Retrieved 7/18 statements.
# Partially parsed test_main_parse_args_none. Retrieved 3/9 statements.
# Partially parsed test_main_parse_args_with_argv. Retrieved 3/5 statements.
# Partially parsed test_main_parse_args_deprecated_args. Retrieved 3/5 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.


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

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '.isort.cfg'
    var_3 = '[settings]\nprofile=black'
    var_4 = '--settings-path'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os'
    var_2 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = '-'
    var_2 = [var_1]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'test.py'
    var_4 = 'file.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = 'sys.exit'
    var_2 = '--show-files'
    var_3 = '-'
    var_4 = [var_2, var_3]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = '--check-only'
    var_2 = '-'
    var_3 = [var_1, var_2]

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os'
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '--check-only'
    var_5 = [var_4, var_1]
    var_6 = module_0.main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = None
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'sp'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding_returns_sort_attempt_with_false_supported_encoding. Retrieved 5/11 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = False
    var_4 = module_1.sort_imports(var_2, var_1, var_3)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/15 statements.
# Partially parsed test_main_no_files_no_config. Retrieved 3/9 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 6/12 statements.
# Partially parsed test_main_settings_path_file. Retrieved 6/12 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 2/8 statements.
# Partially parsed test_main_parse_args_deprecated_single_dash. Retrieved 4/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 4/7 statements.
# Partially parsed test_main_parse_args_none_argv. Retrieved 5/9 statements.


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

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    var_5 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'override.py'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--show-files'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\n'
    var_2 = '--settings-path'
    var_3 = 'test.py'
    var_4 = [var_2, var_1, var_3]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = '--settings-path'
    var_1 = 'test.py'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'argv'
    var_1 = 'isort'
    var_2 = [var_1]
    var_3 = None
    var_4 = module_0.parse_args(var_3)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/13 statements.
# Partially parsed test_main_show_config_and_show_files_error. Retrieved 8/17 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/6 statements.
# Partially parsed test_main_with_arguments_but_no_paths. Retrieved 6/15 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 8/19 statements.
# Partially parsed test_main_virtual_env_does_not_exist. Retrieved 5/8 statements.
# Partially parsed test_main_show_config_flag. Retrieved 3/10 statements.
# Partially parsed test_main_stream_input_with_check. Retrieved 4/8 statements.
# Partially parsed test_main_dangerous_root_operation. Retrieved 6/15 statements.
# Partially parsed test_main_stream_filename_override_error. Retrieved 8/17 statements.
# Partially parsed test_main_show_files_with_stream_error. Retrieved 7/16 statements.


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

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '--check'
    var_4 = [var_3]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\n'
    var_2 = False
    var_3 = 'sys.exit'
    var_4 = '--settings-path'
    var_5 = '--show-config'
    var_6 = [var_4, var_1, var_5]
    var_7 = module_0.main(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-config'

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = '--check'
    var_3 = [var_1, var_2]

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    assert var_1 == 1
    var_2 = 'sys.exit'
    var_3 = '/'
    var_4 = [var_3]
    var_5 = module_0.main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    assert var_1 == 1
    var_2 = 'sys.exit'
    var_3 = 'test.py'
    var_4 = '--filename'
    var_5 = 'other.py'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.main(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'sys.exit'
    var_3 = '-'
    var_4 = '--show-files'
    var_5 = [var_3, var_4]
    var_6 = module_0.main(var_5)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_empty_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_single_deprecated_arg. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 7/10 statements.
# Partially parsed test_parse_args_with_multi_line_output_name. Retrieved 6/9 statements.
# Partially parsed test_parse_args_filters_empty_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 2/3 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_main_show_version. Retrieved 10/16 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 8/15 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 6/11 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 6/18 statements.
# Partially parsed test_main_settings_path_from_file_names. Retrieved 6/15 statements.
# Partially parsed test_main_stream_input_with_check. Retrieved 4/8 statements.
# Partially parsed test_main_stream_input_show_files_error. Retrieved 7/16 statements.
# Partially parsed test_main_dangerous_root_operation. Retrieved 6/12 statements.
# Partially parsed test_main_filename_override_without_stream. Retrieved 8/14 statements.
# Partially parsed test_main_show_files_option. Retrieved 4/17 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = [var_2]
    var_5 = module_0.main(var_4)
    var_6 = 'ASCII_ART'
    var_7 = dir()
    var_8 = var_6 in var_7
    var_9 = 0

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

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = 'QUICK_GUIDE'
    var_3 = dir()
    var_4 = var_2 in var_3
    var_5 = 0

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = []
    var_3 = 'sys.exit'
    var_4 = '--settings-path'
    var_5 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = []
    var_3 = 'sys.exit'
    var_4 = [var_0]
    var_5 = module_0.main(var_4)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = '--check'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = 'import os\n'
    var_3 = '-'
    var_4 = '--show-files'
    var_5 = [var_3, var_4]
    var_6 = len(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '/'
    var_3 = [var_2]
    var_4 = module_0.main(var_3)
    var_5 = len(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = 'sys.exit'
    var_2 = '--filename'
    var_3 = 'override.py'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.main(var_5)
    var_7 = len(var_0)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'
    var_3 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 6/21 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 6/19 statements.
# Partially parsed test_identify_imports_main_with_unique_packages. Retrieved 8/21 statements.
# Partially parsed test_identify_imports_main_with_unique_modules. Retrieved 8/21 statements.
# Partially parsed test_identify_imports_main_with_unique_attributes. Retrieved 10/21 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 7/18 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = [var_1]
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'isort.api.find_imports_in_stream'

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'isort.api.find_imports_in_paths'
    var_5 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = 'os.path'
    var_4 = 'sys.argv'
    var_5 = 'isort.api.find_imports_in_stream'
    var_6 = None
    var_7 = module_0.identify_imports_main(var_2, var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = 'os.path'
    var_4 = 'sys'
    var_5 = 'isort.api.find_imports_in_stream'
    var_6 = None
    var_7 = module_0.identify_imports_main(var_2, var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'path'
    var_5 = 'sys'
    var_6 = 'argv'
    var_7 = 'isort.api.find_imports_in_stream'
    var_8 = None
    var_9 = module_0.identify_imports_main(var_2, var_8)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'isort.api.find_imports_in_stream'
    var_5 = None
    var_6 = module_0.identify_imports_main(var_2, var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = 'os'
    var_4 = 'isort.api.find_imports_in_paths'
    var_5 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test_file.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parse_args_float_to_top_predicate. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = 'other_arg'
    var_3 = True
    var_4 = False
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {k: v for (k, v) in var_0 if v}
    var_8 = 'dont_float_to_top'
    var_9 = var_7[var_8]
    var_10 = 'float_to_top'
    var_11 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_remapped_deprecated_args_added_to_arguments. Retrieved 5/13 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = 'deprecated_arg'
    var_3 = [var_2]
    var_4 = module_0.parse_args(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'Args'
    var_2 = ()
    var_3 = {}
    var_4 = 'somefile.py'
    var_5 = [var_0, var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 2/13 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 2/13 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_value_error. Retrieved 2/12 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 2/14 statements.
# Partially parsed test_sort_imports_unsupported_encoding_not_verbose. Retrieved 2/13 statements.
# Failed to parse test_sort_imports_isort_error.


def test_case_0():
    var_0 = 'test.py'
    var_1 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False

def test_case_0():
    var_0 = 'test.py'
    var_1 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = False



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 'virtual_env'
    var_1 = '/path/to/venv'
    var_2 = {var_0: var_1}
    var_3 = var_0 in var_2
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_verbose. Retrieved 5/12 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, write_to_stdout=var_3)



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = 'settings_path'
    var_1 = '/some/path'
    var_2 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/14 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 3/7 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 5/15 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 5/8 statements.
# Partially parsed test_main_stdin_check_mode. Retrieved 4/8 statements.
# Partially parsed test_main_show_files_with_stdin_error. Retrieved 4/9 statements.
# Partially parsed test_main_parse_args_dont_order_by_type. Retrieved 5/7 statements.
# Partially parsed test_main_parse_args_dont_follow_links. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_with_check_flag. Retrieved 7/16 statements.
# Partially parsed test_sort_imports_without_check_flag. Retrieved 5/12 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'argv'
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
    var_2 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=80\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = '--check'
    var_3 = [var_1, var_2]

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'test.py'
    var_2 = 'somefile.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = '-'
    var_1 = '--show-files'
    var_2 = [var_0, var_1]
    var_3 = 'import os\n'

import isort.main as module_0

def test_case_0():
    var_0 = '--force-single-line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'vertical'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.Config()
    var_3 = True
    var_4 = 'incorrectly_sorted'
    var_5 = 'skipped'
    var_6 = 'supported_encoding'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.Config()
    var_3 = False
    var_4 = 'incorrectly_sorted'

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/nonexistent/file.py'
    var_2 = module_1.sort_imports(var_1, var_0)
    assert var_2 is None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_basic. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_none. Retrieved 2/3 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_empty_values_filtered. Retrieved 4/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = False
    var_3 = True

import isort.main as module_0

def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'force_alphabetical_sort'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = var_3[var_4]
    var_6 = len(var_5)
    assert var_6 == 2



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_args_multi_line_output_predicate_true. Retrieved 10/22 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = 'other_arg'
    var_2 = '3'
    var_3 = None
    var_4 = '--multi-line-output'
    var_5 = '3'
    var_6 = [var_4, var_5]
    var_7 = module_0.parse_args(var_6)
    var_8 = 'multi_line_output'
    var_9 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_print_hard_fail_with_custom_message. Retrieved 3/6 statements.
# Partially parsed test_print_hard_fail_with_offending_file. Retrieved 3/6 statements.
# Partially parsed test_print_hard_fail_default_message. Retrieved 2/5 statements.
# Partially parsed test_print_hard_fail_with_format_error. Retrieved 4/7 statements.
# Partially parsed test_print_hard_fail_with_color_output_disabled. Retrieved 4/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'Custom error message'
    var_2 = module_1._print_hard_fail(var_0, message=var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = module_1._print_hard_fail(var_0, var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = module_1._print_hard_fail(var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '{error}: {message}'
    var_1 = module_0.Config()
    var_2 = 'Test error'
    var_3 = module_1._print_hard_fail(var_1, message=var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'Test message without color'
    var_3 = module_1._print_hard_fail(var_1, message=var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_args_multi_line_output_predicate. Retrieved 6/18 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = '3'
    var_2 = '--multi-line-output'
    var_3 = [var_2, var_1]
    var_4 = module_0.parse_args(var_3)
    var_5 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_remapped_deprecated_args_added_to_arguments. Retrieved 6/14 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'arg_name'
    var_4 = [var_3]
    var_5 = module_0.parse_args(var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/13 statements.
# Partially parsed test_sort_imports_unsupported_encoding_not_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, write_to_stdout=var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'force_single_line'
    var_1 = 'line_length'
    var_2 = [var_0, var_1]
    var_3 = 'force_single_line'
    var_4 = 'some_value'
    var_5 = [var_3, var_4]
    var_6 = 0
    var_7 = var_5[var_6]
    var_8 = var_7 in var_2
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_dont_float_to_top. Retrieved 4/5 statements.


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
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

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
    var_0 = '--multi-line-mode'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 'value'
    var_7 = hasattr(var_5, var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-mode'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 'value'
    var_7 = hasattr(var_5, var_6)

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '120'
    var_2 = '--profile'
    var_3 = 'django'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 >= var_5

import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'order_by_type'
    var_5 = module_1.str(var_3)
    var_6 = var_4 in var_5



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_caught. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 6/23 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 6/21 statements.
# Partially parsed test_identify_imports_main_unique_package. Retrieved 7/22 statements.
# Partially parsed test_identify_imports_main_unique_module. Retrieved 7/22 statements.
# Partially parsed test_identify_imports_main_unique_attribute. Retrieved 9/24 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 6/19 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = [var_1]
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'find_imports_in_stream'

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = [var_3]
    var_5 = module_0.identify_imports_main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = 'sys.version'
    var_2 = 'find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = '--packages'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = 'sys.version'
    var_2 = 'find_imports_in_paths'
    var_3 = 'test.py'
    var_4 = '--modules'
    var_5 = [var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = 'version'
    var_4 = 'find_imports_in_paths'
    var_5 = 'test.py'
    var_6 = '--attributes'
    var_7 = [var_5, var_6]
    var_8 = module_0.identify_imports_main(var_7)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'find_imports_in_paths'
    var_2 = 'test.py'
    var_3 = '--top-only'
    var_4 = [var_2, var_3]
    var_5 = module_0.identify_imports_main(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'find_imports_in_paths'
    var_2 = 'test.py'
    var_3 = '--follow-links'
    var_4 = [var_2, var_3]
    var_5 = module_0.identify_imports_main(var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_float_to_top_predicate_evaluates_to_true. Retrieved 14/23 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = 'dont_order_by_type'
    var_3 = 'dont_follow_links'
    var_4 = 'multi_line_output'
    var_5 = True
    var_6 = False
    var_7 = None
    var_8 = {var_0: var_5, var_1: var_5, var_2: var_6, var_3: var_6, var_4: var_7}
    var_9 = '--dont-float-to-top'
    var_10 = '--float-to-top'
    var_11 = [var_9, var_10]
    var_12 = module_0.parse_args(var_11)
    var_13 = "Can't set both --float-to-top and --dont-float-to-top."



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sort_imports_file_skipped_exception_at_line_27. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_main_show_version. Retrieved 8/13 statements.
# Partially parsed test_main_show_config_and_show_files_conflict. Retrieved 8/10 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 7/12 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 26/36 statements.
# Partially parsed test_main_virtual_env_does_not_exist. Retrieved 13/18 statements.
# Partially parsed test_main_stream_input_check_mode. Retrieved 30/39 statements.
# Partially parsed test_main_stream_input_show_files_error. Retrieved 16/18 statements.
# Partially parsed test_main_root_directory_without_allow_root. Retrieved 24/30 statements.
# Partially parsed test_main_stream_filename_override_error. Retrieved 26/32 statements.
# Partially parsed test_main_show_config. Retrieved 24/35 statements.
# Partially parsed test_sort_imports_with_check_mode. Retrieved 7/11 statements.
# Partially parsed test_sort_imports_file_skipped. Retrieved 8/13 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 7/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = lambda argv: var_3
    var_5 = []
    var_6 = module_0.main(var_5)
    var_7 = 0

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_config'
    var_2 = 'show_files'
    var_3 = True
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = lambda argv: var_4
    var_6 = []
    var_7 = module_0.main(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = lambda argv: var_3
    var_5 = []
    var_6 = module_0.main(var_5)

import isort.main as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[tool:isort]\n'
    var_2 = 'isort.main.parse_args'
    var_3 = 'show_version'
    var_4 = 'show_config'
    var_5 = 'show_files'
    var_6 = 'settings_path'
    var_7 = 'files'
    var_8 = False
    var_9 = []
    var_10 = 'isort.main.Config'
    var_11 = 'Config'
    var_12 = ()
    var_13 = 'quiet'
    var_14 = 'color_output'
    var_15 = 'format_error'
    var_16 = 'format_success'
    var_17 = 'verbose'
    var_18 = 'filter_files'
    var_19 = '__dict__'
    var_20 = True
    var_21 = ''
    var_22 = {}
    var_23 = {var_13: var_20, var_14: var_8, var_15: var_21, var_16: var_21, var_17: var_8, var_18: var_8, var_19: var_22}
    var_24 = []
    var_25 = module_0.main(var_24)

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'virtual_env'
    var_5 = 'files'
    var_6 = False
    var_7 = '/nonexistent/path'
    var_8 = []
    var_9 = {var_1: var_6, var_2: var_6, var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = lambda argv: var_9
    var_11 = []
    var_12 = module_0.main(var_11)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'quiet'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = True
    var_8 = False
    var_9 = ''
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_9, var_6: var_8}
    var_11 = 'isort.main.parse_args'
    var_12 = 'show_version'
    var_13 = 'show_config'
    var_14 = 'show_files'
    var_15 = 'files'
    var_16 = 'check'
    var_17 = 'show_diff'
    var_18 = 'filename'
    var_19 = 'ext_format'
    var_20 = '-'
    var_21 = [var_20]
    var_22 = None
    var_23 = {var_12: var_8, var_13: var_8, var_14: var_8, var_15: var_21, var_16: var_7, var_17: var_8, var_18: var_22, var_19: var_22}
    var_24 = lambda argv: var_23
    var_25 = 'isort.main.Config'
    var_26 = 'isort.main.api.check_stream'
    var_27 = lambda **kwargs: var_7
    var_28 = 'import os\nimport sys\n'
    var_29 = []

import isort.main as module_0

def test_case_0():
    var_0 = 'isort.main.parse_args'
    var_1 = 'show_version'
    var_2 = 'show_config'
    var_3 = 'show_files'
    var_4 = 'files'
    var_5 = 'check'
    var_6 = 'filename'
    var_7 = False
    var_8 = True
    var_9 = '-'
    var_10 = [var_9]
    var_11 = None
    var_12 = {var_1: var_7, var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_7, var_6: var_11}
    var_13 = lambda argv: var_12
    var_14 = []
    var_15 = module_0.main(var_14)

import isort.main as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'quiet'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = True
    var_8 = False
    var_9 = ''
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_9, var_6: var_8}
    var_11 = 'isort.main.parse_args'
    var_12 = 'show_version'
    var_13 = 'show_config'
    var_14 = 'show_files'
    var_15 = 'files'
    var_16 = 'allow_root'
    var_17 = '/'
    var_18 = [var_17]
    var_19 = {var_12: var_8, var_13: var_8, var_14: var_8, var_15: var_18, var_16: var_8}
    var_20 = lambda argv: var_19
    var_21 = 'isort.main.Config'
    var_22 = []
    var_23 = module_0.main(var_22)

import isort.main as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'quiet'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = True
    var_8 = False
    var_9 = ''
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_9, var_6: var_8}
    var_11 = 'isort.main.parse_args'
    var_12 = 'show_version'
    var_13 = 'show_config'
    var_14 = 'show_files'
    var_15 = 'files'
    var_16 = 'filename'
    var_17 = 'allow_root'
    var_18 = 'somefile.py'
    var_19 = [var_18]
    var_20 = 'override.py'
    var_21 = {var_12: var_8, var_13: var_8, var_14: var_8, var_15: var_19, var_16: var_20, var_17: var_8}
    var_22 = lambda argv: var_21
    var_23 = 'isort.main.Config'
    var_24 = []
    var_25 = module_0.main(var_24)

import isort.main as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'quiet'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = '__dict__'
    var_8 = True
    var_9 = False
    var_10 = ''
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_10, var_6: var_9, var_7: var_13}
    var_15 = 'isort.main.parse_args'
    var_16 = 'show_version'
    var_17 = 'show_config'
    var_18 = 'show_files'
    var_19 = 'files'
    var_20 = 'settings_path'
    var_21 = 'isort.main.Config'
    var_22 = []
    var_23 = module_0.main(var_22)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = {}
    var_3 = 'isort.main.api.check_file'
    var_4 = True
    var_5 = lambda **kwargs: var_4
    var_6 = 'test.py'

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = {}
    var_3 = 'isort.main.api.check_file'
    var_4 = 'test'
    var_5 = module_0.FileSkipped(var_4)
    var_6 = 'test.py'
    var_7 = True

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'verbose'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'isort.main.api.check_file'
    var_6 = 'File not found'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_preconvert_set. Retrieved 8/22 statements.
# Partially parsed test_preconvert_frozenset. Retrieved 9/23 statements.
# Partially parsed test_preconvert_enum. Retrieved 2/17 statements.
# Partially parsed test_preconvert_function. Retrieved 2/18 statements.
# Partially parsed test_preconvert_builtin_function. Retrieved 2/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = module_0._preconvert(var_5)
    var_7 = set(var_6)

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = 5
    var_4 = 6
    var_5 = [var_2, var_3, var_4]
    var_6 = frozenset(var_5)
    var_7 = module_0._preconvert(var_6)
    var_8 = set(var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '/tmp/test'
    var_3 = module_0.Path(var_2)
    var_4 = module_1._preconvert(var_3)
    assert var_4 == '/tmp/test'

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

import builtins as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.object()
    var_3 = module_1._preconvert(var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_parse_args_no_arguments. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 7/10 statements.
# Partially parsed test_parse_args_deprecated_single_dash_args. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_multi_line_output_zero. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--file-input'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--file-input'
    var_1 = 'test.py'
    var_2 = '--dont-order-by-type'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_main_show_version. Retrieved 5/11 statements.
# Partially parsed test_main_no_files_no_show_config. Retrieved 2/5 statements.
# Partially parsed test_main_virtual_env_does_not_exist. Retrieved 6/10 statements.
# Partially parsed test_main_settings_path_is_file. Retrieved 5/13 statements.
# Partially parsed test_parse_args_basic. Retrieved 7/9 statements.
# Partially parsed test_parse_args_dont_order_by_type. Retrieved 5/6 statements.
# Partially parsed test_parse_args_dont_follow_links. Retrieved 5/6 statements.
# Partially parsed test_parse_args_dont_float_to_top_alone. Retrieved 5/6 statements.
# Partially parsed test_parse_args_deprecated_single_dash. Retrieved 6/8 statements.
# Partially parsed test_parse_args_show_version. Retrieved 4/5 statements.
# Partially parsed test_parse_args_none_argv. Retrieved 7/9 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 5/13 statements.
# Partially parsed test_sort_imports_write_to_stdout. Retrieved 5/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = 'isort'
    var_4 = 0

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
    var_0 = '--line-length'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/venv'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    var_5 = 0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[isort]\nline_length=88\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = '--settings-path'

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = '--allow-root'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--filename'
    var_1 = 'override.py'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '100'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'line_length'
    var_6 = 'files'

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '3'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'vertical'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)

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
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = 'test.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'float_to_top'

import isort.main as module_0

def test_case_0():
    var_0 = '--line-length'
    var_1 = '80'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'remapped_deprecated_args'

import isort.main as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'show_version'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.argv'
    var_1 = 'isort'
    var_2 = '--version'
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = module_0.parse_args(var_4)
    var_6 = 'show_version'

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = 88
    var_3 = module_0.Config()
    var_4 = True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 88
    var_1 = module_0.Config()
    var_2 = '/nonexistent/file.py'
    var_3 = module_1.sort_imports(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = 88
    var_3 = module_0.Config()
    var_4 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 5/11 statements.
# Partially parsed test_parse_args_with_empty_list. Retrieved 3/4 statements.
# Partially parsed test_parse_args_filters_falsy_values. Retrieved 2/5 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 7/9 statements.
# Partially parsed test_parse_args_multi_line_output_name. Retrieved 6/7 statements.
# Partially parsed test_parse_args_returns_dict. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--profile'
    var_2 = 'black'
    var_3 = None
    var_4 = module_0.parse_args(var_3)

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

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = '--line-length'
    var_3 = '88'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.parse_args(var_4)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '0'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]
    var_6 = 0

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'GRID'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--profile'
    var_1 = 'black'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_main_function_signature_returns_none. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'argv'
    var_1 = 'stdin'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_multi_line_output_predicate_evaluates_to_true. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'multi_line_output'
    var_3 = '1'
    var_4 = '1'
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = 'virtual_env'
    var_1 = 'show_version'
    var_2 = 'settings_path'
    var_3 = '/path/to/venv'
    var_4 = False
    var_5 = '/some/path'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = var_0 in var_6
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_sort_imports_check_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_sort_mode_correctly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_incorrectly_sorted. Retrieved 4/9 statements.
# Partially parsed test_sort_imports_sort_mode_file_skipped. Retrieved 4/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_unsupported_encoding_not_verbose. Retrieved 5/11 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 4/13 statements.
# Partially parsed test_sort_imports_with_ask_to_apply. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_with_write_to_stdout. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)
    assert var_3 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'test.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_1.sort_imports(var_1, var_0, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = module_1.sort_imports(var_1, var_0, var_2, write_to_stdout=var_3)



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_main_show_version. Retrieved 7/12 statements.
# Partially parsed test_main_no_files_no_config_error. Retrieved 5/8 statements.
# Partially parsed test_main_settings_path_file. Retrieved 6/14 statements.
# Partially parsed test_main_settings_path_directory. Retrieved 6/14 statements.
# Partially parsed test_main_virtual_env_nonexistent. Retrieved 5/11 statements.
# Partially parsed test_main_stdin_check. Retrieved 4/7 statements.
# Partially parsed test_main_stdin_sort. Retrieved 3/6 statements.
# Partially parsed test_main_show_files. Retrieved 3/10 statements.
# Partially parsed test_main_check_mode. Retrieved 5/10 statements.
# Partially parsed test_main_verbose_mode. Retrieved 4/13 statements.
# Partially parsed test_main_show_config. Retrieved 3/10 statements.
# Partially parsed test_main_deprecated_args. Retrieved 5/8 statements.
# Partially parsed test_main_multi_line_output_digit. Retrieved 5/11 statements.
# Partially parsed test_main_multi_line_output_name. Retrieved 5/11 statements.


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
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)
    var_2 = len(var_0)
    var_3 = 0
    var_4 = var_2 > var_3

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=80\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = '--settings-path'
    var_5 = '--show-files'

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=80\n'
    var_2 = 'test.py'
    var_3 = 'import os\n'
    var_4 = '--settings-path'
    var_5 = '--show-files'

def test_case_0():
    var_0 = 'nonexistent_venv'
    var_1 = 'test.py'
    var_2 = 'import os\n'
    var_3 = '--virtual-env'
    var_4 = '--show-files'

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '-'
    var_2 = '--check'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = '-'
    var_2 = [var_1]

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--filename'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-files'

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = '--check'
    var_3 = [var_0, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--verbose'
    var_3 = '--show-files'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--show-config'

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = '--show-config'
    var_2 = 'test.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--multi-line-output'
    var_3 = '3'
    var_4 = '--show-files'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\n'
    var_2 = '--multi-line-output'
    var_3 = 'VERTICAL'
    var_4 = '--show-files'



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = True
    var_2 = var_0 and var_1
    assert var_2 is True



