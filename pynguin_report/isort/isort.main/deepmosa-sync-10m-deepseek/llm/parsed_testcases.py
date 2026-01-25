####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_sort_imports_returns_sortattempt_on_unsupported_encoding. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_returns_sortattempt_with_incorrectly_sorted_on_check. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_returns_sortattempt_with_skipped_on_check. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_returns_sortattempt_with_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_returns_sortattempt_with_skipped. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'nonexistent_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'invalid_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = True
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'unsupported_encoding.py'
    var_10 = {}
    var_11 = module_1.sort_imports(var_9, var_8, **var_10)
    var_12 = var_11.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'incorrectly_sorted.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.incorrectly_sorted
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'skipped_file.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.skipped
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'incorrectly_sorted.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'skipped_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = var_9.skipped
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_returns_attempt_with_unsupported_encoding. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.txt'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = var_6.supported_encoding
    assert var_7 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parse_args_with_no_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_empty_argv. Retrieved 3/4 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/6 statements.


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
    var_0 = '--dont_order_by_type'
    var_1 = '--dont_follow_links'
    var_2 = '--dont_float_to_top'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = var_4['order_by_type']
    assert var_5 is False
    var_6 = var_4['follow_links']
    assert var_6 is False
    var_7 = var_4['float_to_top']
    assert var_7 is False

import isort.main as module_0

def test_case_0():
    var_0 = 'single_dash_arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['single_dash_arg'])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 3
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'HANGING_INDENT'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_true_and_file_is_incorrectly_sorted. Retrieved 8/16 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_false_and_file_is_incorrectly_sorted. Retrieved 7/15 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_true_and_file_is_skipped. Retrieved 6/19 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_false_and_file_is_skipped. Retrieved 6/19 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test.py'
    var_2 = 'MockAPI'
    var_3 = ()
    var_4 = 'check_file'
    var_5 = False
    var_6 = lambda *args, **kwargs: var_5
    var_7 = {var_4: var_6}
    var_8 = [var_2, var_3, var_7]
    var_9 = True

def test_case_0():
    var_0 = []
    var_1 = 'test.py'
    var_2 = 'MockAPI'
    var_3 = ()
    var_4 = 'sort_file'
    var_5 = False
    var_6 = lambda *args, **kwargs: var_5
    var_7 = {var_4: var_6}
    var_8 = [var_2, var_3, var_7]

def test_case_0():
    var_0 = []
    var_1 = 'test.py'
    var_2 = 'MockAPI'
    var_3 = ()
    var_4 = 'check_file'
    var_5 = ()
    var_6 = True

def test_case_0():
    var_0 = []
    var_1 = 'test.py'
    var_2 = 'MockAPI'
    var_3 = ()
    var_4 = 'sort_file'
    var_5 = ()
    var_6 = False



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_file.py'
    var_6 = 'Custom error message'
    var_7 = module_1._print_hard_fail(var_4, var_5, var_6)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_file.py'
    var_6 = module_1._print_hard_fail(var_4, var_5)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._print_hard_fail(var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sort_imports_does_not_raise_exception. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'example.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 'old_arg'
    var_1 = {var_0}
    var_2 = [var_0]
    var_3 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 4/8 statements.
# Partially parsed test_parse_args_with_custom_argv. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_with_multi_line_output_string. Retrieved 6/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = module_0.parse_args()

import isort.main as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '-old_arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['old_arg'])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'
    var_4 = bool('order_by_type' in var_2)
    assert var_4 is True
    var_5 = var_2['order_by_type']
    assert var_5 is False

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'
    var_4 = bool('follow_links' in var_2)
    assert var_4 is True
    var_5 = var_2['follow_links']
    assert var_5 is False

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = bool('float_to_top' in var_2)
    assert var_4 is True
    var_5 = var_2['float_to_top']
    assert var_5 is False

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True
    var_6 = 'multi_line_output'
    var_7 = var_3[var_6]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_preconvert_with_set. Retrieved 6/7 statements.
# Partially parsed test_preconvert_with_frozenset. Retrieved 7/8 statements.
# Partially parsed test_preconvert_with_wrapmode. Retrieved 1/4 statements.
# Failed to parse test_preconvert_with_callable.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)
    var_5 = sorted(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)
    var_6 = sorted(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

def test_case_0():
    var_0 = 'SOME_MODE'

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/some/path'

import isort.main as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0._preconvert(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_args_with_float_to_top_and_dont_float_to_top. Retrieved 4/10 statements.
# Partially parsed test_parse_args_with_numeric_multi_line_output. Retrieved 5/6 statements.
# Partially parsed test_parse_args_with_string_multi_line_output. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'some-deprecated-arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['some-deprecated-arg'])
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
    var_0 = []
    var_1 = '--float-to-top'
    var_2 = '--dont-float-to-top'
    var_3 = [var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = "Can't set both --float-to-top and --dont-float-to-top."
    var_6 = bool("Can't set both --float-to-top and --dont-float-to-top." in var_1)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output=3'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = bool('multi_line_output' in var_2)
    assert var_4 is True
    var_5 = 'multi_line_output'
    var_6 = var_2[var_5]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output=VERTICAL_HANGING_INDENT'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = bool('multi_line_output' in var_2)
    assert var_4 is True
    var_5 = 'multi_line_output'
    var_6 = var_2[var_5]



# Parsed testcases at query #11
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = None
    assert var_0 is None
    var_1 = module_0.parse_args(var_0)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
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
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.skipped
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_preconvert_with_callable_with_name.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 3/4 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = '--order_by_type'
    var_2 = module_0.parse_args()
    var_3 = bool(var_2 == {'order_by_type': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = '--follow_links'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'order_by_type': False, 'follow_links': True})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'remapped_deprecated_args': ['x', 'y'], 'x': True, 'y': True})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float_to_top'
    var_1 = '--dont_float_to_top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output'].value
    assert var_4 == 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'HANGING'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output'].name
    assert var_4 == 'HANGING'

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'some_value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = bool('multi_line_output' in var_3)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = {}
    var_10 = module_1.sort_imports(var_8, var_7, var_0, **var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True
    var_12 = bool(not var_10.incorrectly_sorted)
    assert var_12 is True
    var_13 = bool(not var_10.skipped)
    assert var_13 is True
    var_14 = bool(var_10.supported_encoding)
    assert var_14 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = True
    var_10 = {}
    var_11 = module_1.sort_imports(var_8, var_7, var_9, **var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True
    var_13 = bool(var_11.incorrectly_sorted)
    assert var_13 is True
    var_14 = bool(not var_11.skipped)
    assert var_14 is True
    var_15 = bool(var_11.supported_encoding)
    assert var_15 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = True
    var_10 = {}
    var_11 = module_1.sort_imports(var_8, var_7, var_9, **var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True
    var_13 = bool(not var_11.incorrectly_sorted)
    assert var_13 is True
    var_14 = bool(var_11.skipped)
    assert var_14 is True
    var_15 = bool(var_11.supported_encoding)
    assert var_15 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = {}
    var_10 = module_1.sort_imports(var_8, var_7, var_0, **var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True
    var_12 = bool(not var_10.incorrectly_sorted)
    assert var_12 is True
    var_13 = bool(not var_10.skipped)
    assert var_13 is True
    var_14 = bool(not var_10.supported_encoding)
    assert var_14 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = {}
    var_10 = module_1.sort_imports(var_8, var_7, var_0, **var_9)
    assert var_10 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = False
    var_10 = {}
    var_11 = module_1.sort_imports(var_8, var_7, var_9, **var_10)
    var_12 = bool(False)
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'Error: {error}'
    var_2 = 'Success: {success}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'example.py'
    var_9 = False
    var_10 = {}
    var_11 = module_1.sort_imports(var_8, var_7, var_9, **var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_main_with_show_version. Retrieved 3/14 statements.
# Partially parsed test_main_with_empty_args. Retrieved 2/12 statements.
# Partially parsed test_main_with_check_flag. Retrieved 4/14 statements.
# Partially parsed test_main_with_show_config. Retrieved 3/13 statements.
# Partially parsed test_main_with_stdin. Retrieved 4/17 statements.
# Partially parsed test_main_with_invalid_root. Retrieved 3/13 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = '--show-version'
    var_2 = []
    var_3 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = []
    var_2 = module_0.main()
    var_3 = 'Imports'

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = '--check'
    var_2 = 'test_file.py'
    var_3 = []
    var_4 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = '--show-config'
    var_2 = []
    var_3 = module_0.main()
    var_4 = 'settings_path'

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = '-'
    var_2 = 'import os\nimport sys'
    var_3 = [var_2]
    var_4 = []
    var_5 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = 'isort'
    var_1 = '/'
    var_2 = []
    var_3 = module_0.main()
    var_4 = 'dangerous'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_when_file_skipped_during_check. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.skipped
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is True
    var_8 = var_5.incorrectly_sorted
    assert var_8 is False



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None



# Parsed testcases at query #21
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = 'show_version'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.main(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test.py'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_empty_argv. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_non_empty_argv. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--example-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_preconvert_callable_with_name.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_with_skipped_true_when_fileskipped_exception_occurs. Retrieved 6/13 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = True
    var_7 = module_1.SortAttempt(var_3, var_6, var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]



# Parsed testcases at query #27
#--------------------------




import zipfile as module_0
import isort.main as module_1
import locale as module_2

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    var_3 = module_2.str(var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_normal_mode_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = bool(var_10.incorrectly_sorted is True or var_10.incorrectly_sorted is False)
    assert var_11 is True
    var_12 = var_10.skipped
    assert var_12 is False
    var_13 = var_10.supported_encoding
    assert var_13 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = bool(var_10.skipped is True or var_10.skipped is False)
    assert var_11 is True
    var_12 = var_10.supported_encoding
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    var_10 = bool(var_9.incorrectly_sorted is True or var_9.incorrectly_sorted is False)
    assert var_10 is True
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    var_10 = bool(var_9.skipped is True or var_9.skipped is False)
    assert var_10 is True
    var_11 = var_9.supported_encoding
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = True
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'test.py'
    var_10 = {}
    var_11 = module_1.sort_imports(var_9, var_8, **var_10)
    var_12 = var_11.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'invalid_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'invalid_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = bool(False)
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_preconvert_callable_with_name.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = '--unique'
    var_4 = [var_2, var_3]

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_identify_imports_main_with_default_args. Retrieved 3/10 statements.
# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/12 statements.
# Partially parsed test_identify_imports_main_with_unique_packages. Retrieved 4/11 statements.
# Partially parsed test_identify_imports_main_with_unique_modules. Retrieved 4/11 statements.
# Partially parsed test_identify_imports_main_with_unique_attributes. Retrieved 4/11 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 4/11 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 4/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = []
    var_3 = module_0.identify_imports_main()

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = '-'
    var_2 = 'import os\nimport sys'
    var_3 = [var_2]
    var_4 = []

import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = '--packages'
    var_3 = []
    var_4 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = '--modules'
    var_3 = []
    var_4 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = '--attributes'
    var_3 = []
    var_4 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = '--top-only'
    var_3 = []
    var_4 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = 'identify_imports_main'
    var_1 = 'test_file.py'
    var_2 = '--follow-links'
    var_3 = []
    var_4 = module_0.identify_imports_main()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_argv_is_none. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = module_0.parse_args()
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_identify_imports_main_unique_package. Retrieved 7/10 statements.
# Partially parsed test_identify_imports_main_unique_module. Retrieved 7/10 statements.
# Partially parsed test_identify_imports_main_unique_attribute. Retrieved 7/10 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--packages'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = 'module.submodule'
    var_5 = 'attr'
    var_6 = []
    var_7 = module_0.identify_imports_main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--modules'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = 'module.submodule'
    var_5 = 'attr'
    var_6 = []
    var_7 = module_0.identify_imports_main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--attributes'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = 'module.submodule'
    var_5 = 'attr'
    var_6 = []
    var_7 = module_0.identify_imports_main(var_2, var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 5/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = None
    var_4 = module_0.parse_args(var_3)
    var_5 = bool(var_4 == {'arg1': True, 'arg2': True})
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'arg1': True, 'arg2': True})
    assert var_4 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_true_and_file_is_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_true_and_file_is_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_false_and_file_is_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_when_check_is_false_and_file_is_incorrectly_sorted. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = var_5.skipped
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = var_5.skipped
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.supported_encoding
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_sort_imports_check_mode_success. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_unsupported_encoding. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_normal_mode_success. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_unsupported_encoding. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.incorrectly_sorted
    assert var_11 is False
    var_12 = var_10.skipped
    assert var_12 is False
    var_13 = var_10.supported_encoding
    assert var_13 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.skipped
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = True
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'test.py'
    var_10 = {}
    var_11 = module_1.sort_imports(var_9, var_8, var_2, **var_10)
    var_12 = var_11.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    var_10 = var_9.skipped
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = True
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'test.py'
    var_10 = {}
    var_11 = module_1.sort_imports(var_9, var_8, **var_10)
    var_12 = var_11.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, **var_8)
    assert var_9 is None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 1/6 statements.
# Failed to parse test_preconvert_callable.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)
    var_6 = bool(var_5 == [4, 5, 6])
    assert var_6 is True

def test_case_0():
    var_0 = 'WRAP'

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/some/path'

import isort.main as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0._preconvert(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_identify_imports_main_unique_package. Retrieved 6/13 statements.
# Partially parsed test_identify_imports_main_unique_module. Retrieved 6/13 statements.
# Partially parsed test_identify_imports_main_unique_attribute. Retrieved 6/13 statements.
# Partially parsed test_identify_imports_main_default. Retrieved 5/12 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--unique'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = 'import os\nimport sys\nfrom collections import defaultdict'
    var_4 = 'os\nsys\ncollections\n'
    var_5 = module_0.identify_imports_main()
    var_6 = bool(var_1 == var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--modules'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = 'import os\nimport sys\nfrom collections import defaultdict'
    var_4 = 'os\nsys\ncollections.defaultdict\n'
    var_5 = module_0.identify_imports_main()
    var_6 = bool(var_1 == var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--attributes'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = 'import os\nimport sys\nfrom collections import defaultdict'
    var_4 = 'os\nsys\ncollections.defaultdict\n'
    var_5 = module_0.identify_imports_main()
    var_6 = bool(var_1 == var_4)
    assert var_6 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'import os\nimport sys\nfrom collections import defaultdict'
    var_3 = 'os\nsys\ncollections.defaultdict\n'
    var_4 = module_0.identify_imports_main()



# Parsed testcases at query #40
#--------------------------




import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/example/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/example/path'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_build_arg_parser. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0._build_arg_parser()
    var_1 = var_0.description
    var_2 = bool(var_0.description is not None)
    assert var_2 is True
    var_3 = var_0.add_help
    assert var_3 is False
    var_4 = var_0._action_groups
    var_5 = len(var_4)
    assert var_5 == 6
    var_6 = var_0._action_groups[0].title
    assert var_6 == 'general options'
    var_7 = var_0._action_groups[1].title
    assert var_7 == 'target options'
    var_8 = var_0._action_groups[2].title
    assert var_8 == 'general output options'
    var_9 = var_0._action_groups[3].title
    assert var_9 == 'section output options'
    var_10 = var_0._action_groups[4].title
    assert var_10 == 'deprecated options'
    var_11 = var_0._actions
    var_12 = len(var_11)
    var_13 = bool(var_12 > 50)
    assert var_13 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_non_check_mode_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_non_check_mode_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = var_7.incorrectly_sorted
    assert var_8 is True
    var_9 = var_7.skipped
    assert var_9 is False
    var_10 = var_7.supported_encoding
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = var_7.incorrectly_sorted
    assert var_8 is False
    var_9 = var_7.skipped
    assert var_9 is True
    var_10 = var_7.supported_encoding
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is True
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is True
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'color_output'
    var_3 = 'verbose'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, var_0, **var_7)
    var_9 = var_8.incorrectly_sorted
    assert var_9 is False
    var_10 = var_8.skipped
    assert var_10 is False
    var_11 = var_8.supported_encoding
    assert var_11 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    assert var_6 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sort_attempt_with_skipped_file. Retrieved 3/4 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    var_5 = var_4.skipped
    assert var_5 is True
    var_6 = var_4.incorrectly_sorted
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_normal_mode_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_7.incorrectly_sorted is True or var_7.incorrectly_sorted is False)
    assert var_8 is True
    var_9 = var_7.skipped
    assert var_9 is False
    var_10 = var_7.supported_encoding
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_7.skipped is True or var_7.skipped is False)
    assert var_8 is True
    var_9 = var_7.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    var_7 = bool(var_6.incorrectly_sorted is True or var_6.incorrectly_sorted is False)
    assert var_7 is True
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    var_7 = bool(var_6.skipped is True or var_6.skipped is False)
    assert var_7 is True
    var_8 = var_6.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'color_output'
    var_3 = 'verbose'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, var_0, **var_7)
    var_9 = var_8.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'invalid.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    assert var_6 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'invalid.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, var_0, **var_5)
    assert var_6 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.py'
    var_7 = False
    var_8 = {}
    var_9 = module_1.sort_imports(var_6, var_5, var_7, **var_8)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 1/6 statements.
# Failed to parse test_preconvert_callable.
# Failed to parse test_preconvert_unserializable.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)
    var_6 = bool(var_5 == [4, 5, 6])
    assert var_6 is True

def test_case_0():
    var_0 = 'test_mode'

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/some/path'

def test_case_0():
    pass



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_as_digit. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = module_0.parse_args()
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'old_arg'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = module_0.parse_args(var_1)
    var_4 = bool(var_3 == {'remapped_deprecated_args': ['old_arg']})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float_to_top'
    var_1 = '--dont_float_to_top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(False)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 1
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'HANGING'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'non_existent_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '--dont-float-to-top'
    var_4 = [var_3]
    var_5 = module_0.parse_args(var_4)
    var_6 = 'float_to_top'
    var_7 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_preconvert_wrapmodes. Retrieved 1/5 statements.
# Failed to parse test_preconvert_callable.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

def test_case_0():
    var_0 = 'WRAP_MODE'

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/some/path'

def test_case_0():
    pass

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = complex(var_0, var_1)
    var_3 = module_0._preconvert(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_sort_imports_check_mode_file_not_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_file_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_non_check_mode_file_not_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_non_check_mode_file_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.incorrectly_sorted
    assert var_11 is False
    var_12 = var_10.skipped
    assert var_12 is False
    var_13 = var_10.supported_encoding
    assert var_13 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = True
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = var_10.incorrectly_sorted
    assert var_11 is False
    var_12 = var_10.skipped
    assert var_12 is True
    var_13 = var_10.supported_encoding
    assert var_13 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is True
    var_12 = var_9.supported_encoding
    assert var_12 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = True
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = 'verbose'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'test_file.py'
    var_10 = {}
    var_11 = module_1.sort_imports(var_9, var_8, var_0, **var_10)
    var_12 = var_11.incorrectly_sorted
    assert var_12 is False
    var_13 = var_11.skipped
    assert var_13 is False
    var_14 = var_11.supported_encoding
    assert var_14 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_6, var_0, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = False
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = bool(False)
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{}: {}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = 'format_success'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_file.py'
    var_8 = False
    var_9 = {}
    var_10 = module_1.sort_imports(var_7, var_6, var_8, **var_9)
    var_11 = bool(False)
    assert var_11 is True



