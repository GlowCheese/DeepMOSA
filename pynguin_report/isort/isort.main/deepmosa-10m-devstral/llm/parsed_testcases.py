####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.


def test_case_0():
    var_0 = b'import sys'
    var_1 = [var_0]
    var_2 = 'utf-8'
    var_3 = '-'
    var_4 = [var_3]

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--unique'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = '--top-only'
    var_3 = '--follow-links'
    var_4 = '--unique'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.identify_imports_main(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_main_with_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_with_virtual_env_nonexistent. Retrieved 3/7 statements.
# Partially parsed test_main_with_no_files_and_no_show_config. Retrieved 1/4 statements.
# Failed to parse test_main_with_stdout_and_check.
# Failed to parse test_main_with_stdout_and_sort.
# Partially parsed test_main_with_deprecated_flags. Retrieved 3/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'virtual_env dir does not exist: /nonexistent'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0500: Please see the 5.0.0 Upgrade guide: https://pycqa.github.io/isort/docs/upgrade_guides/5.0.0.html'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'remapped_deprecated_args': ['x']})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sort_imports_check_true_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_true_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_true_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_false_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_false_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_false_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_oserror. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_valueerror. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 7/8 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 6/8 statements.
# Partially parsed test_sort_imports_generic_exception. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, var_2, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = 'test.py'
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is True
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'raise FileSkipped'
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    var_8 = var_7.incorrectly_sorted
    assert var_8 is False
    var_9 = var_7.skipped
    assert var_9 is True
    var_10 = var_7.supported_encoding
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'test.py'
    var_4 = False
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, var_2, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'raise FileSkipped'
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    var_8 = var_7.incorrectly_sorted
    assert var_8 is False
    var_9 = var_7.skipped
    assert var_9 is True
    var_10 = var_7.supported_encoding
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'raise OSError'
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    assert var_7 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'raise ValueError'
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    assert var_7 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'raise UnsupportedEncoding'
    var_5 = exec(var_4)
    var_6 = 'test.py'
    var_7 = False
    var_8 = {}
    var_9 = module_1.sort_imports(var_6, var_3, var_7, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "raise ISortError('test error')"
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'raise Exception'
    var_3 = exec(var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_1, var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'some_arg'
    var_2 = bool('some_arg' in var_0)
    assert var_2 is True
    var_3 = var_0['some_arg']
    assert var_3 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = '--custom-arg'
    var_1 = 'custom_value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'custom_arg'
    var_5 = bool('custom_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['custom_arg']
    assert var_6 == 'custom_value'

import isort.main as module_0

def test_case_0():
    var_0 = 'old_arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'old_arg'
    var_4 = bool('old_arg' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['old_arg'])
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
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 1
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #6
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_identified_imports_iteration. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = None
    var_2 = 'sys'
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 2/4 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'some_arg'
    var_2 = bool('some_arg' in var_0)
    assert var_2 is True
    var_3 = var_0['some_arg']
    assert var_3 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = '--custom-arg'
    var_1 = 'custom_value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'custom_arg'
    var_5 = bool('custom_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['custom_arg']
    assert var_6 == 'custom_value'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'old_arg'
    var_2 = bool('old_arg' in var_0['remapped_deprecated_args'])
    assert var_2 is True
    var_3 = 'new_arg'
    var_4 = bool('new_arg' in var_0)
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'order_by_type'
    var_2 = bool('order_by_type' in var_0)
    assert var_2 is True
    var_3 = var_0['order_by_type']
    assert var_3 is False

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'follow_links'
    var_2 = bool('follow_links' in var_0)
    assert var_2 is True
    var_3 = var_0['follow_links']
    assert var_3 is False

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'float_to_top'
    var_2 = bool('float_to_top' in var_0)
    assert var_2 is True
    var_3 = var_0['float_to_top']
    assert var_3 is False

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 2
    var_2 = var_0['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = var_0['multi_line_output']



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1._print_hard_fail(var_7)
    var_9 = bool(True)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'test.py'
    var_9 = 'Custom error message'
    var_10 = module_1._print_hard_fail(var_7, var_8, var_9)
    var_11 = bool(True)
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1._print_hard_fail(var_7)
    var_9 = bool(True)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'test.py'
    var_9 = 'Custom error message'
    var_10 = module_1._print_hard_fail(var_7, var_8, var_9)
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = 'verbose'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_0, var_4, var_5, **var_6)
    var_8 = var_7.supported_encoding
    assert var_8 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dont_float_to_top_without_float_to_top. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = 'arg'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sort_imports_successful_sort. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_skipped_file. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(not var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(not var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(not var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(not var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(not var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(not var_5.skipped)
    assert var_7 is True
    var_8 = bool(not var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'argv'
    var_1 = parse_args()[var_0]



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, var_3, var_3, **var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_21. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #17
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = var_2['float_to_top']
    assert var_3 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_main_with_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_with_no_files_and_no_show_config. Retrieved 1/4 statements.
# Partially parsed test_main_with_virtual_env. Retrieved 3/8 statements.
# Partially parsed test_main_with_stream_filename_and_no_stream. Retrieved 2/5 statements.
# Partially parsed test_main_with_check_and_incorrectly_sorted. Retrieved 2/6 statements.
# Partially parsed test_main_with_no_valid_encodings. Retrieved 2/6 statements.
# Partially parsed test_main_with_deprecated_flags. Retrieved 3/6 statements.
# Partially parsed test_main_with_remapped_deprecated_args. Retrieved 3/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'virtual_env dir does not exist: /path/to/venv'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = bool(True)
    assert var_1 is True

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 1

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 1

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 1

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0501: The following deprecated CLI flags were used and ignored: dont_order_by_type!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0502: The following deprecated single dash CLI flags were used and translated: o!'
    var_2 = 2



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = 'verbose'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_0, var_4, var_5, **var_6)
    var_8 = module_1.SortAttempt(var_5, var_5, var_5)
    var_9 = bool(var_7 == var_8)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_parse_args_with_none_argv. Retrieved 3/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_identified_imports_is_iterable. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'os.path'
    var_1 = 'path'
    var_2 = 1
    var_3 = 'import os.path'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'sys'
    var_6 = None
    var_7 = 2
    var_8 = 'import sys'
    var_9 = [var_5, var_6, var_7, var_8]



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = '-h'
    var_2 = [var_1]
    var_3 = parse_args(var_2)[var_0]
    var_4 = bool(var_3 == ['h'])
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['x'])
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_identified_imports_is_iterable. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #25
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--show-version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_main_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_no_files_or_content. Retrieved 1/4 statements.
# Partially parsed test_main_virtual_env_invalid. Retrieved 3/7 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 3/8 statements.
# Partially parsed test_main_filename_override_without_stream. Retrieved 2/6 statements.
# Partially parsed test_main_deprecated_flags_warning. Retrieved 3/6 statements.
# Partially parsed test_main_remapped_deprecated_args_warning. Retrieved 3/6 statements.
# Partially parsed test_main_no_valid_encodings_exit. Retrieved 2/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = var_0['settings_file']
    assert var_1 == '/path/settings.cfg'
    var_2 = var_0['settings_path']
    assert var_2 == '/path'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = var_0['settings_path']
    assert var_1 == '/path'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'virtual_env dir does not exist: /invalid/path'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = "it is dangerous to operate recursively on '/'"
    var_2 = 'use --allow-root to override this failsafe'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Filename override is intended only for stream (-) sorting.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0501: The following deprecated CLI flags were used and ignored: dont_order_by_type!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0502: The following deprecated single dash CLI flags were used and translated: o!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'No valid encodings.'



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = '-v'
    var_2 = [var_1]
    var_3 = parse_args(var_2)[var_0]
    var_4 = bool(var_3 == ['v'])
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'file.py'
    var_1 = True
    var_2 = 'verbose'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_0, var_4, var_5, **var_6)
    var_8 = var_7.supported_encoding
    assert var_8 is False



# Parsed testcases at query #29
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = bool(not var_6.supported_encoding)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = '-v'
    var_2 = [var_1]
    var_3 = parse_args(var_2)[var_0]
    var_4 = bool(var_3 == ['v'])
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 4/6 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 4/6 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 4/6 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/7 statements.
# Partially parsed test_sort_imports_isorterror. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_unexpected_error. Retrieved 5/9 statements.


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
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
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
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
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
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
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
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    assert var_6 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    assert var_6 is None

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
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)
    var_9 = bool(var_1)
    assert var_9 is True
    var_10 = var_8.incorrectly_sorted
    assert var_10 is False
    var_11 = var_8.skipped
    assert var_11 is False
    var_12 = var_8.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = 1

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'some_arg'
    var_2 = bool('some_arg' in var_0)
    assert var_2 is True
    var_3 = var_0['some_arg']
    assert var_3 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'some_arg'
    var_5 = bool('some_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['some_arg']
    assert var_6 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'remapped_deprecated_args'
    var_2 = bool('remapped_deprecated_args' in var_0)
    assert var_2 is True
    var_3 = 'old_arg'
    var_4 = bool('old_arg' in var_0['remapped_deprecated_args'])
    assert var_4 is True
    var_5 = 'new_arg'
    var_6 = bool('new_arg' in var_0)
    assert var_6 is True
    var_7 = var_0['new_arg']
    assert var_7 == 'value'

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
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 2
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'SOME_MODE'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dont_float_to_top_with_float_to_top_false. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = True



# Parsed testcases at query #34
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'correctly_sorted.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'incorrectly_sorted.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'correctly_sorted.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'incorrectly_sorted.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is True
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is True
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'unsupported_encoding.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'isorterror.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'unexpected_error.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #35
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, var_3, var_3, **var_4)



# Parsed testcases at query #36
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'correctly_sorted_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'incorrectly_sorted_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped_file.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'correctly_sorted_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'incorrectly_sorted_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is True
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is True
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'unsupported_encoding_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'isort_error_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'unexpected_error_file.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #37
#--------------------------




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



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = 'old_arg'



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = bool(['-'] == ['-'])
    assert var_2 is True



# Parsed testcases at query #40
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.incorrectly_sorted
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = bool(['-'] == ['-'])
    assert var_2 is True



# Parsed testcases at query #42
#--------------------------




import isort.settings as module_0
import isort.api as module_1
import isort.main as module_2

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_file.py'
    var_5 = {}
    var_6 = module_1.check_file(var_4, config=var_3, **var_5)
    var_7 = {}
    var_8 = module_2.sort_imports(var_4, var_3, **var_7)
    var_9 = var_8.supported_encoding
    assert var_9 is False



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_main_function_exists.




# Parsed testcases at query #44
#--------------------------

# Partially parsed test_main_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_no_files_or_content. Retrieved 1/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_main_with_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_with_no_files_and_no_arguments. Retrieved 1/4 statements.
# Partially parsed test_main_with_virtual_env_invalid. Retrieved 3/8 statements.
# Failed to parse test_main_with_stream_input_check.
# Failed to parse test_main_with_stream_input_sort.
# Partially parsed test_main_with_root_path_no_allow_root. Retrieved 3/8 statements.
# Partially parsed test_main_with_filename_override_not_stream. Retrieved 2/6 statements.
# Partially parsed test_main_with_show_files. Retrieved 2/6 statements.
# Partially parsed test_main_with_skipped_files. Retrieved 2/6 statements.
# Partially parsed test_main_with_broken_paths. Retrieved 2/8 statements.
# Partially parsed test_main_with_deprecated_flags. Retrieved 3/7 statements.
# Partially parsed test_main_with_remapped_deprecated_args. Retrieved 3/7 statements.
# Partially parsed test_main_with_no_valid_encodings. Retrieved 2/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'virtual_env dir does not exist: invalid_path'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = "it is dangerous to operate recursively on '/'"
    var_2 = 'use --allow-root to override this failsafe'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Filename override is intended only for stream (-) sorting.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'file.py'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Skipped 1 files'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Broken 1 paths'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0501: The following deprecated CLI flags were used and ignored: dont_order_by_type!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0502: The following deprecated single dash CLI flags were used and translated: o!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'No valid encodings.'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'some_arg'
    var_5 = bool('some_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['some_arg']
    assert var_6 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = 'old-arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'old_arg'
    var_4 = bool('old_arg' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['old-arg'])
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
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 1
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'SOME_MODE'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_true_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_true_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_true_skipped. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_check_false_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_false_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_false_skipped. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_oserror. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_valueerror. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/9 statements.
# Partially parsed test_sort_imports_isorterror. Retrieved 6/11 statements.
# Partially parsed test_sort_imports_exception. Retrieved 6/11 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, var_2, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = 'test.py'
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is True
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test.py'
    var_4 = True
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is True
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = 'test.py'
    var_4 = False
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is False
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, var_2, **var_4)
    var_6 = var_5.incorrectly_sorted
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test.py'
    var_4 = False
    var_5 = {}
    var_6 = module_1.sort_imports(var_3, var_1, var_4, **var_5)
    var_7 = var_6.incorrectly_sorted
    assert var_7 is False
    var_8 = var_6.skipped
    assert var_8 is True
    var_9 = var_6.supported_encoding
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = False
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_1, var_6, **var_7)
    assert var_8 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = False
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_1, var_6, **var_7)
    assert var_8 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test.py'
    var_6 = False
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_3, var_6, **var_7)
    var_9 = var_8.incorrectly_sorted
    assert var_9 is False
    var_10 = var_8.skipped
    assert var_10 is False
    var_11 = var_8.supported_encoding
    assert var_11 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = False
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_1, var_6, **var_7)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = False
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_1, var_6, **var_7)



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = False
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = var_7.supported_encoding
    assert var_8 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'remapped_deprecated_args': ['x'], 'x': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.incorrectly_sorted
    assert var_8 is False
    var_9 = var_5.supported_encoding
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1._print_hard_fail(var_7)
    var_9 = bool(True)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'test.py'
    var_9 = 'Custom error message'
    var_10 = module_1._print_hard_fail(var_7, var_8, var_9)
    var_11 = bool(True)
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = 'color_output'
    var_4 = 'format_error'
    var_5 = 'format_success'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1._print_hard_fail(var_7)
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'remapped_deprecated_args': ['x'], 'x': True})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import sys'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--unique'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #10
#--------------------------




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
    var_8 = var_7.supported_encoding
    assert var_8 is False



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = var_5.skipped
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_main_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_no_files_or_content. Retrieved 1/5 statements.
# Partially parsed test_main_stream_input. Retrieved 1/6 statements.
# Partially parsed test_main_allow_root. Retrieved 2/6 statements.
# Partially parsed test_main_filename_override. Retrieved 2/6 statements.
# Partially parsed test_main_show_files. Retrieved 2/5 statements.
# Partially parsed test_main_check_incorrectly_sorted. Retrieved 1/5 statements.
# Partially parsed test_main_no_valid_encodings. Retrieved 2/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = "it is dangerous to operate recursively on '/'"

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Filename override is intended only for stream (-) sorting.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'file.py'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'No valid encodings.'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unique_module_predicate. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'remapped_deprecated_args': ['x', 'y']})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 2

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'CLAMP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'some_arg': 'value'})
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(not var_6.supported_encoding)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_parse_args_with_none_input. Retrieved 3/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = 1



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = True
    var_2 = 'verbose'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = False
    var_6 = 'unsupported_encoding'
    var_7 = {var_6: var_1}
    var_8 = 'unsupported_encoding'
    var_9 = {var_8: var_1}
    var_10 = module_1.sort_imports(var_0, var_4, var_5, var_5, var_5, **var_9)
    var_11 = var_10.supported_encoding
    assert var_11 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'some_arg': 'value'})
    assert var_4 is True

import isort.main as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'remapped_deprecated_args': ['x'], 'x': None})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'order_by_type': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'follow_links': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {'float_to_top': False})
    assert var_3 is True

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'SOME_MODE'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_81_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'attr'



# Parsed testcases at query #21
#--------------------------




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
    var_7 = var_5.skipped
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True

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
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is True
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is True
    var_7 = var_4.supported_encoding
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'nonexistent.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'some_arg'
    var_2 = bool('some_arg' in var_0)
    assert var_2 is True
    var_3 = var_0['some_arg']
    assert var_3 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'some_arg'
    var_5 = bool('some_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['some_arg']
    assert var_6 == 'value'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = 'remapped_deprecated_args'
    var_2 = bool('remapped_deprecated_args' in var_0)
    assert var_2 is True
    var_3 = 'old_arg'
    var_4 = bool('old_arg' in var_0['remapped_deprecated_args'])
    assert var_4 is True

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

import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = '2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 2
    var_5 = var_3['multi_line_output']

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'some_mode'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_identify_imports_main_unique_package. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'os.path'
    var_4 = 'path'
    var_5 = 1
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.


def test_case_0():
    var_0 = b'import sys'
    var_1 = [var_0]
    var_2 = 'utf-8'
    var_3 = '-'
    var_4 = [var_3]

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--unique'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 3/4 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 3/4 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 3/4 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(not var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(not var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_unsorted.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(not var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_skip.py'
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_2, var_1, var_3, **var_4)
    var_6 = bool(not var_5.incorrectly_sorted)
    assert var_6 is True
    var_7 = bool(var_5.skipped)
    assert var_7 is True
    var_8 = bool(var_5.supported_encoding)
    assert var_8 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = bool(not var_4.incorrectly_sorted)
    assert var_5 is True
    var_6 = bool(not var_4.skipped)
    assert var_6 is True
    var_7 = bool(var_4.supported_encoding)
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_unsorted.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = bool(var_4.incorrectly_sorted)
    assert var_5 is True
    var_6 = bool(not var_4.skipped)
    assert var_6 is True
    var_7 = bool(var_4.supported_encoding)
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_skip.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = bool(not var_4.incorrectly_sorted)
    assert var_5 is True
    var_6 = bool(var_4.skipped)
    assert var_6 is True
    var_7 = bool(var_4.supported_encoding)
    assert var_7 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_encoding.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(not var_6.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_error.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_unexpected.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, var_3, var_3, **var_4)



# Parsed testcases at query #27
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, var_3, var_3, **var_4)



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'arg1'
    var_1 = 'deprecated_arg'
    var_2 = 'arg2'
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_1}
    var_5 = 'deprecated_arg'
    var_6 = bool('deprecated_arg' in var_4)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = [var_1, var_2]
    var_4 = parse_args(var_3)[var_0]
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = 'deprecated_arg'
    var_7 = [var_1, var_6, var_2]
    var_8 = parse_args(var_7)[var_0]
    var_9 = bool(var_8 == ['deprecated_arg'])
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dont_float_to_top_with_float_to_top_false. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'
    var_4 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_81. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_attr'



# Parsed testcases at query #34
#--------------------------




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
    var_7 = var_5.incorrectly_sorted
    assert var_7 is False
    var_8 = var_5.supported_encoding
    assert var_8 is True



# Parsed testcases at query #35
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = bool(not var_0)
    assert var_1 is True



