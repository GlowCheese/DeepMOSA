####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_4 = 'test_unsorted.py'
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
    var_4 = 'test_skipped.py'
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
    var_4 = 'test_unsorted.py'
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
    var_4 = 'test_skipped.py'
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
    var_4 = 'nonexistent.py'
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
    var_6 = 'test_encoding.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)
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
    var_4 = 'test_isort_error.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_unexpected_error.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sort_imports_check_false. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_true. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

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

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)
    var_5 = var_4.supported_encoding
    assert var_5 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
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

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = {}
    var_4 = module_1.sort_imports(var_0, var_2, **var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_args_with_none_input. Retrieved 4/7 statements.
# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script.py'
    var_1 = '--some-arg'
    var_2 = 'value'
    var_3 = module_0.parse_args()
    var_4 = bool(var_3 == {'some_arg': 'value'})
    assert var_4 is True

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
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'remapped_deprecated_args': ['arg1', 'arg2']})
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



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 6/9 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_value_error. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/9 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 6/11 statements.
# Partially parsed test_sort_imports_unexpected_error. Retrieved 6/11 statements.


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
    var_4 = True
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, var_4, **var_6)
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
    var_4 = ()
    var_5 = 'test.py'
    var_6 = True
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_3, var_6, **var_7)
    var_9 = var_8.incorrectly_sorted
    assert var_9 is False
    var_10 = var_8.skipped
    assert var_10 is True
    var_11 = var_8.supported_encoding
    assert var_11 is True

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
    var_4 = True
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, **var_6)
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
    var_4 = ()
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, **var_6)
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
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'color_output'
    var_3 = 'verbose'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = ()
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_5, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_21. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #11
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
    var_1 = 'AUTO'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--some-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool('some_arg' in var_3 and var_3['some_arg'] == 'value')
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_main_with_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_with_show_config. Retrieved 1/4 statements.
# Partially parsed test_main_with_show_files. Retrieved 1/4 statements.
# Partially parsed test_main_with_no_files_and_no_arguments. Retrieved 1/4 statements.
# Partially parsed test_main_with_no_files_and_arguments. Retrieved 1/4 statements.
# Partially parsed test_main_with_stream_input. Retrieved 1/4 statements.
# Partially parsed test_main_with_check_stream_input. Retrieved 1/4 statements.
# Partially parsed test_main_with_allow_root. Retrieved 1/4 statements.
# Partially parsed test_main_with_stream_filename_override. Retrieved 1/4 statements.
# Partially parsed test_main_with_skipped_files. Retrieved 1/4 statements.
# Partially parsed test_main_with_broken_paths. Retrieved 1/4 statements.
# Partially parsed test_main_with_unsupported_encoding. Retrieved 1/4 statements.
# Partially parsed test_main_with_deprecated_flags. Retrieved 1/4 statements.
# Partially parsed test_main_with_remapped_deprecated_args. Retrieved 1/4 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'isort'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = '{'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'test.py'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Quick Guide'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Error: arguments passed in without any paths or content.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'import os'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Filename override is intended only for stream (-) sorting.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'was skipped'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'was broken path'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'Encoding not supported'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0501'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0502'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sort_imports_check_success. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_fail. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_sort_success. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_fail. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 4/5 statements.
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
    var_8 = bool(not var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(not var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
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
    var_8 = bool(var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(not var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
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
    var_8 = bool(not var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
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
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
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
    var_7 = bool(var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
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
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'nonexistent.py'
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
    var_9 = bool(not var_8.incorrectly_sorted)
    assert var_9 is True
    var_10 = bool(not var_8.skipped)
    assert var_10 is True
    var_11 = bool(not var_8.supported_encoding)
    assert var_11 is True

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



# Parsed testcases at query #14
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
    var_0 = '-old_arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'old_arg'
    var_5 = bool('old_arg' in var_3)
    assert var_5 is True
    var_6 = var_3['remapped_deprecated_args']
    var_7 = bool(var_3['remapped_deprecated_args'] == ['old_arg'])
    assert var_7 is True

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



# Parsed testcases at query #15
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
    var_0 = 'x'
    var_1 = '--other-arg'
    var_2 = 'value'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'x'
    var_6 = bool('x' in var_4['remapped_deprecated_args'])
    assert var_6 is True
    var_7 = 'other_arg'
    var_8 = bool('other_arg' in var_4)
    assert var_8 is True
    var_9 = var_4['other_arg']
    assert var_9 == 'value'

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
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #16
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
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'remapped_deprecated_args': ['x', 'y'], 'x': None, 'y': None})
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
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output'
    var_1 = 'NORMAL'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = '-h'
    var_2 = [var_1]
    var_3 = parse_args(var_2)[var_0]
    var_4 = bool(var_3 == ['h'])
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
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
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'
    var_4 = bool('remapped_deprecated_args' in var_2)
    assert var_4 is True
    var_5 = var_2['remapped_deprecated_args']
    var_6 = bool(var_2['remapped_deprecated_args'] == ['x'])
    assert var_6 is True
    var_7 = '-x'
    var_8 = bool('-x' in var_2)
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
    var_3 = 'dont_float_to_top'
    var_4 = bool('dont_float_to_top' not in var_2)
    assert var_4 is True
    var_5 = 'float_to_top'

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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_sort_imports_successful_sort. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_mode. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_file_skipped. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_file.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(not var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(not var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skipped_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(var_6.skipped)
    assert var_7 is True
    var_8 = bool(var_6.supported_encoding)
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
    var_6 = 'unsupported_encoding_file.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)
    var_9 = bool(not var_8.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'nonexistent_file.py'
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
    var_4 = 'invalid_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'error_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = b'import sys\nimport os'
    var_1 = [var_0]
    var_2 = 'utf-8'
    var_3 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.identify_imports_main()



# Parsed testcases at query #21
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = 'remapped_deprecated_args'
    var_1 = '-v'
    var_2 = [var_1]
    var_3 = parse_args(var_2)[var_0]
    var_4 = bool(var_3 == ['v'])
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_file_names_is_stdin. Retrieved 2/3 statements.


def test_case_0():
    var_0 = '-'
    var_1 = [var_0]



# Parsed testcases at query #24
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
    var_7 = [var_6, var_1]
    var_8 = parse_args(var_7)[var_0]
    var_9 = bool(var_8 == ['deprecated_arg'])
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


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
    var_8 = bool(not var_7.supported_encoding)
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_identified_imports_iteration. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'path'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'sys'
    var_5 = None
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #27
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
    var_0 = 'dir/'
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



# Parsed testcases at query #28
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
    var_2 = '--another-arg'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_args(var_3)
    var_5 = 'some_arg'
    var_6 = bool('some_arg' in var_4)
    assert var_6 is True
    var_7 = var_4['some_arg']
    assert var_7 == 'value'
    var_8 = 'another_arg'
    var_9 = bool('another_arg' in var_4)
    assert var_9 is True
    var_10 = var_4['another_arg']
    assert var_10 is True

import isort.main as module_0

def test_case_0():
    var_0 = '-x'
    var_1 = '-y'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'remapped_deprecated_args'
    var_5 = bool('remapped_deprecated_args' in var_3)
    assert var_5 is True
    var_6 = var_3['remapped_deprecated_args']
    var_7 = bool(var_3['remapped_deprecated_args'] == ['x', 'y'])
    assert var_7 is True

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
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = var_3['multi_line_output']



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/7 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 3/7 statements.
# Partially parsed test_identify_imports_main_unique_packages. Retrieved 4/8 statements.
# Partially parsed test_identify_imports_main_unique_modules. Retrieved 4/8 statements.
# Partially parsed test_identify_imports_main_unique_attributes. Retrieved 4/8 statements.
# Partially parsed test_identify_imports_main_top_only. Retrieved 4/8 statements.
# Partially parsed test_identify_imports_main_follow_links. Retrieved 4/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--packages'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--modules'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--attributes'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--top-only'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--follow-links'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #30
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 4/7 statements.
# Partially parsed test_sort_imports_oserror. Retrieved 5/9 statements.
# Partially parsed test_sort_imports_valueerror. Retrieved 5/9 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_isorterror. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_generic_exception. Retrieved 5/10 statements.


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
    var_2 = False
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, **var_4)
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
    var_2 = True
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, **var_4)
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
    var_2 = ()
    var_3 = 'test.py'
    var_4 = {}
    var_5 = module_1.sort_imports(var_3, var_1, **var_4)
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
    var_2 = ()
    var_3 = 'test error'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_1, **var_6)
    assert var_7 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test error'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_1, **var_6)
    assert var_7 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, **var_6)
    var_8 = var_7.incorrectly_sorted
    assert var_8 is False
    var_9 = var_7.skipped
    assert var_9 is False
    var_10 = var_7.supported_encoding
    assert var_10 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test error'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_1, **var_6)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ()
    var_3 = 'test error'
    var_4 = [var_3]
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_1, **var_6)



# Parsed testcases at query #32
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--show-version'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.main(var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsupported_encoding_returns_sort_attempt_with_false_supported_encoding. Retrieved 4/5 statements.


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



# Parsed testcases at query #34
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = bool(not var_0)
    assert var_1 is True



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




import isort.api as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.find_imports_in_paths(var_1, **var_2)
    var_4 = '__iter__'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = 'arg1'
    var_1 = 'deprecated_arg'
    var_2 = 'arg2'
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_1}
    var_5 = var_3[1]
    var_6 = bool(var_3[1] in var_4)
    assert var_6 is True



# Parsed testcases at query #39
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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 4/6 statements.
# Partially parsed test_identify_imports_main_unique_packages. Retrieved 4/6 statements.
# Partially parsed test_identify_imports_main_unique_modules. Retrieved 4/6 statements.
# Partially parsed test_identify_imports_main_unique_attributes. Retrieved 4/6 statements.
# Partially parsed test_identify_imports_main_top_only. Retrieved 4/6 statements.
# Partially parsed test_identify_imports_main_follow_links. Retrieved 4/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = [var_0]
    var_2 = '-'
    var_3 = [var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--packages'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--modules'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--attributes'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--top-only'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--follow-links'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_build_arg_parser. Retrieved 35/44 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0._build_arg_parser()
    var_1 = 'Sort Python import definitions alphabetically'
    var_2 = 0
    var_3 = var_0._action_groups
    var_4 = 'general options'
    var_5 = [group for group in var_3 if group.title == var_4][var_2]
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_0._action_groups
    var_8 = 'target options'
    var_9 = [group for group in var_7 if group.title == var_8][var_2]
    var_10 = bool(var_9 is not None)
    assert var_10 is True
    var_11 = var_0._action_groups
    var_12 = 'general output options'
    var_13 = [group for group in var_11 if group.title == var_12][var_2]
    var_14 = bool(var_13 is not None)
    assert var_14 is True
    var_15 = var_0._action_groups
    var_16 = 'section output options'
    var_17 = [group for group in var_15 if group.title == var_16][var_2]
    var_18 = bool(var_17 is not None)
    assert var_18 is True
    var_19 = var_0._action_groups
    var_20 = 'deprecated options'
    var_21 = [group for group in var_19 if group.title == var_20][var_2]
    var_22 = bool(var_21 is not None)
    assert var_22 is True
    var_23 = var_13._group_actions[var_2]
    var_24 = var_0._actions
    var_25 = var_0._actions
    var_26 = var_0._actions
    var_27 = 'files'
    var_28 = [action for action in var_26 if action.dest == var_27][var_2]
    var_29 = var_28.nargs
    assert var_29 == '*'
    var_30 = var_0._actions
    var_31 = 'append'
    var_32 = [action for action in var_30 if action.action == var_31]
    var_33 = len(var_32)
    var_34 = bool(var_33 > 0)
    assert var_34 is True
    var_35 = var_0._actions
    var_36 = 'store_true'
    var_37 = [action for action in var_35 if action.action == var_36]
    var_38 = len(var_37)
    var_39 = bool(var_38 > 0)
    assert var_39 is True
    var_40 = var_0._actions
    var_41 = 'multi_line_output'
    var_42 = [action for action in var_40 if action.dest == var_41][var_2]
    var_43 = var_42.choices
    var_44 = bool(var_42.choices is not None)
    assert var_44 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = bool(var_0 == {'some_arg': 'value'})
    assert var_1 is True

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
    var_0 = '-a'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = bool(var_3 == {'a': 'value', 'remapped_deprecated_args': ['a']})
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



# Parsed testcases at query #3
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_4 = 'test.py'
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
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 5/10 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 6/10 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 7/11 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 7/11 statements.
# Partially parsed test_identify_imports_main_with_unique. Retrieved 7/11 statements.
# Partially parsed test_identify_imports_main_with_packages. Retrieved 6/11 statements.
# Partially parsed test_identify_imports_main_with_modules. Retrieved 6/11 statements.
# Partially parsed test_identify_imports_main_with_attributes. Retrieved 6/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = 0
    var_2 = '-'
    var_3 = [var_2]
    var_4 = module_0.identify_imports_main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0, var_1]
    var_5 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--top-only'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = False
    var_6 = True

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--follow-links'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = False
    var_6 = True

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--unique'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = True
    var_6 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--packages'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--modules'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = '--attributes'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)
    var_4 = [var_0]
    var_5 = False



# Parsed testcases at query #6
#--------------------------




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
    var_4 = 'nonexistent.py'
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
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 5/13 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 7/13 statements.
# Partially parsed test_identify_imports_main_with_unique_packages. Retrieved 8/18 statements.
# Partially parsed test_identify_imports_main_with_unique_modules. Retrieved 8/18 statements.
# Partially parsed test_identify_imports_main_with_unique_attributes. Retrieved 9/19 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 6/11 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 6/11 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = [var_0]
    var_2 = 'sys'
    var_3 = 'os'
    var_4 = module_0.identify_imports_main()
    var_5 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = 'os'
    var_2 = module_0.identify_imports_main()
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = [var_3, var_4]
    var_6 = False

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.path'
    var_1 = 'os.path'
    var_2 = module_0.identify_imports_main()
    var_3 = 'file.py'
    var_4 = [var_3]
    var_5 = False
    var_6 = 'sys'
    var_7 = 'os'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys.path'
    var_1 = 'os.path'
    var_2 = module_0.identify_imports_main()
    var_3 = 'file.py'
    var_4 = [var_3]
    var_5 = False
    var_6 = 'sys.path'
    var_7 = 'os.path'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = 'path'
    var_2 = 'os'
    var_3 = module_0.identify_imports_main()
    var_4 = 'file.py'
    var_5 = [var_4]
    var_6 = False
    var_7 = 'sys.path'
    var_8 = 'os.path'

import isort.main as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0.identify_imports_main()
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = False
    var_5 = True

import isort.main as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0.identify_imports_main()
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = False
    var_5 = True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sort_imports_isorterror. Retrieved 5/10 statements.
# Partially parsed test_sort_imports_exception. Retrieved 3/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
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
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.incorrectly_sorted
    assert var_5 is False
    var_6 = var_4.skipped
    assert var_6 is False
    var_7 = var_4.supported_encoding
    assert var_7 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = 'test'
    var_6 = 1

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_main_no_args_shows_quick_guide. Retrieved 1/4 statements.
# Partially parsed test_main_show_version. Retrieved 1/4 statements.
# Partially parsed test_main_show_config. Retrieved 1/4 statements.
# Partially parsed test_main_show_files. Retrieved 1/4 statements.
# Partially parsed test_main_stream_input. Retrieved 1/5 statements.
# Partially parsed test_main_stream_input_check. Retrieved 1/4 statements.
# Partially parsed test_main_deprecated_flags_warning. Retrieved 1/4 statements.
# Partially parsed test_main_remapped_deprecated_args_warning. Retrieved 1/4 statements.


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

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_sort_attempt_unsupported_encoding. Retrieved 5/6 statements.


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
    var_7 = module_1.sort_imports(var_0, var_4, var_5, var_5, var_5, **var_6)
    var_8 = bool(not var_7.supported_encoding)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sort_imports_check_false. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_true. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = True
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

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

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    var_6 = var_5.supported_encoding
    assert var_6 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = False
    var_4 = {}
    var_5 = module_1.sort_imports(var_0, var_2, var_3, **var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_main_version_flag. Retrieved 1/4 statements.
# Partially parsed test_main_no_files_or_content. Retrieved 1/4 statements.
# Partially parsed test_main_virtual_env_not_exists. Retrieved 3/8 statements.
# Partially parsed test_main_stream_input. Retrieved 1/5 statements.
# Partially parsed test_main_root_path_without_allow_root. Retrieved 2/6 statements.
# Partially parsed test_main_filename_override_without_stream. Retrieved 2/6 statements.
# Partially parsed test_main_show_files. Retrieved 2/6 statements.
# Partially parsed test_main_deprecated_flags. Retrieved 3/6 statements.
# Partially parsed test_main_remapped_deprecated_args. Retrieved 3/6 statements.
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
    var_0 = module_0.parse_args()
    var_1 = var_0['settings_file']
    assert var_1 == 'setup.cfg'
    var_2 = var_0['settings_path']
    assert var_2 == '.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = var_0['settings_path']
    assert var_1 == 'config'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'virtual_env dir does not exist: venv'
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
    var_1 = 'W0501: The following deprecated CLI flags were used and ignored: !'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'W0502: The following deprecated single dash CLI flags were used and translated: c!'
    var_2 = 2

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()
    var_1 = 'No valid encodings.'

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()

import isort.main as module_0

def test_case_0():
    var_0 = module_0.main()



# Parsed testcases at query #14
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
    assert var_6 is False
    var_7 = var_5.skipped
    assert var_7 is True
    var_8 = var_5.supported_encoding
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.parse_args(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_remapped_deprecated_args_predicate. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'old_arg'
    var_1 = [var_0]
    var_2 = '-new_arg'
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #17
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
    var_2 = 'correctly_sorted_file.py'
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
    var_2 = 'incorrectly_sorted_file.py'
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
    var_2 = 'skipped_file.py'
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
    var_2 = 'correctly_sorted_file.py'
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
    var_2 = 'incorrectly_sorted_file.py'
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
    var_2 = 'skipped_file.py'
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_sort_imports_check_true_returns_sort_attempt. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_false_returns_sort_attempt. Retrieved 4/5 statements.


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
    var_6 = var_5.skipped
    assert var_6 is True

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
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)
    var_5 = var_4.supported_encoding
    assert var_5 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = {}
    var_4 = module_1.sort_imports(var_2, var_1, **var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_main_with_no_files_and_no_show_config. Retrieved 5/7 statements.


import isort.main as module_0
import locale as module_1

def test_case_0():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.main(var_1, var_2)
    var_4 = module_1.str(var_0)
    var_5 = 'Error: arguments passed in without any paths or content.'
    var_6 = bool('Error: arguments passed in without any paths or content.' in var_4)
    assert var_6 is True



# Parsed testcases at query #20
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
    var_6 = False
    var_7 = module_1.SortAttempt(var_6, var_3, var_3)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_skipped. Retrieved 6/9 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_sort_skipped. Retrieved 5/8 statements.
# Partially parsed test_sort_imports_os_error. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_value_error. Retrieved 6/10 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/9 statements.
# Partially parsed test_sort_imports_isort_error. Retrieved 6/11 statements.
# Partially parsed test_sort_imports_generic_exception. Retrieved 6/11 statements.


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
    var_4 = True
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, var_4, **var_6)
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
    var_4 = ()
    var_5 = 'test.py'
    var_6 = True
    var_7 = {}
    var_8 = module_1.sort_imports(var_5, var_3, var_6, **var_7)
    var_9 = var_8.incorrectly_sorted
    assert var_9 is False
    var_10 = var_8.skipped
    assert var_10 is True
    var_11 = var_8.supported_encoding
    assert var_11 is True

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
    var_4 = True
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, **var_6)
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
    var_4 = ()
    var_5 = 'test.py'
    var_6 = {}
    var_7 = module_1.sort_imports(var_5, var_3, **var_6)
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
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)
    assert var_9 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'color_output'
    var_3 = 'verbose'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = ()
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_5, **var_8)
    var_10 = var_9.incorrectly_sorted
    assert var_10 is False
    var_11 = var_9.skipped
    assert var_11 is False
    var_12 = var_9.supported_encoding
    assert var_12 is False

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ()
    var_5 = 'test error'
    var_6 = [var_5]
    var_7 = 'test.py'
    var_8 = {}
    var_9 = module_1.sort_imports(var_7, var_3, **var_8)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.
# Partially parsed test_identify_imports_main_with_files. Retrieved 1/7 statements.
# Partially parsed test_identify_imports_main_with_top_only. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_unique. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_packages. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_modules. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_attributes. Retrieved 2/8 statements.
# Partially parsed test_identify_imports_main_with_follow_links. Retrieved 2/8 statements.


def test_case_0():
    var_0 = b'import sys\nimport os'
    var_1 = [var_0]
    var_2 = 'utf-8'
    var_3 = '-'
    var_4 = [var_3]
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = bool(True)
    assert var_1 is True

def test_case_0():
    var_0 = 'import sys\nimport os\ndef foo():\n    import json'
    var_1 = '--top-only'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'import sys\nimport os\nimport sys'
    var_1 = '--unique'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'import sys\nimport os.path'
    var_1 = '--packages'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'import sys\nimport os.path'
    var_1 = '--modules'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv'
    var_1 = '--attributes'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'import sys\nimport os'
    var_1 = '--follow-links'
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_21. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_sort_imports_check_correctly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_check_skipped_file. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_sort_correctly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_incorrectly_sorted. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_sort_skipped_file. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'correctly_sorted_file.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(not var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(not var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'incorrectly_sorted_file.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(not var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skipped_file.py'
    var_5 = True
    var_6 = {}
    var_7 = module_1.sort_imports(var_4, var_3, var_5, **var_6)
    var_8 = bool(not var_7.incorrectly_sorted)
    assert var_8 is True
    var_9 = bool(var_7.skipped)
    assert var_9 is True
    var_10 = bool(var_7.supported_encoding)
    assert var_10 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'correctly_sorted_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'incorrectly_sorted_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(not var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
    assert var_9 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skipped_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    var_7 = bool(not var_6.incorrectly_sorted)
    assert var_7 is True
    var_8 = bool(var_6.skipped)
    assert var_8 is True
    var_9 = bool(var_6.supported_encoding)
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
    var_6 = 'unsupported_encoding_file.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)
    var_9 = bool(not var_8.incorrectly_sorted)
    assert var_9 is True
    var_10 = bool(not var_8.skipped)
    assert var_10 is True
    var_11 = bool(not var_8.supported_encoding)
    assert var_11 is True

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'nonexistent_file.py'
    var_5 = {}
    var_6 = module_1.sort_imports(var_4, var_3, **var_5)
    assert var_6 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'ERROR: {error} - {message}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'file_with_isort_error.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = 'ERROR: {error} - {message}'
    var_2 = 'color_output'
    var_3 = 'format_error'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'file_causing_exception.py'
    var_7 = {}
    var_8 = module_1.sort_imports(var_6, var_5, **var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 4/7 statements.


def test_case_0():
    var_0 = b'import sys\nimport os'
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_identified_imports_is_iterable. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'attribute'
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dont_float_to_top_with_float_to_top_set. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 5/6 statements.


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
    var_7 = module_1.sort_imports(var_0, var_4, var_5, var_5, var_5, **var_6)
    var_8 = bool(not var_7.supported_encoding)
    assert var_8 is True



# Parsed testcases at query #29
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
    var_6 = False
    var_7 = module_1.SortAttempt(var_6, var_3, var_3)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_numeric. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()
    var_1 = bool(var_0 == {})
    assert var_1 is True

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
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



