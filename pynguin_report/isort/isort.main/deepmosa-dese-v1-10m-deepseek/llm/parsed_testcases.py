####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = '-'
    var_2 = [var_1]

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--top-only'
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
    var_1 = '--unique'
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'l'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'float_to_top'
    var_1 = 'dont_float_to_top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 3

import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output'
    var_1 = 'VERTICAL_HANGING_INDENT'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_multi_line_output_is_not_none_and_is_digit. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'VERTICAL'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = module_0.Config()
    var_4 = module_1._print_hard_fail(var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = module_0.Config()
    var_4 = 'Custom error message'
    var_5 = module_1._print_hard_fail(var_3, message=var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = module_0.Config()
    var_4 = 'example.py'
    var_5 = module_1._print_hard_fail(var_3, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = '{error}: {message}'
    var_2 = '{success}: {message}'
    var_3 = module_0.Config()
    var_4 = module_1._print_hard_fail(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sort_imports_check_mode_with_incorrectly_sorted_file. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_mode_with_skipped_file. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_check_mode_with_correctly_sorted_file. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_with_unsupported_encoding. Retrieved 4/5 statements.


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
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test_file.py'
    var_3 = module_1.sort_imports(var_2, var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = module_1.sort_imports(var_1, var_0)
    assert var_2 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = module_1.sort_imports(var_1, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = module_1.sort_imports(var_1, var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_remapped_deprecated_args_evaluates_to_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '-d'
    var_1 = [var_0]
    var_2 = {var_0}
    var_3 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parse_args_with_argv_none. Retrieved 4/7 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_with_unsupported_encoding. Retrieved 3/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'unsupported_encoding_file.py'
    var_2 = module_1.sort_imports(var_1, var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 6/8 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
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
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'HANGING_INDENT'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--custom_arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = lambda file_name, config, **kwargs: var_0
    var_2 = 'test_file.py'
    var_3 = module_0.Config()
    var_4 = True
    var_5 = module_1.sort_imports(var_2, var_3, var_4)



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = False
    var_3 = module_1.sort_imports(var_1, var_0, var_2, var_2, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parse_args_float_to_top_conflict. Retrieved 5/10 statements.
# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 5/6 statements.
# Partially parsed test_parse_args_multi_line_output_enum. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'some-deprecated-arg'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = None
    var_1 = '--float-to-top'
    var_2 = '--dont-float-to-top'
    var_3 = [var_1, var_2]
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output=1'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = var_2[var_3]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output=VERTICAL_HANGING_INDENT'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = var_2[var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_when_no_exception_occurs. Retrieved 4/5 statements.
# Partially parsed test_sort_imports_returns_sort_attempt_with_unsupported_encoding_on_unsupportedencoding. Retrieved 3/4 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = False
    var_3 = module_1.sort_imports(var_0, var_1, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)
    assert var_2 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = '--foo=bar'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'f'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
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
    var_0 = '--multi_line_output=1'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 1

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output=VERTICAL_HANGING_INDENT'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Config()
    var_2 = True
    var_3 = module_1.sort_imports(var_0, var_1, var_2)



# Parsed testcases at query #16
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.main(var_0, var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-version'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.main(var_1, var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.main(var_1, var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-files'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.main(var_1, var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = '--show-files'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--settings-path'
    var_1 = '/tmp'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/tmp'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '-'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.main(var_2, var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/'
    var_2 = '--allow-root'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0.main(var_3, var_4)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = '--check'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0.main(var_3, var_4)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = '--jobs'
    var_3 = '2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.main(var_4, var_5)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = '--resolve-all-configs'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0.main(var_3, var_4)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = '--color'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0.main(var_3, var_4)

import isort.main as module_0

def test_case_0():
    var_0 = '--files'
    var_1 = '/tmp/file.py'
    var_2 = '--verbose'
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0.main(var_3, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_preconvert_WrapModes. Retrieved 1/6 statements.
# Failed to parse test_preconvert_callable.


import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0._preconvert(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = frozenset(var_3)
    var_5 = module_0._preconvert(var_4)

def test_case_0():
    var_0 = 'test'

import zipfile as module_0
import isort.main as module_1

def test_case_0():
    var_0 = '/test/path'
    var_1 = module_0.Path(var_0)
    var_2 = module_1._preconvert(var_1)
    assert var_2 == '/test/path'

def test_case_0():
    pass

import isort.main as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0._preconvert(var_0)



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'unsupported_file.txt'
    var_3 = module_1.sort_imports(var_2, var_1)



# Parsed testcases at query #19
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #20
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = 'float_to_top'
    var_2 = True
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = '--dont-float-to-top'
    var_5 = '--float-to-top'
    var_6 = [var_4, var_5]
    var_7 = module_0.parse_args(var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_filter_files_evaluates_to_true_when_config_filter_files_is_true. Retrieved 14/16 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'filter_files'
    var_3 = 'is_skipped'
    var_4 = True
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = 'file1.py'
    var_9 = 'file2.py'
    var_10 = [var_8, var_9]
    var_11 = 'files'
    var_12 = {var_11: var_10, var_2: var_4}
    var_13 = module_0.main(var_12)



# Parsed testcases at query #22
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--arg1'
    var_1 = 'value1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 2/6 statements.
# Partially parsed test_parse_args_with_numeric_multi_line_output. Retrieved 5/6 statements.
# Partially parsed test_parse_args_with_string_multi_line_output. Retrieved 5/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = module_0.parse_args()

import isort.main as module_0

def test_case_0():
    var_0 = '--order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_0 = '--multi-line-output=3'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = var_2[var_3]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi-line-output=HANGING_INDENT'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = var_2[var_3]



# Parsed testcases at query #24
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--float-to-top'
    var_1 = '--dont-float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #25
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--show-version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-files'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--settings-path'
    var_1 = '/tmp'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = '/tmp'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = 'file1.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--ask-to-apply'
    var_1 = 'file1.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--write-to-stdout'
    var_1 = 'file1.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--deprecated-flags'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-d'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '-'
    var_1 = '--filename'
    var_2 = 'file1.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

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
    var_0 = '--jobs'
    var_1 = '2'
    var_2 = 'file1.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--resolve-all-configs'
    var_1 = 'file1.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--filter-files'
    var_1 = 'file1.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_main_with_stdin. Retrieved 3/6 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '--show-version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-config'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-files'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--check'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = '-'
    var_2 = [var_1]

import isort.main as module_0

def test_case_0():
    var_0 = '/'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--allow-root'
    var_1 = '/'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '-ac'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file_with_invalid_encoding.py'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = 'non_existent_file.py'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--skip'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--quiet'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--color'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.main(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--settings-path'
    var_1 = '.'
    var_2 = 'test_file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--virtual-env'
    var_1 = 'venv'
    var_2 = 'test_file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--jobs'
    var_1 = '2'
    var_2 = 'test_file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--show-diff'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--stdout'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--ext-format'
    var_1 = 'py'
    var_2 = 'test_file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = '--resolve-all-configs'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.main(var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_with_no_arguments. Retrieved 2/3 statements.
# Partially parsed test_parse_args_with_remapped_deprecated_args. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_order_by_type. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_follow_links. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_dont_float_to_top. Retrieved 4/5 statements.
# Partially parsed test_parse_args_with_multi_line_output_digit. Retrieved 5/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_args(var_0)

import isort.main as module_0

def test_case_0():
    var_0 = 'd'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'remapped_deprecated_args'

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'order_by_type'

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'follow_links'

import isort.main as module_0

def test_case_0():
    var_0 = 'dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'float_to_top'

import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output=5'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    var_3 = 'multi_line_output'
    var_4 = 5

import isort.main as module_0

def test_case_0():
    var_0 = 'multi_line_output=VERTICAL_HANGING_INDENT'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)



# Parsed testcases at query #2
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont-float-to-top'
    var_1 = '--float-to-top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_main_show_version. Retrieved 4/10 statements.
# Partially parsed test_main_show_config. Retrieved 4/10 statements.
# Partially parsed test_main_show_files. Retrieved 4/10 statements.
# Partially parsed test_main_check. Retrieved 5/11 statements.
# Partially parsed test_main_sort. Retrieved 4/10 statements.
# Partially parsed test_main_stdin. Retrieved 4/10 statements.


import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = '--show-version'
    var_2 = module_0.StringIO()
    var_3 = module_1.main()

import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = '--show-config'
    var_2 = module_0.StringIO()
    var_3 = module_1.main()

import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = '--show-files'
    var_2 = module_0.StringIO()
    var_3 = module_1.main()

import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = '--check'
    var_2 = 'test_file.py'
    var_3 = module_0.StringIO()
    var_4 = module_1.main()

import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = 'test_file.py'
    var_2 = module_0.StringIO()
    var_3 = module_1.main()

import _io as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = '-'
    var_2 = module_0.StringIO()
    var_3 = module_1.main()



# Parsed testcases at query #4
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sort_imports_check_mode_with_incorrectly_sorted_file. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_with_skipped_file. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_with_unsupported_encoding. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_normal_mode_with_incorrectly_sorted_file. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_with_skipped_file. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_with_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_multi_line_output_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_multi_line_output_str. Retrieved 6/7 statements.


import isort.main as module_0

def test_case_0():
    var_0 = '-a'
    var_1 = '-b'
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3}
    var_5 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

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
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'HANGING'
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
    var_0 = '-b'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = {var_2}
    var_4 = module_0.parse_args(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sort_imports_check_mode_success. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_normal_mode_success. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_normal_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_unsupported_encoding. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = True
    var_3 = module_0.Config()
    var_4 = 'test.py'
    var_5 = module_1.sort_imports(var_4, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test.py'
    var_4 = module_1.sort_imports(var_3, var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 4/5 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'script_name'
    var_1 = '--order_by_type'
    var_2 = '--follow_links'
    var_3 = module_0.parse_args()

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = '--dont_follow_links'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '-o'
    var_1 = '-f'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--float_to_top'
    var_1 = '--dont_float_to_top'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '3'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'VERTICAL_HANGING_INDENT'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'example.py'
    var_3 = True
    var_4 = module_1.sort_imports(var_2, var_1, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sort_imports_predicate_on_line_27_evaluates_to_true. Retrieved 4/16 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_file.py'
    var_2 = False
    var_3 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_argv_is_none_uses_sys_argv. Retrieved 5/9 statements.


import isort.main as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'arg1'
    var_2 = 'arg2'
    var_3 = None
    var_4 = module_0.parse_args(var_3)

import isort.main as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'args'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parse_args_with_default_argv. Retrieved 4/7 statements.
# Partially parsed test_parse_args_with_custom_argv. Retrieved 4/5 statements.
# Partially parsed test_parse_args_handles_multi_line_output_as_digit. Retrieved 6/7 statements.
# Partially parsed test_parse_args_handles_multi_line_output_as_string. Retrieved 6/7 statements.


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
    var_0 = '-arg1'
    var_1 = '-arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_order_by_type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_follow_links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--dont_float_to_top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]

import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = 'WRAP'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    var_4 = 'multi_line_output'
    var_5 = var_3[var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sort_attempt_predicate_evaluates_true. Retrieved 3/4 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = module_1.sort_imports(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sort_imports_check_mode_incorrectly_sorted. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_skipped. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_check_mode_unsupported_encoding. Retrieved 6/7 statements.
# Partially parsed test_sort_imports_non_check_mode_incorrectly_sorted. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_non_check_mode_skipped. Retrieved 5/6 statements.
# Partially parsed test_sort_imports_non_check_mode_unsupported_encoding. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = True
    var_5 = module_1.sort_imports(var_3, var_2, var_4)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)
    assert var_4 is None

import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1.sort_imports(var_3, var_2, var_0)
    assert var_4 is None



# Parsed testcases at query #9
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = 'dont_order_by_type'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '--dont-order-by-type'
    var_4 = [var_3]
    var_5 = module_0.parse_args(var_4)



# Parsed testcases at query #10
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '--multi_line_output'
    var_1 = '1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_sort_imports_returns_sort_attempt_with_unsupported_encoding. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'test_file.txt'
    var_3 = module_1.sort_imports(var_2, var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_identify_imports_main_with_stdin. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = '-'
    var_2 = [var_1]

import isort.main as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.identify_imports_main(var_1)

import isort.main as module_0

def test_case_0():
    var_0 = '--top-only'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--follow-links'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--unique'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--packages'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--modules'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = '--attributes'
    var_1 = 'test_file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.identify_imports_main(var_2)



# Parsed testcases at query #13
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)

import isort.main as module_0

def test_case_0():
    var_0 = module_0.parse_args()



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.main as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.Config()
    var_2 = True
    var_3 = module_1.sort_imports(var_0, var_1, var_2)



# Parsed testcases at query #15
#--------------------------




import isort.main as module_0

def test_case_0():
    var_0 = '-o'
    var_1 = [var_0]
    var_2 = 'o'
    var_3 = {var_2}
    var_4 = module_0.parse_args(var_1)



