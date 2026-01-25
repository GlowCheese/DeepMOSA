####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_wrap_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_wrap. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 3/6 statements.
# Partially parsed test_line_use_parentheses_false. Retrieved 4/7 statements.
# Partially parsed test_line_include_trailing_comma_false. Retrieved 7/10 statements.
# Partially parsed test_line_wrap_length_shorter_than_line_length. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True

def test_case_0():
    var_0 = 'object.very_long_method_name(arg1, arg2)'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line # comment'

def test_case_0():
    var_0 = 'long line that needs wrapping # important comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'very long line that should not be wrapped'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'very long line # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'long line with import statement'
    var_1 = '\n'
    var_2 = 20
    var_3 = False

def test_case_0():
    var_0 = 'long line with dot separator'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = False
    var_5 = -1
    var_6 = result.split(var_1)[var_5]
    var_7 = ','
    var_8 = bool(',' not in var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'very long line that needs wrapping'
    var_1 = '\n'
    var_2 = 50
    var_3 = 20
    var_4 = True



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 10
    var_3 = None
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = len(var_0)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_7.wrap_length
    var_12 = var_7.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    var_15 = bool(not var_14)
    assert var_15 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_line_no_wrap_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the line length limit'
    var_1 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_function_name'
    var_2 = 'from module import (\n    long_function_name)'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'object.long_attribute_name'
    var_2 = 'object.(\n    long_attribute_name)'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = 'import module as alias'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # comment'
    var_3 = 'import (\n    module # comment\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # noqa'
    var_3 = 'import module # noqa'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = 'import (\n    module,\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module'
    var_3 = 'import \\\n    module'
    var_4 = '\n'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_line_predicate_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 79
    var_1 = 80
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_import_statement_with_trailing_comma. Retrieved 9/10 statements.
# Partially parsed test_import_statement_with_multi_line_output. Retrieved 5/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from module import (\n    a,\n    b,\n    c,\n)\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = False
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from module import (a, b, c)\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment'
    var_6 = [var_5]
    var_7 = module_0.import_statement(var_0, var_4, var_6)
    var_8 = '# comment'
    var_9 = bool('# comment' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = '\n'
    var_13 = bool('\n' in var_11)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = ',\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' not in var_11)
    assert var_13 is True

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = '    '
    var_11 = bool('    ' in var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '# '
    var_1 = 'comment_prefix'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = 'comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' in var_11)
    assert var_13 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the default line length but has a NOQA comment # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line that exceeds the default line length without NOQA comment'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_module_name, another_long_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    var_9 = bool(var_8 == f'from module import \\{var_1}    long_module_name, another_long_name')
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_module_name, another_long_name # comment'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    var_10 = bool(var_9 == f'from module import ({var_1}    long_module_name,{var_1}) # comment')
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import long_module_name as short_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    var_9 = bool(var_8 == f'import long_module_name as{var_1}    short_name')
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_module_name, another_long_name # noqa: F401'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    var_10 = bool(var_9 == f'from module import ({var_1}    long_module_name,{var_1}    another_long_name,  # noqa: F401{var_1})')
    assert var_10 is True

def test_case_0():
    var_0 = 'from module import long_module_name, another_long_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True

def test_case_0():
    var_0 = 'from module import long_module_name, another_long_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_import_statement_multi_line_output. Retrieved 4/6 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 8/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from os import path, sys'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = '# Comment'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)
    var_7 = '# Comment'
    var_8 = bool('# Comment' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    var_6 = '\n'
    var_7 = bool('\n' in var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = '\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = '\r\n'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = '\r\n'
    var_7 = bool('\r\n' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import'
    var_5 = 'path'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = bool('\n' in var_8 or var_8 == 'from os import path, sys')
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import'
    var_5 = 'path'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import'
    var_5 = 'path'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = '# Comment'
    var_9 = [var_8]
    var_10 = module_1.import_statement(var_4, var_7, var_9, config=var_3)
    var_11 = '# Comment'
    var_12 = bool('# Comment' not in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import'
    var_5 = 'path'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = bool('    ' in var_8 or var_8 == 'from os import path, sys')
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '# '
    var_1 = 'comment_prefix'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import'
    var_5 = 'path'
    var_6 = 'sys'
    var_7 = [var_5, var_6]
    var_8 = '# Comment'
    var_9 = [var_8]
    var_10 = module_1.import_statement(var_4, var_7, var_9, config=var_3)
    var_11 = '# Comment'
    var_12 = bool('# Comment' in var_10)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import'
    var_1 = 'path'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'
    var_5 = bool('\n' not in var_3)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 9/17 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'import os.path as osp'
    var_6 = 'import os.path as osp'
    var_7 = ','
    var_8 = ','



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 14/25 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import ('
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = 'd'
    var_12 = 'e'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = '\n'
    var_15 = -1
    var_16 = -1
    var_17 = 3



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_30_predicate_evaluates_to_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 50
    var_1 = 100
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_wrap_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 10
    var_3 = 5
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = len(var_0)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_7.wrap_length
    var_12 = var_7.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    var_15 = bool(not var_14)
    assert var_15 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = ''
    var_5 = '#'
    var_6 = False
    var_7 = len(var_0)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'module.long_function_name()'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'long_line # some comment'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'
    var_6 = {}
    var_7 = module_0.Config(**var_6)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from module import (\n    a,\n    b,\n    c,\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment'
    var_6 = [var_5]
    var_7 = '\n'
    var_8 = {}
    var_9 = module_0.Config(**var_8)
    var_10 = module_1.import_statement(var_0, var_4, var_6, var_7, var_9)
    assert var_10 == 'from module import (\n    a,  # comment\n    b,\n    c,\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.import_statement(var_0, var_2, line_separator=var_3, config=var_5)
    assert var_6 == 'from module import a\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'very_long_name_a'
    var_8 = 'very_long_name_b'
    var_9 = 'very_long_name_c'
    var_10 = [var_7, var_8, var_9]
    var_11 = '\n'
    var_12 = module_1.import_statement(var_6, var_10, line_separator=var_11, config=var_5)
    assert var_12 == 'from module import (\n    very_long_name_a,\n    very_long_name_b,\n    very_long_name_c,\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = '\n'
    var_9 = module_1.import_statement(var_4, var_7, line_separator=var_8, config=var_3)
    assert var_9 == 'from module import (\n    a,\n    b,\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = False
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = '\n'
    var_9 = module_1.import_statement(var_4, var_7, line_separator=var_8, config=var_3)
    assert var_9 == 'from module import (\n    a,\n    b\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = '\n'
    var_9 = module_1.import_statement(var_4, var_7, line_separator=var_8, config=var_3)
    assert var_9 == 'from module import (\n    a,\n    b,\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = '\r\n'
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.import_statement(var_0, var_3, line_separator=var_4, config=var_6)
    assert var_7 == 'from module import (\r\n    a,\r\n    b,\r\n)\r\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = '\n'
    var_11 = module_1.import_statement(var_4, var_7, var_9, var_10, var_3)
    assert var_11 == 'from module import (\n    a,\n    b,\n)\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_line_noqa_mode_no_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses_and_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_different_line_separator. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment_prefix. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_wrap_length. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_no_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_no_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_trailing_comma. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the line length limit'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line that exceeds the line length limit # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module cimport long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.long_module_name.function'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as long_alias'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_module_name # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '    '
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_module_name'
    var_2 = '\r\n'

def test_case_0():
    var_0 = 20
    var_1 = '# '
    var_2 = 'from module import long_module_name # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 15
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name # noqa: F401'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import long_module_name # noqa: F401'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'from module import long_module_name # noqa: F401'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name # noqa: F401'
    var_3 = '\n'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_import. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'

def test_case_0():
    var_0 = 'cimport module.long_function_name, another_function'
    var_1 = '\n'

def test_case_0():
    var_0 = 'module.long_function_name.another_function'
    var_1 = '\n'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function  # comment'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function  # noqa'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = True
    var_3 = 'include_trailing_comma'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function,\n)'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function  # comment'
    var_1 = '\n'
    var_2 = '# '
    var_3 = 'comment_prefix'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function,  # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'wrap_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = '    '
    var_3 = 'indent'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n        long_function_name,\n        another_function,\n    )'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_no_wrap_needed. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 2/4 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 'import '
    var_3 = '\\'

def test_case_0():
    var_0 = 'from module import very_long_function_name # some comment'
    var_1 = '\n'
    var_2 = '# some comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    var_7 = '('
    var_8 = bool('(' in var_6)
    assert var_8 is True
    var_9 = ')'
    var_10 = bool(')' in var_6)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_3: var_2, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.line(var_0, var_1, var_6)
    var_8 = ','
    var_9 = bool(',' in var_7)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name # noqa'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    var_7 = '# noqa'
    var_8 = bool('# noqa' in var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'import module as very_long_alias_name'
    var_1 = '\n'
    var_2 = 'as '
    var_3 = '\\'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'some_content'
    var_1 = '\n'
    var_2 = True
    var_3 = '#'
    var_4 = 100
    var_5 = None
    var_6 = ''
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_import_statement_multi_line. Retrieved 4/7 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 8/9 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 8/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from foo import bar, baz'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = '# comment'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)
    var_7 = '# comment'
    var_8 = bool('# comment' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    assert var_5 == 'from foo import (\n    bar,\n    baz,\n)'

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = 0
    var_10 = '\n'
    var_11 = var_8.split(var_10)[var_9]
    var_12 = len(var_11)
    var_13 = bool(var_12 <= 20)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = '    '
    var_10 = bool('    ' in var_8)
    assert var_10 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = '\r\n'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = '\r\n'
    var_7 = bool('\r\n' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = module_1.import_statement(var_4, var_7, var_9, config=var_3)
    var_11 = '# comment'
    var_12 = bool('# comment' not in var_10)
    assert var_12 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = len(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 23/30 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = 0
    var_6 = False
    var_7 = False
    var_8 = '#'
    var_9 = '    '
    var_10 = 25
    var_11 = var_0 * var_10
    var_12 = 'b'
    var_13 = var_12 * var_10
    var_14 = 'c'
    var_15 = var_14 * var_10
    var_16 = 'd'
    var_17 = var_16 * var_10
    var_18 = [var_11, var_13, var_15, var_17]
    var_19 = 'import '
    var_20 = len(var_2)
    var_21 = 2
    var_22 = var_20 + var_21



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 5/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = 100
    var_3 = 'line_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'short line'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_10 = module_1.line(var_0, var_1, var_8)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'very.long.module.name.function_call()'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'very.long.module.name(\n    .function_call()\n)'
    var_9 = module_1.line(var_0, var_1, var_7)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import module as very_long_alias_name'
    var_1 = '\n'
    var_2 = 25
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import module as (\n    very_long_alias_name\n)'
    var_9 = module_1.line(var_0, var_1, var_7)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line # some comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = 'long_line(\n    # some comment,\n)'
    var_10 = module_1.line(var_0, var_1, var_8)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = 'long_line # noqa'
    var_10 = module_1.line(var_0, var_1, var_8)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 5
    var_3 = 'long_line # NOQA'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import \\\n    long_function_name, another_function_name'
    var_9 = module_1.line(var_0, var_1, var_7)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_wrap. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 3/6 statements.
# Partially parsed test_line_use_parentheses_false. Retrieved 4/7 statements.
# Partially parsed test_line_include_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_no_trailing_comma. Retrieved 6/10 statements.
# Partially parsed test_line_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_in_comment. Retrieved 3/6 statements.
# Partially parsed test_line_comment_prefix. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_length. Retrieved 4/7 statements.
# Partially parsed test_line_indent. Retrieved 4/7 statements.
# Partially parsed test_line_starts_with_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_no_splitter. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport very_long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.very_long_function_name()'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = '\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line # comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'very_long_line'
    var_2 = '\n'
    var_3 = 'very_long_line # NOQA'

def test_case_0():
    var_0 = 10
    var_1 = 'very_long_line # NOQA'
    var_2 = '\n'
    var_3 = 'very_long_line # NOQA'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import very_long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very_long_module_name'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'import very_long_module_name'
    var_4 = '\n'
    var_5 = ','

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name # noqa'
    var_2 = '\n'
    var_3 = 'import very_long_module_name # noqa'

def test_case_0():
    var_0 = 20
    var_1 = '# '
    var_2 = 'import very_long_module_name # comment'
    var_3 = '\n'
    var_4 = '# comment'

def test_case_0():
    var_0 = 100
    var_1 = 20
    var_2 = 'import very_long_module_name'
    var_3 = '\n'
    var_4 = 'import ('

def test_case_0():
    var_0 = 20
    var_1 = '    '
    var_2 = 'import very_long_module_name'
    var_3 = '\n'
    var_4 = '    '

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'import ('

def test_case_0():
    var_0 = 20
    var_1 = 'very_long_line_without_splitter'
    var_2 = '\n'
    var_3 = 'very_long_line_without_splitter'



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_predicate_at_line_48_evaluates_to_true.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 3/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some content'
    var_3 = ','



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_while_condition_evaluates_to_true. Retrieved 8/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (a, b, c, d, e)'
    var_8 = '\n'
    var_9 = -1
    var_10 = 20
    var_11 = -1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_line_30_predicate_evaluates_to_true. Retrieved 17/19 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = 40
    var_7 = var_2 * var_6
    var_8 = var_2 * var_6
    var_9 = [var_7, var_8]
    var_10 = 'import '
    var_11 = len(var_4)
    var_12 = 2
    var_13 = var_11 + var_12
    var_14 = var_1.wrap_length
    var_15 = var_1.line_length
    var_16 = var_14 or var_15
    var_17 = var_13 > var_16
    var_18 = bool(var_17 and var_9)
    assert var_18 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_line_noqa_mode_no_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma_and_noqa_in_comment_and_noqa_mode. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma_and_noqa_in_comment_and_noqa_mode_and_noqa_in_content. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 10
    var_1 = 'long line without comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = 'from module import \\\n    long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # comment'
    var_7 = 'from module import \\\n    long_module_name # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name'
    var_8 = 'from module import (\n    long_module_name,\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # comment'
    var_8 = 'from module import (\n    long_module_name, # comment\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as long_alias'
    var_7 = 'import module as \\\n    long_alias'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module as long_alias'
    var_8 = 'import module as long_alias'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_attribute_name'
    var_7 = 'module.\\\n    long_attribute_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'module.long_attribute_name'
    var_8 = 'module.\\\n    long_attribute_name'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport module.long_module_name'
    var_7 = 'cimport module.\\\n    long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'cimport module.long_module_name'
    var_8 = 'cimport module.\\\n    long_module_name'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = 'from module import (\n    long_module_name,\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = 'from module import (\n    long_module_name,\n)'
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = 'from module import (\n    long_module_name, # noqa\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa: F401'
    var_8 = 'from module import (\n    long_module_name, # noqa: F401\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import long_module_name # noqa'
    var_9 = 'from module import (\n    long_module_name # noqa\n)'
    var_10 = '\n'
    var_11 = module_1.line(var_8, var_10, var_7)
    var_12 = bool(var_11 == var_9)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # noqa'
    var_7 = 'from module import \\\n    long_module_name # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = 'from module import \\\n    long_module_name # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa: F401'
    var_8 = 'from module import \\\n    long_module_name # noqa: F401'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import long_module_name # noqa: F401'
    var_3 = 'from module import long_module_name # noqa: F401'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import long_module_name # NOQA'
    var_3 = 'from module import long_module_name # NOQA'
    var_4 = '\n'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_30. Retrieved 14/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = [var_2]
    var_7 = var_6 * var_3
    var_8 = len(var_4)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_1.wrap_length
    var_12 = var_1.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    var_15 = bool(var_14 and var_7)
    assert var_15 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_line_wrapping_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_with_existing_noqa. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import function  # some comment'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import function  # noqa'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import very_long_function_name # NOQA'
    var_1 = '\n'
    var_2 = 20



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'some_long_import_statement'
    var_6 = '\n'
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'import os, sys  # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = None
    var_4 = True
    var_5 = '#'
    var_6 = '    '
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = None
    var_6 = '    '
    var_7 = '# '
    var_8 = True
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_false. Retrieved 11/22 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'balanced_wrapping'
    var_4 = 'line_length'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_0}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from x import'
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = '\n'
    var_11 = module_1.import_statement(var_7, var_9, line_separator=var_10, config=var_6)
    var_12 = -1
    var_13 = -1
    var_14 = var_0 > var_0



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = '# '
    var_6 = len(var_2)
    var_7 = '# NOQA'
    var_8 = bool('# NOQA' not in var_2)
    assert var_8 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_71. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = len(var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 17/37 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import ('
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = ()
    var_8 = '\n'
    var_9 = None
    var_10 = False
    var_11 = module_1.import_statement(var_2, var_6, var_7, var_8, var_1, var_9, var_10)
    var_12 = 1
    var_13 = -1
    var_14 = 0
    var_15 = var_1.line_length
    var_16 = -1



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = len(var_0)
    var_5 = 2
    var_6 = var_4 + var_5



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_import_statement_while_loop_predicate_false. Retrieved 14/31 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import ('
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = ()
    var_8 = '\n'
    var_9 = None
    var_10 = False
    var_11 = module_1.import_statement(var_2, var_6, var_7, var_8, var_1, var_9, var_10)
    var_12 = -1
    var_13 = -1
    var_14 = 2



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = len(var_0)
    var_5 = 2
    var_6 = var_4 + var_5
    var_7 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import function1, function2, function3'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'long line # comment'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'long line # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'long line with parentheses'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'long line with trailing comma'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'cimport module.function'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'module.function1.function2'
    var_1 = '\n'
    var_2 = 20



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_line_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = 'include_trailing_comma'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os, sys  # comment'
    var_6 = '\n'
    var_7 = module_1.line(var_5, var_6, var_4)
    assert var_7 == 'import (\n    os,\n    sys,  # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os, sys  # noqa'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import (\n    os,\n    sys  # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import function as alias'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'from module import function as (\n    alias\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module.submodule.function'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'module.submodule.(\n    function\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'cimport module.submodule'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'cimport (\n    module.submodule\n)'

def test_case_0():
    var_0 = True
    var_1 = 'import os, sys'
    var_2 = '\n'

def test_case_0():
    var_0 = True
    var_1 = 'import os, sys'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = False
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os, sys'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os,\\n    sys'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = 'include_trailing_comma'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os, sys'
    var_6 = '\n'
    var_7 = module_1.line(var_5, var_6, var_4)
    assert var_7 == 'import (\n    os,\n    sys,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'use_parentheses'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os, sys'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import (\n    os,\n    sys\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os, sys  # noqa'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import (\n    os,\n    sys  # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = False
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os, sys  # noqa'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os,\\n    sys  # noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import module.submodule.function'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import (\n    module.submodule.function\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os, sys  # noqa: F401'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import (\n    os,\n    sys  # noqa: F401\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = 'include_trailing_comma'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os, sys  # noqa: F401'
    var_6 = '\n'
    var_7 = module_1.line(var_5, var_6, var_4)
    assert var_7 == 'import (\n    os,\n    sys,  # noqa: F401\n)'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_48. Retrieved 1/2 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = var_1.wrap_mode



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = len(var_0)
    var_5 = 2
    var_6 = var_4 + var_5



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = len(var_4)
    var_6 = 2
    var_7 = var_5 + var_6
    var_8 = bool(var_7 > (var_1.wrap_length or var_1.line_length))
    assert var_8 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 14/22 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'import os.path as osp'
    var_6 = '\n'
    var_7 = 'import os.path as osp'
    var_8 = None
    var_9 = 'import os.path'
    var_10 = ' osp'
    var_11 = [var_9, var_10]
    var_12 = ','
    var_13 = ','



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line that exceeds line length'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module \\\n    as alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_module_name.function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'module.long_module_name.\\\n    function'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name,  # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import long_module_name # noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport module.long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'cimport module.\\\n    long_module_name'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import very_long_function_name, another_function'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'line with comment # comment'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'line with noqa comment # NOQA'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'line with parentheses (comment)'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'line with trailing comma,'
    var_1 = '\n'
    var_2 = 20
    var_3 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 6/8 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 8/9 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/7 statements.
# Partially parsed test_import_statement_single_line. Retrieved 7/8 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 12/20 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 9/11 statements.
# Partially parsed test_import_statement_custom_indent. Retrieved 9/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment'
    var_6 = [var_5]
    var_7 = module_0.import_statement(var_0, var_4, var_6)
    var_8 = '# comment'
    var_9 = bool('# comment' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = '\n'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = 100
    var_4 = 'wrap_length'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.import_statement(var_0, var_2, config=var_6)
    var_8 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = '\n'
    var_13 = -1
    var_14 = -1

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' not in var_11)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = 'from module import\n    '

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '# '
    var_1 = 'comment_prefix'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = 'comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' in var_11)
    assert var_13 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_import_statement_custom_config. Retrieved 10/11 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 12/18 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_indent. Retrieved 8/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from module import (\n    a,\n    b,\n    c,\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import (a, b, c)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment'
    var_6 = [var_5]
    var_7 = module_0.import_statement(var_0, var_4, var_6)
    var_8 = '# comment'
    var_9 = bool('# comment' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = '\n'
    var_13 = -1
    var_14 = -1

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'
    var_5 = bool('\n' not in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' not in var_11)
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '    '
    var_2 = 88
    var_3 = None
    var_4 = ' # '
    var_5 = 'from module import something as alias'
    var_6 = '\n'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 8/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = len(var_4)
    var_7 = 2
    var_8 = var_6 + var_7
    var_9 = bool(var_8 > (var_1.wrap_length or var_1.line_length))
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_import_statement_single_line. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'from module import'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = 100
    var_5 = True
    var_6 = '\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = len(var_4)
    var_6 = 2
    var_7 = var_5 + var_6
    var_8 = bool(var_7 > (var_1.wrap_length or var_1.line_length))
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = '# '
    var_2 = 'a'
    var_3 = 11
    var_4 = var_2 * var_3
    var_5 = '\n'
    var_6 = '# NOQA'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_import_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_as_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_parentheses_and_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'comment'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15
    var_3 = 'as'

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = ','



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 13/26 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'module1'
    var_8 = 'module2'
    var_9 = 'module3'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'from . import'
    var_12 = module_1.import_statement(var_11, var_10, config=var_6)
    var_13 = '\n'
    var_14 = -1
    var_15 = 0
    var_16 = -1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line that exceeds line length'
    var_1 = '\n'
    var_2 = 10

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'from module import \\\n    long_function_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'from module import (\n    long_function_name,\n    another_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name # noqa'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'from module import (\n    long_function_name,  # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import module as long_alias_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import module as long_alias_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'module.\\\n    long_function_name'

def test_case_0():
    var_0 = 'from module import long_function_name, another_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'from module import long_function_name, another_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = 50
    var_4 = len(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_line_65_predicate_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import (something, something_else,  # comment'
    var_6 = '\n'
    var_7 = -1
    var_8 = ')'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_17. Retrieved 9/18 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import function, another_function'
    var_6 = ','
    var_7 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_56_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'import os  # some comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = '# '
    var_5 = -1
    var_6 = var_0.split(var_1)[var_5]
    var_7 = -1
    var_8 = var_0.split(var_1)[var_7]
    var_9 = ')'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = '# '
    var_6 = len(var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line that exceeds line length'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module cimport long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module cimport \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_module_name.function()'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'module.long_module_name.\\\n    function()'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as long_alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module as \\\n    long_alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (long_module_name,)'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import (\n    long_module_name,  # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import (\n    long_module_name,  # noqa\n)'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as long_alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module as long_alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # noqa: F401'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import (\n    long_module_name,  # noqa: F401\n)'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_import_statement_explode. Retrieved 8/9 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/7 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 7/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = 'a'
    var_8 = bool('a' in var_6)
    assert var_8 is True
    var_9 = 'b'
    var_10 = bool('b' in var_6)
    assert var_10 is True
    var_11 = 'c'
    var_12 = bool('c' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'a'
    var_8 = bool('a' in var_6)
    assert var_8 is True
    var_9 = 'b'
    var_10 = bool('b' in var_6)
    assert var_10 is True
    var_11 = 'c'
    var_12 = bool('c' in var_6)
    assert var_12 is True
    var_13 = '\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment'
    var_6 = [var_5]
    var_7 = module_0.import_statement(var_0, var_4, var_6)
    var_8 = '# comment'
    var_9 = bool('# comment' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import a'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    assert var_3 == 'from module import a'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = 'a'
    var_13 = bool('a' in var_11)
    assert var_13 is True
    var_14 = 'b'
    var_15 = bool('b' in var_11)
    assert var_15 is True
    var_16 = 'c'
    var_17 = bool('c' in var_11)
    assert var_17 is True

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = '    '
    var_11 = bool('    ' in var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment'
    var_13 = bool('# comment' not in var_11)
    assert var_13 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_line_predicate_evaluates_to_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_line_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_line_no_wrapping_needed. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_import. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_noqa. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_vertical_hanging_indent. Retrieved 2/5 statements.
# Partially parsed test_line_wrapping_with_vertical_grid_grouped. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = 'from module import ('
    var_3 = 'long_function_name'

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = 'module.'
    var_3 = 'long_function_name'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 'import module as alias'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = '# comment'

def test_case_0():
    var_0 = 'long_line # NOQA'
    var_1 = '\n'
    var_2 = 'long_line # NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    var_7 = bool('(' in var_6 and ')' in var_6)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = True
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = {var_3: var_2, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.line(var_0, var_1, var_6)
    var_8 = ','
    var_9 = bool(',' in var_7)
    assert var_9 is True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = None
    var_6 = False
    var_7 = '#'
    var_8 = '    '
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = False
    var_4 = '#'
    var_5 = ''
    var_6 = len(var_0)
    var_7 = 2
    var_8 = var_6 + var_7
    var_9 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 10
    var_3 = 5
    var_4 = len(var_0)
    var_5 = 2
    var_6 = var_4 + var_5



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line that exceeds line length'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import very_long_name'
    var_7 = 'from module import \\\n    very_long_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.very_long_function_name()'
    var_7 = 'module.\\\n    very_long_function_name()'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as very_long_alias'
    var_7 = 'import module as \\\n    very_long_alias'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (very_long_name)'
    var_8 = 'from module import (\n    very_long_name,\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import very_long_name # some comment'
    var_7 = 'from module import \\\n    very_long_name # some comment'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import very_long_name # NOQA: some comment'
    var_9 = 'from module import (\n    very_long_name, # NOQA: some comment\n)'
    var_10 = '\n'
    var_11 = module_1.line(var_8, var_10, var_7)
    var_12 = bool(var_11 == var_9)
    assert var_12 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import very_long_name'
    var_3 = 'from module import (\n    very_long_name,\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import very_long_name'
    var_3 = 'from module import (\n    very_long_name,\n)'
    var_4 = '\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = '    '
    var_3 = 88
    var_4 = None
    var_5 = 'from module import something, another_thing, yet_another'
    var_6 = '\n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_with_noqa_comment_and_noqa_mode. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 10
    var_1 = '# '
    var_2 = 'shortline'
    var_3 = '\n'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_line_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_import_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_as_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_parentheses_and_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_dot_split. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_line_with_noqa_mode_and_no_noqa_comment. Retrieved 5/8 statements.
# Partially parsed test_line_with_noqa_mode_and_noqa_comment. Retrieved 7/10 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = ' # NOQA'
    var_4 = var_2 + var_3
    var_5 = '\n'
    var_6 = 50

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import function, other_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'from module import (\n    function,\n    other_function\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'module.function.other_function'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'module.function(\n    .other_function\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import module\n    as alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'long_line  # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'long_line(\n    # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'long_line(\n    # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'long_line(\n    ,\n)'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 14/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = None
    var_3 = '    '
    var_4 = '# '
    var_5 = 'import os.path as osp'
    var_6 = '\n'
    var_7 = None
    var_8 = 'as '
    var_9 = 'import os.path '
    var_10 = ' osp'
    var_11 = [var_9, var_10]
    var_12 = ','
    assert var_12 == ','



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = None
    var_3 = '    '
    var_4 = '# '
    var_5 = 'from module import long_module_name, another_long_module_name'
    var_6 = '\n'
    var_7 = ', '



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = len(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import (something, something_else, # noqa\n)'
    var_6 = '\n'
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_59. Retrieved 6/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'noqa'
    var_3 = var_1.include_trailing_comma
    var_4 = ','
    var_5 = ''
    var_6 = var_4 if var_3 else var_5
    assert var_6 == ','



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 8/13 statements.


def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = None
    var_3 = '    '
    var_4 = '# '
    var_5 = 'import os.path as osp, sys'
    var_6 = ','



# Parsed testcases at query #19
#--------------------------




import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'invalid_name'
    var_1 = module_0.formatter_from_string(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_wrapping_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa_in_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_comment_prefix. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_wrap_length. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa_mode_and_noqa_comment. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'module.very_long_function_name'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module  # some comment'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module  # NOQA'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30
    var_3 = True

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30
    var_3 = True

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = '    import module1, module2'
    var_1 = '\n'
    var_2 = 30
    var_3 = '    '

def test_case_0():
    var_0 = 'import module  # some comment'
    var_1 = '\n'
    var_2 = 30
    var_3 = '# '

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30
    var_3 = 20

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module1, module2  # NOQA'
    var_1 = '\n'
    var_2 = 30



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_line_noqa_mode_no_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 6/9 statements.
# Partially parsed test_line_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = ' # NOQA'
    var_4 = var_2 + var_3
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module cimport long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module cimport \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_module_name.function()'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'module.long_module_name.\\\n    function()'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as long_alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module as \\\n    long_alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'import (\n    module,  # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'import module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module1, module2'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module1, module2'
    var_3 = '\n'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = None
    var_3 = '    '
    var_4 = '#'
    var_5 = 'from module import something, another_thing as alias'
    var_6 = '\n'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = 30
    var_2 = '\n'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = 10
    var_2 = '\n'
    var_3 = '# comment'

def test_case_0():
    var_0 = 'long_line # NOQA'
    var_1 = 10
    var_2 = '\n'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = 15
    var_2 = '\n'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = 10
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = 10
    var_2 = True
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = 10
    var_2 = '\n'

def test_case_0():
    var_0 = 'cimport module.long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = 15
    var_2 = '\n'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 90
    var_4 = var_2 * var_3
    var_5 = len(var_4)
    var_6 = 2
    var_7 = var_5 + var_6
    var_8 = bool(var_7 > (var_1.wrap_length or var_1.line_length))
    assert var_8 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'some_content'
    var_1 = '\n'
    var_2 = '#'
    var_3 = 100
    var_4 = None
    var_5 = True
    var_6 = '    '
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_line_no_wrapping_needed. Retrieved 2/4 statements.
# Partially parsed test_line_wrapping_with_import_splitter. Retrieved 3/5 statements.
# Partially parsed test_line_wrapping_with_dot_splitter. Retrieved 3/5 statements.
# Partially parsed test_line_wrapping_with_as_splitter. Retrieved 3/5 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/5 statements.
# Partially parsed test_line_wrapping_with_noqa_comment. Retrieved 3/5 statements.
# Partially parsed test_line_wrapping_with_config_multi_line_output. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_config_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_config_noqa_mode_and_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_config_noqa_mode_and_no_noqa_comment. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'

def test_case_0():
    var_0 = 'module.long_function_name.another_function_name'
    var_1 = '\n'
    var_2 = 'module.long_function_name.\n    another_function_name'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 'import module as alias'

def test_case_0():
    var_0 = 'import module  # comment'
    var_1 = '\n'
    var_2 = 'import module  # comment'

def test_case_0():
    var_0 = 'import module  # noqa'
    var_1 = '\n'
    var_2 = 'import module  # noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'include_trailing_comma'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'wrap_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import module  # comment'
    var_1 = '\n'
    var_2 = '# '
    var_3 = 'comment_prefix'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module  # comment'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = '    '
    var_3 = 'indent'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'line_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (\n    long_function_name,\n    another_function_name,\n)'
    var_7 = module_1.line(var_0, var_1, var_5)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = 'from module import long_function_name, another_function_name  # NOQA'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name  # NOQA'
    var_1 = '\n'
    var_2 = 'from module import long_function_name, another_function_name  # NOQA'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name  # comment'
    var_1 = '\n'
    var_2 = 'from module import long_function_name, another_function_name  # comment  # NOQA'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_in_config. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'module.very_long_attribute_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import module as very_long_alias'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import module # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import module # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import module'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'import module1, module2'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = True
    var_3 = 'import module1, module2'
    var_4 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'cimport very_long_module_name'
    var_2 = '\n'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 12/14 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = False
    var_3 = '# '
    var_4 = '    '
    var_5 = 'a'
    var_6 = 101
    var_7 = var_5 * var_6
    var_8 = '\n'
    var_9 = len(var_7)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_46_evaluates_to_true. Retrieved 5/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = None
    var_3 = var_1.include_trailing_comma
    var_4 = ','
    var_5 = ''



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_balanced_wrapping_predicate. Retrieved 13/22 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = False
    var_3 = 'balanced_wrapping'
    var_4 = 'wrap_length'
    var_5 = 'line_length'
    var_6 = 'include_trailing_comma'
    var_7 = 'ignore_comments'
    var_8 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_0, var_7: var_2}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import ('
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_1.import_statement(var_10, var_14, config=var_9)
    var_16 = '\n'
    var_17 = -1
    var_18 = -1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_while_loop_condition. Retrieved 8/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (a, b, c, d, e, f)'
    var_8 = '\n'
    var_9 = -1
    var_10 = 100
    var_11 = -1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_function_name'
    var_2 = '\n'
    var_3 = 'from module import (\n    long_function_name\n)'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import func # comment'
    var_2 = '\n'
    var_3 = 'from module import (\n    func  # comment\n)'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_function_name # NOQA'
    var_2 = '\n'
    var_3 = 'from module import long_function_name # NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = '\n'
    var_3 = 'import module as alias'

def test_case_0():
    var_0 = 20
    var_1 = 'module.long_function_name()'
    var_2 = '\n'
    var_3 = 'module.long_function_name()'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import func1, func2'
    var_3 = '\n'
    var_4 = 'from module import (\n    func1,\n    func2,\n)'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import func1, func2'
    var_3 = '\n'
    var_4 = 'from module import (\n    func1,\n    func2,\n)'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import func # noqa'
    var_2 = '\n'
    var_3 = 'from module import (\n    func  # noqa\n)'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import func1, func2'
    var_2 = '\n'
    var_3 = 'from module import (\n    func1,\n    func2,\n)'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_line_no_wrapping_needed. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_with_existing_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = 100

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,\n    another_function,\n)'

def test_case_0():
    var_0 = 'from module import function  # comment'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,  # comment\n)'

def test_case_0():
    var_0 = 'from module import function  # noqa'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,  # noqa\n)'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'import module as alias'

def test_case_0():
    var_0 = 'module.submodule.function'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'module.submodule.function'

def test_case_0():
    var_0 = 'from module import function'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'from module import function  # NOQA'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'cimport module.function'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'cimport module.function'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'from module import function,\\n    another_function'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,\n    another_function,\n)'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = False
    var_5 = 'from module import (\n    function\n    another_function\n)'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,\n    another_function,\n)'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'from module import (\n    function,\n    another_function,\n)'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_balanced_wrapping_with_multiple_lines. Retrieved 12/19 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = 'line_length'
    var_5 = 'include_trailing_comma'
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_1, var_5: var_0}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import ('
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_1.import_statement(var_8, var_12, config=var_7)
    var_14 = '\n'
    var_15 = -1
    var_16 = -1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 20
    var_1 = 'import very.long.module.name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport very.long.module.name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'very.long.module.name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = '\n'
    var_3 = 'import module as alias'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # comment'
    var_2 = '\n'
    var_3 = '# comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # noqa'
    var_2 = '\n'
    var_3 = '# noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_comment_added. Retrieved 4/8 statements.
# Partially parsed test_line_noqa_comment_not_added. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_without_trailing_comma. Retrieved 5/9 statements.
# Partially parsed test_line_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_without_parentheses. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_function_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as m'
    var_2 = '\n'
    var_3 = 'as ('

def test_case_0():
    var_0 = 20
    var_1 = 'module.long_function_name'
    var_2 = '\n'
    var_3 = '.('

def test_case_0():
    var_0 = 20
    var_1 = 'cimport module.long_function_name'
    var_2 = '\n'
    var_3 = 'cimport ('

def test_case_0():
    var_0 = 10
    var_1 = 'very long line without noqa'
    var_2 = '\n'
    var_3 = ' NOQA'

def test_case_0():
    var_0 = 10
    var_1 = 'very long line # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # comment'
    var_2 = '\n'
    var_3 = '# comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # noqa'
    var_2 = '\n'
    var_3 = 'noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module'
    var_3 = '\n'
    var_4 = '\\'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 17/35 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = ()
    var_8 = '\n'
    var_9 = None
    var_10 = False
    var_11 = module_1.import_statement(var_2, var_6, var_7, var_8, var_1, var_9, var_10)
    var_12 = -1
    var_13 = var_1.wrap_length
    var_14 = var_1.line_length
    var_15 = var_13 or var_14
    var_16 = -1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_36. Retrieved 9/18 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import ('
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = False
    var_8 = module_1.import_statement(var_2, var_6, config=var_1, explode=var_7)
    var_9 = '\n'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_46_evaluates_to_true. Retrieved 5/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some comment'
    var_3 = var_1.include_trailing_comma
    var_4 = ','
    var_5 = ''



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = len(var_2)
    var_4 = len(var_2)
    var_5 = bool(var_4 > 1)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_line_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_with_parentheses_and_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import ('

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'comment'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15
    var_3 = 'as ('

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'noqa'
    var_5 = '('

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = ','

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_46_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import (long_function_name1, long_function_name2)'
    var_6 = '\n'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 2/5 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import (\n    long_function_name\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'cimport module.long_function_name'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'cimport module.(\n    long_function_name\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'module.(\n    long_function_name\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import module as alias'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import long_function_name  # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import (\n    long_function_name  # comment\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import long_function_name  # noqa'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from module import long_function_name  # noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = 'use_parentheses'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'from module import long_function_name'
    var_6 = '\n'
    var_7 = module_1.line(var_5, var_6, var_4)
    assert var_7 == 'from module import (\n    long_function_name,\n)'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'



# Parsed testcases at query #45
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 10
    var_3 = 5
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = len(var_0)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_7.wrap_length
    var_12 = var_7.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    var_15 = bool(not var_14)
    assert var_15 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 6/7 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 8/9 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/8 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 8/9 statements.
# Partially parsed test_import_statement_single_line. Retrieved 5/6 statements.
# Partially parsed test_import_statement_with_trailing_comma. Retrieved 9/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 'from module import'
    var_7 = bool('from module import' in var_5)
    assert var_7 is True
    var_8 = 'a'
    var_9 = bool('a' in var_5)
    assert var_9 is True
    var_10 = 'b'
    var_11 = bool('b' in var_5)
    assert var_11 is True
    var_12 = 'c'
    var_13 = bool('c' in var_5)
    assert var_13 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment1'
    var_6 = '# comment2'
    var_7 = [var_5, var_6]
    var_8 = module_0.import_statement(var_0, var_4, var_7)
    var_9 = '# comment1'
    var_10 = bool('# comment1' in var_8)
    assert var_10 is True
    var_11 = '# comment2'
    var_12 = bool('# comment2' in var_8)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = '\n'

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# comment1'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# comment1'
    var_13 = bool('# comment1' not in var_11)
    assert var_13 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_36. Retrieved 7/19 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import ('
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = '\n'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_while_loop_predicate_evaluates_to_true. Retrieved 14/27 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = False
    var_3 = 'balanced_wrapping'
    var_4 = 'wrap_length'
    var_5 = 'line_length'
    var_6 = 'ignore_comments'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import ('
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = module_1.import_statement(var_9, var_13, config=var_8)
    var_15 = '\n'
    var_16 = -1
    var_17 = 20
    var_18 = -1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = len(var_4)
    var_6 = var_1.line_length
    var_7 = var_5 > var_6
    var_8 = bool(var_7 and (var_1.wrap_length or var_1.line_length) == 80)
    assert var_8 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/8 statements.
# Partially parsed test_line_wrap_noqa_mode. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_noqa_mode_with_noqa_comment. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 3/7 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 3/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import long_function_name'
    var_2 = '\n'
    var_3 = 'import'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport module.long_function_name'
    var_2 = '\n'
    var_3 = 'cimport'

def test_case_0():
    var_0 = 20
    var_1 = 'module.long_function_name'
    var_2 = '\n'
    var_3 = '.'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = '\n'
    var_3 = 'as'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # comment'
    var_2 = '\n'
    var_3 = '# comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # noqa'
    var_2 = '\n'
    var_3 = '# noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'import os, sys  # NOQA'
    var_1 = '\n'
    var_2 = 10
    var_3 = None
    var_4 = True
    var_5 = '#'
    var_6 = '    '
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = True
    var_3 = '#'
    var_4 = '    '
    var_5 = 'from module import very_long_module_name, another_very_long_module_name, yet_another_very_long_module_name'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_line_predicate_true. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_line_wrapping_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_trailing_comma. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = 10
    var_2 = '\n'

def test_case_0():
    var_0 = 'long_line # NOQA'
    var_1 = 10
    var_2 = '\n'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'cimport module.long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'long_line'
    var_1 = 10
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'long_line'
    var_1 = 10
    var_2 = True
    var_3 = '\n'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_predicate_at_line_59_evaluates_to_true. Retrieved 8/12 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import (something, another_thing, # noqa: F401'
    var_6 = '\n'
    var_7 = '# noqa: F401'
    var_8 = ',)'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_line_noqa_mode_no_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_import_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_cimport_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_dot_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_as_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_no_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_noqa_comment_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_with_indent. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'very long line that exceeds the line length limit'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'very long line that exceeds the line length limit # NOQA'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'module.very_long_function_name()'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module # some comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = False

def test_case_0():
    var_0 = 'import module # some comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = '    import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = '    '



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = 20
    var_3 = None
    var_4 = True
    var_5 = '#'
    var_6 = '    '
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 7/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = len(var_2)
    var_6 = 2
    var_7 = var_5 + var_6
    var_8 = bool(var_7 > (var_4.wrap_length or var_4.line_length))
    assert var_8 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = ''
    var_5 = '#'
    var_6 = False
    var_7 = len(var_0)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_line_length_greater_than_10. Retrieved 12/22 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import ('
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = module_1.import_statement(var_7, var_11, config=var_6)
    var_13 = '\n'
    var_14 = -1
    var_15 = -1



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '# '
    var_2 = '    '
    var_3 = 88
    var_4 = None
    var_5 = 'from module import something, another_thing, yet_another_thing'
    var_6 = '\n'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_line_noqa_mode_with_no_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 10
    var_1 = 'long line without noqa'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = 'from module import \\\n    long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name'
    var_8 = 'from module import (\n    long_module_name,\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # comment'
    var_7 = 'from module import \\\n    long_module_name # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = 'from module import (\n    long_module_name,\n) # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as long_alias'
    var_7 = 'import module as long_alias'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_module_name'
    var_7 = 'module.\\\n    long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = 'from module import (\n    long_module_name,\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = 'from module import (\n    long_module_name,\n)'
    var_4 = '\n'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 12/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = None
    var_6 = False
    var_7 = '# '
    var_8 = '    '
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_line_with_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport module.long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'cimport module.\\\n    long_module_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.long_module_name.function()'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'module.\\\n    long_module_name.function()'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module \\\n    as alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name, # noqa\n)'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_module_name'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import long_module_name # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    long_module_name # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name, # comment\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import long_module_name # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import (\n    long_module_name, # noqa\n)'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_comma_maybe_predicate_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'from module import function, other_function'
    var_6 = '\n'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = '#'
    var_3 = '    '
    var_4 = None
    var_5 = 'import some_module, another_module  # some comment'
    var_6 = ','
    var_7 = ''



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 5/11 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some content without trailing comma'
    var_3 = var_1.include_trailing_comma
    var_4 = var_1.use_parentheses
    var_5 = ','



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_line_17_predicate_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = '    '
    var_4 = None
    var_5 = 'import os.path as osp'
    var_6 = '\n'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_17. Retrieved 9/14 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'import os.path as osp, sys'
    var_6 = None
    var_7 = ','



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_line_with_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'very long line that exceeds line length'
    var_1 = '\n'

def test_case_0():
    var_0 = 'very long line that exceeds line length # NOQA'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very.long.module.name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import very.long.\\n    module.name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport very.long.module.name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'cimport very.long.\\n    module.name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'very.long.module.name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'very.long.\\n    module.name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module\\n    as alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import very.long.module.name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'import (\\n    very.long.module.name,\\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'import (\\n    module,  # comment\\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'import (\\n    module\\n) # noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very.long.module.name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very.long.module.name'
    var_3 = '\n'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 8/13 statements.


def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = None
    var_3 = '    '
    var_4 = '# '
    var_5 = 'from module import something, another_thing  # comment'
    var_6 = 'from module import something, another_thing'
    var_7 = ','



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 13/34 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = ()
    var_8 = module_1.import_statement(var_2, var_6, config=var_1)
    var_9 = '\n'
    var_10 = 1
    var_11 = -1
    var_12 = 0
    var_13 = -1



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_vertical_hanging_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_vertical_grid_grouped. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'very long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'very long line'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import very_long_name'
    var_7 = 'from module import \\\n    very_long_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.very_long_function_name()'
    var_7 = 'module.\\\n    very_long_function_name()'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as very_long_alias'
    var_7 = 'import module as \\\n    very_long_alias'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (item1, item2, item3)'
    var_8 = 'from module import (\n    item1,\n    item2,\n    item3,\n)'
    var_9 = '\n'
    var_10 = module_1.line(var_7, var_9, var_6)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import item # some comment'
    var_9 = 'from module import (\n    item,  # some comment\n)'
    var_10 = '\n'
    var_11 = module_1.line(var_8, var_10, var_7)
    var_12 = bool(var_11 == var_9)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import item # noqa: F401'
    var_9 = 'from module import item # noqa: F401'
    var_10 = '\n'
    var_11 = module_1.line(var_8, var_10, var_7)
    var_12 = bool(var_11 == var_9)
    assert var_12 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import (item1, item2, item3)'
    var_3 = 'from module import (\n    item1,\n    item2,\n    item3,\n)'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import (item1, item2, item3)'
    var_3 = 'from module import (\n    item1,\n    item2,\n    item3,\n)'
    var_4 = '\n'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_line_71_predicate_evaluates_to_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = '# '
    var_6 = len(var_2)
    var_7 = '# NOQA'
    var_8 = bool('# NOQA' not in var_2)
    assert var_8 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_other_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa. Retrieved 3/6 statements.
# Failed to parse test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name as lmn'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long_module_name.function_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'cimport long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'import long_module_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'import long_module_name'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'import long_module_name # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA NOQA'
    var_2 = '\n'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line that exceeds line length'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'from module import \\\n    function'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import module as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module \\\n    as alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import function'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    assert var_9 == 'from module import(\n    function,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    assert var_10 == 'from module import(\n    function,  # noqa\n)'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import function'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function # comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    assert var_10 == 'from module import \\\n    function  # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport module.function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'cimport module.\\\n    function'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.function(argument)'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'module.\\\n    function(argument)'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_import_statement_multi_line_output. Retrieved 5/7 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 11/17 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 9/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import A, B, C'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# Comment'
    var_6 = [var_5]
    var_7 = module_0.import_statement(var_0, var_4, var_6)
    var_8 = '# Comment'
    var_9 = bool('# Comment' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from module import (\n    A,\n    B,\n    C,\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'wrap_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'C'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = 0
    var_11 = '\n'
    var_12 = var_9.split(var_11)[var_10]
    var_13 = len(var_12)
    var_14 = bool(var_13 <= 20)
    assert var_14 is True

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'C'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = '\n'
    var_11 = -1
    var_12 = -1

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'A'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'
    var_5 = bool('\n' not in var_3)
    assert var_5 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'C'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'ignore_comments'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'C'
    var_8 = [var_5, var_6, var_7]
    var_9 = '# Comment'
    var_10 = [var_9]
    var_11 = module_1.import_statement(var_4, var_8, var_10, config=var_3)
    var_12 = '# Comment'
    var_13 = bool('# Comment' not in var_11)
    assert var_13 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 6/7 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 10/11 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/7 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 13/18 statements.
# Partially parsed test_import_statement_single_line_output. Retrieved 5/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    var_6 = 'from module import'
    var_7 = bool('from module import' in var_5)
    assert var_7 is True
    var_8 = 'a'
    var_9 = bool('a' in var_5)
    assert var_9 is True
    var_10 = 'b'
    var_11 = bool('b' in var_5)
    assert var_11 is True
    var_12 = 'c'
    var_13 = bool('c' in var_5)
    assert var_13 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment 1'
    var_6 = '# comment 2'
    var_7 = [var_5, var_6]
    var_8 = module_0.import_statement(var_0, var_4, var_7)
    var_9 = '# comment 1'
    var_10 = bool('# comment 1' in var_8)
    assert var_10 is True
    var_11 = '# comment 2'
    var_12 = bool('# comment 2' in var_8)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = '\r\n'
    var_8 = bool('\r\n' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = '\n'
    var_8 = [var_1, var_2, var_3]
    var_9 = len(var_8)

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    var_10 = '\n'
    var_11 = -1
    var_12 = len(var_4)
    var_13 = -1
    var_14 = min(var_8)
    var_15 = bool(var_12 >= var_14)
    assert var_15 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 50
    var_2 = 'indent'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = '    '
    var_13 = bool('    ' in var_11)
    assert var_13 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)
    var_3 = 'from module import'
    var_4 = bool('from module import' in var_2)
    assert var_4 is True



