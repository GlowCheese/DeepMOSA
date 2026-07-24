####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'cimport module.long_function_name, another_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'module.long_function_name.another_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name  # comment'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name  # noqa'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'use_parentheses'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'
    var_2 = True
    var_3 = 'include_trailing_comma'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)
    assert var_6 == 'from module import (\n    long_function_name,\n    another_function_name,\n)'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'

def test_case_0():
    var_0 = 'from module import long_function_name, another_function_name'
    var_1 = '\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.


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
    var_3 = 'from module import (\n    long_function_name,\n    another_function,\n)'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line,  # comment'

def test_case_0():
    var_0 = 'very_long_line # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15
    var_3 = 'import module as (\n    alias,\n)'

def test_case_0():
    var_0 = 'cimport module.long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'cimport module.(\n    long_function_name,\n)'

def test_case_0():
    var_0 = 'module.long_function_name.another_function'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'module.(\n    long_function_name.another_function,\n)'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'long_line,  # comment'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = 'long_line,'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line\n'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 8/13 statements.


def test_case_0():
    var_0 = True
    var_1 = 88
    var_2 = None
    var_3 = '    '
    var_4 = '# '
    var_5 = 'from module import something, another'
    var_6 = 'from module import something, another'
    var_7 = ','



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 6/7 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 8/9 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/8 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 8/9 statements.
# Partially parsed test_import_statement_single_line. Retrieved 5/6 statements.


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
    var_3 = [var_1, var_2]
    var_4 = '# comment 1'
    var_5 = '# comment 2'
    var_6 = [var_4, var_5]
    var_7 = module_0.import_statement(var_0, var_3, var_6)
    var_8 = '# comment 1'
    var_9 = bool('# comment 1' in var_7)
    assert var_9 is True
    var_10 = '# comment 2'
    var_11 = bool('# comment 2' in var_7)
    assert var_11 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = '\r\n'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = '\r\n'
    var_7 = bool('\r\n' in var_5)
    assert var_7 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_re_search_predicate_true. Retrieved 7/9 statements.


import re as module_0

def test_case_0():
    var_0 = 'import os.path as osp'
    var_1 = 'as '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_wrap_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import long_function_name, another_long_function_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'from module import (\n    long_function_name,\n    another_long_function_name,\n)'

def test_case_0():
    var_0 = 'object.very_long_method_name(arg1, arg2)'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'object.\n    very_long_method_name(\n        arg1,\n        arg2,\n    )'

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module\n    as very_long_alias'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line,  # comment\n'

def test_case_0():
    var_0 = 'long_line # noqa'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line # noqa'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = '(\n    long_line,\n)'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = '(\n    long_line,\n)'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line,\n'

def test_case_0():
    var_0 = 'long_line'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'long_line,\n'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.


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
    var_1 = 'import long_module # some comment'
    var_2 = '\n'
    var_3 = '# some comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import very_long_module_name # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import long_module as lm'
    var_2 = '\n'
    var_3 = 'as lm'

def test_case_0():
    var_0 = 20
    var_1 = True
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
    var_1 = 'cimport very_long_module_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.very_long_function_name()'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very_long_module_name'
    var_3 = '\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_line_30_predicate_evaluates_to_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = True
    var_3 = '#'
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa. Retrieved 4/8 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid_grouped_and_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_hanging_indent_and_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped_and_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent_and_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid_and_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid_grouped_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_hanging_indent_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_horizontal_grid_and_noqa. Retrieved 3/6 statements.


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
    var_1 = 'short line # comment'
    var_2 = '\n'
    var_3 = 'short line # comment'

def test_case_0():
    var_0 = 20
    var_1 = 'very long line that exceeds line length'
    var_2 = '\n'
    var_3 = ' NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as alias'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.long_function_name()'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport module.long_function_name'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module, other_module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module, other_module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # noqa'
    var_3 = '\n'
    var_4 = 'import module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module'
    var_2 = '\n'
    var_3 = 'import module, other_module'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module'
    var_2 = '\n'
    var_3 = 'import module, other_module'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module'
    var_2 = '\n'
    var_3 = 'import module, other_module'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # comment'
    var_2 = '\n'
    var_3 = 'import module, other_module # comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # comment'
    var_2 = '\n'
    var_3 = 'import module, other_module # comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # comment'
    var_2 = '\n'
    var_3 = 'import module, other_module # comment'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # noqa'
    var_2 = '\n'
    var_3 = 'import module, other_module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # noqa'
    var_2 = '\n'
    var_3 = 'import module, other_module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # noqa'
    var_2 = '\n'
    var_3 = 'import module, other_module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # noqa'
    var_2 = '\n'
    var_3 = 'import module, other_module # noqa'

def test_case_0():
    var_0 = 20
    var_1 = 'import module, other_module # noqa'
    var_2 = '\n'
    var_3 = 'import module, other_module # noqa'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_false. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'import os  # noqa'
    var_1 = '\n'
    var_2 = True
    var_3 = '# '
    var_4 = '#'
    var_5 = '\\bimport \\b'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = False
    var_5 = '#'
    var_6 = '    '
    var_7 = len(var_0)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_without_trailing_comma. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the length limit # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'long line that exceeds the length limit'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'import module # some comment'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'import module # noqa'
    var_1 = '\n'
    var_2 = 15

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 15
    var_3 = True

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 15
    var_3 = False
    var_4 = True



# Parsed testcases at query #16
#--------------------------




import isort.wrap_modes as module_0

def test_case_0():
    var_0 = 'invalid_name'
    var_1 = module_0.formatter_from_string(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_no_wrap_with_noqa_mode. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_import_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_dot_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment_and_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses_and_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses_and_no_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_parentheses_and_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_parentheses_and_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_no_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_no_parentheses_and_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_no_parentheses_and_noqa_comment. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short content'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short content'

def test_case_0():
    var_0 = 'short content'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long content that exceeds line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something, another'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport module.something, module.another'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.something.another'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module as m'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'long content # comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'long content # noqa comment'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something, another'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'from module import something, another'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something, another # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something, another # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something, another'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something, another'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import something, another'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import something, another # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import something, another # noqa'
    var_3 = '\n'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_present. Retrieved 3/6 statements.


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

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import foo # some comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('(' in var_8 and ')' in var_8 and ('# some comment' in var_8))
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import foo as bar'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('(' in var_8 and ')' in var_8 and ('as bar' in var_8))
    assert var_9 is True

def test_case_0():
    var_0 = 20
    var_1 = 'cimport very.long.module.name'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'very.long.module.name.function()'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(',' in var_8 and '\n' in var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 20
    var_1 = 'import very.long.module.name'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'import very.long.module.name # NOQA'
    var_2 = '\n'

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
    var_7 = 'import foo, bar, baz'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = bool('(' in var_9 and ')' in var_9 and (',' in var_9))
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
    var_6 = 'import foo, bar, baz'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('\\' in var_8 and '\n' in var_8)
    assert var_9 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_use_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_include_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment_prefix. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_wrap_length. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_indent. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_use_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_include_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_comment_prefix. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_wrap_length. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment_and_indent. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'from module import (\n    long_function_name,\n)'

def test_case_0():
    var_0 = 'cimport module.long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'cimport module.long_function_name'

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'module.long_function_name'

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module as alias'

def test_case_0():
    var_0 = 'import module  # comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module  # comment'

def test_case_0():
    var_0 = 'import module  # noqa'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module  # noqa'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module # NOQA'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module  # comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = '# '
    var_4 = 'import module  # comment'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = 10
    var_4 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 20
    var_3 = '    '
    var_4 = 'import (\n    module,\n)'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = '# '
    var_4 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = 10
    var_4 = 'import module  # noqa: F401'

def test_case_0():
    var_0 = 'import module  # noqa: F401'
    var_1 = '\n'
    var_2 = 20
    var_3 = '    '
    var_4 = 'import module  # noqa: F401'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_predicate_evaluates_to_true. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 100
    var_1 = 80
    var_2 = True
    var_3 = '#'
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_false. Retrieved 14/25 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'balanced_wrapping'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = False
    var_12 = module_1.import_statement(var_6, var_10, config=var_5, explode=var_11)
    var_13 = '\n'
    var_14 = -1
    var_15 = -1
    var_16 = var_0 > var_0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_import_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_cimport_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_dot_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_as_split. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_and_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_and_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_backslash_continuation. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the line length # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 'long line that exceeds the line length'
    var_1 = 10
    var_2 = '\n'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'module.very_long_function_name'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = 20
    var_2 = '\n'

def test_case_0():
    var_0 = 'import module.very_long_function_name # noqa'
    var_1 = 20
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'import module.very_long_function_name # comment'
    var_1 = 20
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'import module.very_long_function_name'
    var_1 = 20
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'import module.very_long_function_name'
    var_1 = 20
    var_2 = False
    var_3 = '\n'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = 50
    var_4 = len(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_wrap_length_predicate_evaluates_to_true. Retrieved 13/19 statements.


def test_case_0():
    var_0 = 100
    var_1 = 88
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_6 = '\n'
    var_7 = 'import '
    var_8 = 'a_very_long_module_name_that_exceeds_the_line_length_limit'
    var_9 = [var_7, var_8]
    var_10 = len(var_5)
    var_11 = 2
    var_12 = var_10 + var_11



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_line_42_predicate_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 88
    var_3 = '# '
    var_4 = '    '
    var_5 = 'from module import something, another_thing, third_thing'
    var_6 = '\n'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_true. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'import os # some comment'
    var_1 = '\n'
    var_2 = True
    var_3 = '# '
    var_4 = 20
    var_5 = None
    var_6 = ''
    var_7 = '#'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_line_wrapping_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 4/7 statements.


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
    var_4 = 'import ('
    var_5 = 'long_function_name,'
    var_6 = 'another_function'

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10
    var_3 = True
    var_4 = '('
    var_5 = 'comment'

def test_case_0():
    var_0 = 'long_line # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 15
    var_3 = True
    var_4 = 'as ('
    var_5 = 'alias'

def test_case_0():
    var_0 = 'module.long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = '('
    var_5 = 'long_function_name'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_import_statement_multi_line_output. Retrieved 4/7 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 8/9 statements.


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
    var_4 = '\r\n'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = '\r\n'
    var_7 = bool('\r\n' in var_5)
    assert var_7 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    var_6 = 'from foo import (\n    bar,\n    baz,\n)'
    var_7 = bool('from foo import (\n    bar,\n    baz,\n)' == var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = 'from foo import (\n    bar,\n    baz,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from foo import'
    var_7 = 'bar'
    var_8 = 'baz'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'from foo import (\n    bar,\n    baz,\n)'
    var_12 = bool('from foo import (\n    bar,\n    baz,\n)' == var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 'wrap_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import'
    var_5 = 'bar'
    var_6 = 'baz'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    assert var_8 == 'from foo import bar, baz'

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



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_line_length_predicate. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 80
    var_1 = 70
    var_2 = True
    var_3 = '# '
    var_4 = '    '
    var_5 = 'import some_module, another_module, third_module, fourth_module'
    var_6 = '\n'
    var_7 = len(var_5)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 15/30 statements.


def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'
    var_6 = True
    var_7 = 100
    var_8 = False
    var_9 = '#'
    var_10 = '    '
    var_11 = -1
    var_12 = -1
    var_13 = 10
    var_14 = var_7 > var_13



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 10
    var_3 = None
    var_4 = True
    var_5 = '#'
    var_6 = '    '
    var_7 = len(var_0)
    var_8 = 2
    var_9 = var_7 + var_8
    var_10 = 'short'
    var_11 = [var_10]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 10/19 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'import os.path as osp'
    var_6 = 'import os.path as osp'
    var_7 = None
    var_8 = ','
    var_9 = ''



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = '#'
    var_6 = len(var_2)
    var_7 = '# NOQA'
    var_8 = bool('# NOQA' not in var_2)
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = bool(var_3.use_parentheses)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 18/25 statements.


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
    var_9 = 40
    var_10 = var_0 * var_9
    var_11 = 'b'
    var_12 = var_11 * var_9
    var_13 = [var_10, var_12]
    var_14 = '.'
    var_15 = len(var_2)
    var_16 = 2
    var_17 = var_15 + var_16



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_vertical_hanging_indent. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_grid_grouped. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the line length # NOQA'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'long line that exceeds the line length'
    var_1 = '\n'
    var_2 = 10

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'from module import \\\n    very_long_function_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'cimport module.\\\n    very_long_function_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'module.very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'module.\\\n    very_long_function_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'import module as \\\n    very_long_alias'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'from module import (\n    very_long_function_name,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name # noqa'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'from module import (\n    very_long_function_name, # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name # comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.line(var_0, var_1, var_7)
    assert var_8 == 'from module import \\\n    very_long_function_name  # comment'

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = 20
    var_3 = True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name # noqa'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.line(var_0, var_1, var_8)
    assert var_9 == 'from module import (\n    very_long_function_name, # noqa\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import very_long_function_name # comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = '# '
    var_5 = 'line_length'
    var_6 = 'use_parentheses'
    var_7 = 'include_trailing_comma'
    var_8 = 'comment_prefix'
    var_9 = {var_5: var_2, var_6: var_3, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = module_1.line(var_0, var_1, var_10)
    assert var_11 == 'from module import (\n    very_long_function_name, # comment\n)'



# Parsed testcases at query #39
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
    var_0 = 'long line that exceeds the line length limit # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line that exceeds the line length limit'
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
    var_6 = 'import module as long_alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import module \\\n    as long_alias'

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
    var_2 = '# '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import long_module_name # comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    assert var_10 == 'from module import (\n    long_module_name,  # comment\n)'

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
    var_8 = 'from module import long_module_name # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    assert var_10 == 'from module import (\n    long_module_name,\n)  # noqa'

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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_wrap_mode_predicate. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 100
    var_1 = None
    var_2 = ''
    var_3 = '# '
    var_4 = True
    var_5 = 'import os.path as osp'
    var_6 = '\n'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = 60
    var_6 = False
    var_7 = '# '
    var_8 = '    '
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 6/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test'
    var_3 = var_1.include_trailing_comma
    var_4 = var_1.use_parentheses
    var_5 = ','
    var_6 = ''



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_import_statement_multi_line_output. Retrieved 5/8 statements.
# Partially parsed test_import_statement_trailing_comma. Retrieved 8/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from x import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from x import a, b, c'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from x import'
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
    var_0 = 'from x import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    assert var_6 == 'from x import (\n    a,\n    b,\n    c,\n)'

def test_case_0():
    var_0 = 'from x import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from x import'
    var_7 = 'short'
    var_8 = 'longer_name'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    assert var_10 == 'from x import short,\n    longer_name'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 'wrap_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from x import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    assert var_8 == 'from x import a, b'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from x import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = ','

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from x import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    assert var_9 == 'from x import (\n    a,\n    b,\n    c,\n)'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from x import'
    var_1 = 'a'
    var_2 = 'b'
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
    var_4 = 'from x import'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = '# comment'
    var_9 = [var_8]
    var_10 = module_1.import_statement(var_4, var_7, var_9, config=var_3)
    var_11 = '# comment'
    var_12 = bool('# comment' not in var_10)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from x import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    assert var_4 == 'from x import a, b'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = len(var_0)
    var_5 = 2
    var_6 = var_4 + var_5
    var_7 = var_3.wrap_length
    var_8 = var_3.line_length
    var_9 = var_7 or var_8
    var_10 = var_6 > var_9
    var_11 = bool(not var_10)
    assert var_11 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_wrap. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 4/7 statements.


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
    var_3 = 'from module import (\n    very_long_function_name,\n    another_function\n)'

def test_case_0():
    var_0 = 'cimport module.very_long_function_name, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'cimport module.very_long_function_name,\n    another_function'

def test_case_0():
    var_0 = 'module.very_long_function_name.another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'module.very_long_function_name\n    .another_function'

def test_case_0():
    var_0 = 'import module as very_long_alias_name'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import module\n    as very_long_alias_name'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line # comment'

def test_case_0():
    var_0 = 'from module import very_long_function_name, another_function # noqa'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'from module import (\n    very_long_function_name,\n    another_function  # noqa\n)'

def test_case_0():
    var_0 = 'very_long_line_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'very_long_line_that_exceeds_line_length # NOQA'

def test_case_0():
    var_0 = 'very_long_line_that_exceeds_line_length # NOQA'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'very_long_line_that_exceeds_line_length # NOQA'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_wrapping_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_trailing_comma. Retrieved 6/9 statements.
# Partially parsed test_line_wrapping_with_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_vertical_grid_grouped. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_vertical_hanging_indent. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_cimport. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_no_wrapping_with_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_no_wrapping_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_custom_indent. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_custom_comment_prefix. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_custom_wrap_length. Retrieved 4/7 statements.
# Partially parsed test_line_wrapping_with_custom_line_separator. Retrieved 2/3 statements.


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
    var_1 = 'import module as alias'
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
    var_3 = '# noqa'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module1, module2'
    var_3 = '\n'
    var_4 = -2
    var_5 = result.split(var_3)[var_4]
    var_6 = ','
    var_7 = bool(',' in var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module1, module2'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module1, module2'
    var_2 = '\n'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module1, module2'
    var_2 = '\n'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'cimport module1, module2'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'module.submodule.function'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = 'import module # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = '    '
    var_2 = 'import module1, module2'
    var_3 = '\n'
    var_4 = '    '

def test_case_0():
    var_0 = 20
    var_1 = '# '
    var_2 = 'import module # comment'
    var_3 = '\n'
    var_4 = '# comment'

def test_case_0():
    var_0 = 20
    var_1 = 15
    var_2 = 'import module1, module2'
    var_3 = '\n'
    var_4 = '\n'

def test_case_0():
    var_0 = 'import module1, module2'
    var_1 = '\r\n'
    var_2 = '\r\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_import_statement_with_custom_config. Retrieved 6/9 statements.


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
    var_5 = module_0.import_statement(var_0, var_4)
    assert var_5 == 'from module import (a, b, c)\n'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import ('
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment 1'
    var_6 = '# comment 2'
    var_7 = [var_5, var_6]
    var_8 = module_0.import_statement(var_0, var_4, var_7)
    assert var_8 == 'from module import (a, b, c)\n# comment 1\n# comment 2\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import ('
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\r\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    assert var_6 == 'from module import (a, b, c)\r\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'balanced_wrapping'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import ('
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    assert var_11 == 'from module import (\n    a, b, c\n)\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 'wrap_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import ('
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)
    assert var_9 == 'from module import (a, b, c)\n'

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
    var_9 = '# comment 1'
    var_10 = '# comment 2'
    var_11 = [var_9, var_10]
    var_12 = module_1.import_statement(var_4, var_8, var_11, config=var_3)
    assert var_12 == 'from module import (a, b, c)\n'

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
    assert var_9 == 'from module import (a, b, c,)\n'

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
    assert var_9 == 'from module import (a, b, c)\n'

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
    var_9 = 'comment 1'
    var_10 = 'comment 2'
    var_11 = [var_9, var_10]
    var_12 = module_1.import_statement(var_4, var_8, var_11, config=var_3)
    assert var_12 == 'from module import (a, b, c)\n# comment 1\n# comment 2\n'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_line_predicate_evaluates_to_true. Retrieved 10/15 statements.


import re as module_0

def test_case_0():
    var_0 = 'import  os.path as path'
    var_1 = '\n'
    var_2 = 20
    var_3 = 'as '
    var_4 = '\\b'
    var_5 = module_0.escape(var_3)
    var_6 = var_4 + var_5
    var_7 = var_6 + var_4
    var_8 = module_0.search(var_7, var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 8/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'balanced_wrapping'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import (a, b, c)'
    var_7 = '\n'
    var_8 = -1
    var_9 = 100
    var_10 = -1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_line_71_predicate_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = 50
    var_4 = '\n'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = 100
    var_3 = len(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = 50
    var_4 = len(var_2)
    var_5 = '# NOQA'
    var_6 = bool('# NOQA' not in var_2)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_import_statement_with_multi_line_output. Retrieved 4/6 statements.


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
    var_4 = '\r\n'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = '\r\n'
    var_7 = bool('\r\n' in var_5)
    assert var_7 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    assert var_5 == 'from foo import (\n    bar,\n    baz,\n)'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from foo import'
    var_7 = 'bar'
    var_8 = 'baz'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = ','
    var_12 = bool(',' in var_10)
    assert var_12 is True

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = 'baz'
    var_3 = [var_1, var_2]
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from foo import'
    var_7 = 'bar'
    var_8 = 'baz'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 0
    var_12 = '\n'
    var_13 = var_10.split(var_12)[var_11]
    var_14 = len(var_13)
    var_15 = -1
    var_16 = var_10.split(var_12)[var_15]
    var_17 = len(var_16)
    var_18 = bool(var_14 >= var_17)
    assert var_18 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'bar'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'
    var_5 = bool('\n' not in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = 'very_long_module_name'
    var_2 = 'another_very_long_module_name'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = '\n'
    var_6 = bool('\n' in var_4)
    assert var_6 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from foo import'
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)
    assert var_2 == 'from foo import '

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_line_wrapping_with_import_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_dot_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_as_splitter. Retrieved 3/6 statements.
# Partially parsed test_line_wrapping_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 'module.long_function_name.another_function'
    var_1 = '\n'
    var_2 = 30

def test_case_0():
    var_0 = 'import module as alias'
    var_1 = '\n'
    var_2 = 20

def test_case_0():
    var_0 = 'long_line # comment'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'very_long_line'
    var_1 = '\n'
    var_2 = 10

def test_case_0():
    var_0 = 'very_long_line # NOQA'
    var_1 = '\n'
    var_2 = 10



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 11/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = [var_2, var_3]
    var_5 = 'from package import'
    var_6 = '\n'
    var_7 = ()
    var_8 = False
    var_9 = 1
    var_10 = -1
    var_11 = -1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_regex_search_and_startswith_condition. Retrieved 8/11 statements.


import re as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = 'import '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_line_wrap_with_noqa. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_trailing_comma. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_parentheses_and_no_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_parentheses_and_no_trailing_comma_and_no_wrap. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_parentheses_and_no_trailing_comma_and_no_wrap_and_no_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment_and_no_parentheses_and_no_trailing_comma_and_no_wrap_and_no_comment_and_no_import. Retrieved 4/7 statements.


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
    var_0 = 20
    var_1 = True
    var_2 = 'from module import long_function_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'cimport module.long_function_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'module.long_function_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module as alias'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import module # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'import module # noqa'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = True
    var_3 = 'import module # noqa'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'module'
    var_3 = '\n'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_regex_search_and_startswith_condition. Retrieved 8/11 statements.


import re as module_0

def test_case_0():
    var_0 = 'from module import function'
    var_1 = 'import '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 8/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z)'
    var_8 = '\n'
    var_9 = -1
    var_10 = 20
    var_11 = -1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_while_loop_condition. Retrieved 15/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = 'balanced_wrapping'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'short'
    var_8 = 'even shorter'
    var_9 = [var_7, var_8]
    var_10 = len(var_9)
    var_11 = -1
    var_12 = var_9[:var_11]
    var_13 = 20
    var_14 = -1
    var_15 = var_9[var_14]
    var_16 = len(var_15)
    var_17 = len(var_9)
    var_18 = var_17 == var_10



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 6/7 statements.
# Partially parsed test_import_statement_explode. Retrieved 7/8 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 9/10 statements.
# Partially parsed test_import_statement_custom_line_separator. Retrieved 7/8 statements.
# Partially parsed test_import_statement_custom_config. Retrieved 9/10 statements.
# Partially parsed test_import_statement_multi_line_output. Retrieved 5/8 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 8/9 statements.
# Partially parsed test_import_statement_single_line. Retrieved 5/7 statements.


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
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'from module import'
    var_8 = bool('from module import' in var_6)
    assert var_8 is True
    var_9 = 'a'
    var_10 = bool('a' in var_6)
    assert var_10 is True
    var_11 = 'b'
    var_12 = bool('b' in var_6)
    assert var_12 is True
    var_13 = 'c'
    var_14 = bool('c' in var_6)
    assert var_14 is True

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

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = 'from module import'
    var_13 = bool('from module import' in var_11)
    assert var_13 is True

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'from module import'

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
    var_10 = 'from module import'
    var_11 = bool('from module import' in var_9)
    assert var_11 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = '\n'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_while_loop_predicate_false. Retrieved 18/29 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = False
    var_3 = '#'
    var_4 = '    '
    var_5 = 'balanced_wrapping'
    var_6 = 'wrap_length'
    var_7 = 'line_length'
    var_8 = 'ignore_comments'
    var_9 = 'comment_prefix'
    var_10 = 'include_trailing_comma'
    var_11 = 'indent'
    var_12 = {var_5: var_0, var_6: var_1, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_2, var_11: var_4}
    var_13 = module_0.Config(**var_12)
    var_14 = 'from module import ('
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = [var_15, var_16, var_17]
    var_19 = 'from module import (\n    a,\n    b,\n    c\n)'
    var_20 = '\n'
    var_21 = -1
    var_22 = 10
    var_23 = -1
    var_24 = var_22 > var_1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 7/8 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 8/9 statements.
# Partially parsed test_import_statement_custom_multi_line_output. Retrieved 6/9 statements.
# Partially parsed test_import_statement_single_line. Retrieved 5/6 statements.
# Partially parsed test_import_statement_balanced_wrapping. Retrieved 9/10 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 11/13 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'
    var_6 = module_0.import_statement(var_0, var_4, line_separator=var_5)
    var_7 = 'from module import'
    var_8 = bool('from module import' in var_6)
    assert var_8 is True
    var_9 = 'a'
    var_10 = bool('a' in var_6)
    assert var_10 is True
    var_11 = 'b'
    var_12 = bool('b' in var_6)
    assert var_12 is True
    var_13 = 'c'
    var_14 = bool('c' in var_6)
    assert var_14 is True

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
    var_8 = '\n'
    var_9 = module_0.import_statement(var_0, var_4, var_7, var_8)
    var_10 = '# comment1'
    var_11 = bool('# comment1' in var_9)
    assert var_11 is True
    var_12 = '# comment2'
    var_13 = bool('# comment2' in var_9)
    assert var_13 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = '\n'
    var_7 = module_0.import_statement(var_0, var_4, line_separator=var_6, explode=var_5)

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = '\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = '\n'
    var_4 = module_0.import_statement(var_0, var_2, line_separator=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import'
    var_7 = 'very_long_import_name_a'
    var_8 = 'very_long_import_name_b'
    var_9 = [var_7, var_8]
    var_10 = '\n'
    var_11 = module_1.import_statement(var_6, var_9, line_separator=var_10, config=var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = '    '
    var_1 = True
    var_2 = 'indent'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = '\n'
    var_12 = module_1.import_statement(var_6, var_10, line_separator=var_11, config=var_5)
    var_13 = '    '
    var_14 = bool('    ' in var_12)
    assert var_14 is True
    var_15 = ','

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import'
    var_1 = []
    var_2 = '\n'
    var_3 = module_0.import_statement(var_0, var_1, line_separator=var_2)
    var_4 = 'from module import'
    var_5 = bool('from module import' in var_3)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_while_loop_predicate. Retrieved 10/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 100
    var_2 = False
    var_3 = 'balanced_wrapping'
    var_4 = 'wrap_length'
    var_5 = 'line_length'
    var_6 = 'ignore_comments'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import (a, b, c)'
    var_10 = '\n'
    var_11 = -1
    var_12 = 100
    var_13 = -1



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_17. Retrieved 9/18 statements.


def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = None
    var_4 = '    '
    var_5 = 'import os.path as osp'
    var_6 = 'import os.path as osp'
    var_7 = ','
    var_8 = ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_regex_search_and_startswith_condition. Retrieved 8/11 statements.


import re as module_0

def test_case_0():
    var_0 = 'import os.path as osp'
    var_1 = 'as '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_import_statement_predicate_false. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'from module import'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 100
    var_6 = True
    var_7 = '\n'
    var_8 = -1
    var_9 = -1
    var_10 = 10
    var_11 = var_5 > var_10



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_line_11_predicate_true. Retrieved 8/11 statements.


import re as module_0

def test_case_0():
    var_0 = 'import os.path as osp'
    var_1 = 'as '
    var_2 = '\\b'
    var_3 = module_0.escape(var_1)
    var_4 = var_2 + var_3
    var_5 = var_4 + var_2
    var_6 = module_0.search(var_5, var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_line_predicate_false. Retrieved 12/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 10
    var_5 = None
    var_6 = ''
    var_7 = '# '
    var_8 = False
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'long line that exceeds the line length but has # NOQA'
    var_1 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = 'long line without noqa'
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
    var_6 = 'from module import long_function_name'
    var_7 = 'from module import \\\n    long_function_name'
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
    var_7 = 'from module import long_function_name, another_name'
    var_8 = 'from module import (\n    long_function_name,\n    another_name,\n)'
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
    var_7 = 'import module as (\n    long_alias\n)'
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
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import name  # comment'
    var_7 = 'from module import (\n    name,  # comment\n)'
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
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import name  # noqa'
    var_7 = 'from module import (\n    name\n)  # noqa'
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
    var_6 = 'cimport module.long_function_name'
    var_7 = 'cimport module.\\\n    long_function_name'
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
    var_6 = 'module.long_function_name'
    var_7 = 'module.\\\n    long_function_name'
    var_8 = '\n'
    var_9 = module_1.line(var_6, var_8, var_5)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_wrap_with_import. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_mode. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_mode_and_noqa_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_vertical_grid_grouped. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_use_parentheses_false. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_include_trailing_comma_false. Retrieved 5/8 statements.
# Partially parsed test_line_wrap_with_include_trailing_comma_true. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'short line'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'short line'

def test_case_0():
    var_0 = 'from module import function, another_function'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'from module import (\n    function,\n    another_function,\n)'

def test_case_0():
    var_0 = 'cimport numpy as np, pandas as pd'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'cimport (\n    numpy as np,\n    pandas as pd,\n)'

def test_case_0():
    var_0 = 'module.submodule.function(arg1, arg2)'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'module.submodule.function(\n    arg1,\n    arg2,\n)'

def test_case_0():
    var_0 = 'import module as m, other_module as om'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import module as m,\n    other_module as om'

def test_case_0():
    var_0 = 'import module  # some comment'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import (\n    module,  # some comment\n)'

def test_case_0():
    var_0 = 'import module  # noqa'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import (\n    module,  # noqa\n)'

def test_case_0():
    var_0 = 'import module'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'import module # NOQA'

def test_case_0():
    var_0 = 'import module  # NOQA'
    var_1 = '\n'
    var_2 = 10
    var_3 = 'import module  # NOQA'

def test_case_0():
    var_0 = 'import module, another_module'
    var_1 = '\n'
    var_2 = 30
    var_3 = 'import (\n    module,\n    another_module,\n)'

def test_case_0():
    var_0 = 'import module, another_module'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'import module,\\\n    another_module'

def test_case_0():
    var_0 = 'import module, another_module'
    var_1 = '\n'
    var_2 = 30
    var_3 = False
    var_4 = 'import (\n    module\n    another_module\n)'

def test_case_0():
    var_0 = 'import module, another_module'
    var_1 = '\n'
    var_2 = 30
    var_3 = True
    var_4 = 'import (\n    module,\n    another_module,\n)'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_line_noqa_mode_with_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_noqa_mode_without_noqa_comment. Retrieved 2/5 statements.
# Partially parsed test_line_wrap_with_import. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_as. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_comment. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_parentheses_and_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_noqa_in_comment. Retrieved 4/7 statements.
# Partially parsed test_line_wrap_with_dot. Retrieved 3/6 statements.
# Partially parsed test_line_wrap_with_cimport. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 'from module import very_long_function_name'
    var_1 = '\n'
    var_2 = f'from module import (\n    very_long_function_name)'

def test_case_0():
    var_0 = 'import module as very_long_alias'
    var_1 = '\n'
    var_2 = f'import module as very_long_alias'

def test_case_0():
    var_0 = 'import module # some comment'
    var_1 = '\n'
    var_2 = f'import module # some comment'

def test_case_0():
    var_0 = 'import module1, module2 # some comment'
    var_1 = '\n'
    var_2 = True
    var_3 = f'import (\n    module1,\n    module2,  # some comment\n)'

def test_case_0():
    var_0 = 'import module # noqa: F401'
    var_1 = '\n'
    var_2 = True
    var_3 = f'import module # noqa: F401'

def test_case_0():
    var_0 = 'module.very_long_function_name()'
    var_1 = '\n'
    var_2 = f'module.very_long_function_name()'

def test_case_0():
    var_0 = 'cimport module.very_long_function_name'
    var_1 = '\n'
    var_2 = f'cimport module.very_long_function_name'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = None
    var_6 = True
    var_7 = '#'
    var_8 = '    '
    var_9 = len(var_2)
    var_10 = 2
    var_11 = var_9 + var_10



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_import_statement_predicate_false. Retrieved 13/24 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'wrap_length'
    var_3 = 'balanced_wrapping'
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
    var_15 = 0



# Parsed testcases at query #31
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 88
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = 'comment_prefix'
    var_6 = 'line_length'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something'
    var_10 = '\n'
    var_11 = 'import '
    var_12 = 'from module import something'
    var_13 = None
    var_14 = 'from module '
    var_15 = ' something'
    var_16 = [var_14, var_15]
    var_17 = [var_15]
    var_18 = ' something'
    var_19 = ','
    assert var_19 == ','
    var_20 = ''
    var_21 = ''
    var_22 = f'{var_9}{var_11}({var_21}{var_10}{var_18}{var_19}{var_20})'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'some_content'
    var_1 = '\n'
    var_2 = True
    var_3 = '#'
    var_4 = 100
    var_5 = None
    var_6 = '    '
    var_7 = -1
    var_8 = -1
    var_9 = ')'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 18/25 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = '\n'
    var_4 = 50
    var_5 = None
    var_6 = True
    var_7 = '#'
    var_8 = '    '
    var_9 = 40
    var_10 = var_0 * var_9
    var_11 = 'b'
    var_12 = var_11 * var_9
    var_13 = [var_10, var_12]
    var_14 = '.'
    var_15 = len(var_2)
    var_16 = 2
    var_17 = var_15 + var_16



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = '\n'
    var_2 = 100
    var_3 = None
    var_4 = False
    var_5 = '# '
    var_6 = '    '
    var_7 = len(var_0)
    var_8 = 2
    var_9 = var_7 + var_8



# Parsed testcases at query #35
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



