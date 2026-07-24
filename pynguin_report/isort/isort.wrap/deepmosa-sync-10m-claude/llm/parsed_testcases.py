####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_with_dot_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_noqa_mode_adds_comment. Retrieved 4/7 statements.
# Partially parsed test_line_noqa_mode_existing_noqa. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma_and_parentheses. Retrieved 7/8 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 5/9 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 5/9 statements.
# Partially parsed test_line_without_parentheses_uses_backslash. Retrieved 7/8 statements.
# Partially parsed test_line_with_noqa_in_comment_and_parentheses. Retrieved 8/9 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 7/8 statements.
# Partially parsed test_line_cimport_splitter. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'from module import something'

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
    var_8 = 'from module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True
    var_13 = len(var_10)
    var_14 = bool(var_13 > 0)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import x  # comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'from module import x  # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = 'comment_prefix'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import something  # test'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)
    var_13 = bool('#' in var_12 or 'import' in var_12)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'module.submodule.function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

def test_case_0():
    var_0 = 20
    var_1 = ' #'
    var_2 = 'from module import something'
    var_3 = '\n'
    var_4 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = ' #'
    var_2 = 'from module import something  # NOQA'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = ' #'
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'comment_prefix'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = ' #'
    var_3 = 'from module import something'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = ' #'
    var_3 = 'from module import something'
    var_4 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = ' #'
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = 'comment_prefix'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import something  # noqa'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 50
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport module'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_line_with_comment. Retrieved 8/10 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_as_keyword. Retrieved 7/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 88
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 6
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_very_long_module_name import function_one, function_two, function_three'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import a, b, c  # comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_module import function'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package.subpackage.module import func'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import a, b, c, d'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_module_name import something  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 88
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import sys'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import sys'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = '    '
    var_6 = 'from some_very_long_module_name import some_very_long_function_name_that_exceeds'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_line_content_exceeds_line_length_noqa_mode. Retrieved 10/15 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 8/10 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 8/10 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 8/10 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 8/10 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 7/9 statements.
# Partially parsed test_line_with_backslash_continuation. Retrieved 6/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 8/10 statements.
# Partially parsed test_line_with_vertical_grid_grouped. Retrieved 8/10 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 7
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import '
    var_7 = ', '
    var_8 = 10
    var_9 = range(var_8)
    var_10 = 'module'
    var_11 = [var_10 + str(i) for i in var_9]
    var_12 = '\n'
    var_13 = '# NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = True
    var_2 = 0
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import func1, func2, func3  # comment'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from some_module import function_one, function_two'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from package.subpackage.module import something'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import very_long_module_name as alias_name'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = 'include_trailing_comma'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_1}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import a, b, c, d, e, f'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import func1, func2  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import function_one, function_two'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import x'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)
    assert var_12 == 'import x'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 2
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import func1, func2, func3'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 3
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import func1, func2, func3'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name_that_exceeds_line_length'
    var_7 = 40
    var_8 = 50
    var_9 = 'line_length'
    var_10 = 'wrap_length'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.Config(**var_11)
    var_13 = 'import very_long_module_name_that_exceeds'
    var_14 = 30
    var_15 = 'line_length'
    var_16 = 'wrap_length'
    var_17 = {var_15: var_7, var_16: var_14}
    var_18 = module_0.Config(**var_17)
    var_19 = 'import some_module'
    var_20 = 10
    var_21 = 'line_length'
    var_22 = 'wrap_length'
    var_23 = {var_21: var_7, var_22: var_20}
    var_24 = module_0.Config(**var_23)
    var_25 = 'import some_module_with_long_name'
    var_26 = len(var_25)
    var_27 = 2
    var_28 = var_26 + var_27
    var_29 = var_24.wrap_length
    var_30 = var_24.line_length
    var_31 = var_29 or var_30
    var_32 = var_28 > var_31
    assert var_32 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_noqa_mode_adds_noqa_comment. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 50
    var_1 = 'from some_very_long_module_name import some_very_long_function_name'
    var_2 = '\n'
    var_3 = '# NOQA'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/9 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 7/9 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 5/7 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 7/9 statements.
# Partially parsed test_import_statement_with_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/6 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/5 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 9/12 statements.
# Partially parsed test_import_statement_with_trailing_comma. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_custom_indent. Retrieved 7/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = [var_1]
    var_3 = ';'
    var_4 = module_0.import_statement(var_0, var_2, line_separator=var_3)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True
    var_9 = 'func2'
    var_10 = bool('func2' in var_6)
    assert var_10 is True
    var_11 = 'func3'
    var_12 = bool('func3' in var_6)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 2
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'func1'
    var_8 = 'func2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_func'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_func'
    var_5 = bool('single_func' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'function_one'
    var_8 = 'function_two'
    var_9 = 'function_three'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_line_noqa_mode_adds_noqa. Retrieved 3/8 statements.
# Partially parsed test_line_noqa_mode_no_duplicate_noqa. Retrieved 3/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os  # comment'

def test_case_0():
    var_0 = 40
    var_1 = 'from some_very_long_module_name import function_name'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 40
    var_1 = 'from some_module import name  # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module.submodule.deep import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'as'
    var_12 = bool('as' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_very_long_module_name import function_name'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function_name  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from some_very_long_module_name import function_name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = var_10 > var_1
    var_12 = bool('\\' in var_9 or var_11)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import os'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_line_exceeds_length_with_noqa_mode. Retrieved 3/8 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_as_keyword. Retrieved 7/9 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 7/9 statements.
# Partially parsed test_line_already_has_noqa. Retrieved 3/8 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/10 statements.
# Partially parsed test_line_without_parentheses_backslash. Retrieved 6/8 statements.
# Partially parsed test_line_comment_with_trailing_comma. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function1, function2'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool('import' in var_10 or '(' in var_10)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os  # comment'

def test_case_0():
    var_0 = 40
    var_1 = 'from some_very_long_module_name import function1, function2, function3'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some.very.long.module.path import func'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import very_long_function_name as short_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_module import func1, func2, func3'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module import function  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 40
    var_1 = 'from module import func  # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from cython_module cimport very_long_cython_function_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'from some_module import func1, func2, func3'
    var_3 = '\n'

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'from some_module import func1, func2, func3'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from some_module import func1, func2, func3'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import func1, func2  # comment'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name_that_exceeds_line_length_significantly'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 32/40 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = 'comment_prefix'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import ('
    var_11 = '    something,'
    var_12 = '    another'
    var_13 = [var_10, var_11, var_12]
    var_14 = var_9.comment_prefix
    var_15 = -1
    var_16 = var_13[var_15]
    var_17 = var_14 in var_16
    var_18 = -1
    var_19 = var_13[var_18]
    var_20 = ')'
    var_21 = '    another)'
    var_22 = [var_10, var_11, var_21]
    var_23 = var_9.comment_prefix
    var_24 = -1
    var_25 = var_22[var_24]
    var_26 = var_23 in var_25
    var_27 = -1
    var_28 = var_22[var_27]
    var_29 = '    another # comment'
    var_30 = [var_10, var_11, var_29]
    var_31 = var_9.comment_prefix
    var_32 = -1
    var_33 = var_30[var_32]
    var_34 = var_31 in var_33
    var_35 = -1
    var_36 = var_30[var_35]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 9/12 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 3
    var_5 = 'line_length'
    var_6 = 'use_parentheses'
    var_7 = 'include_trailing_comma'
    var_8 = 'comment_prefix'
    var_9 = 'multi_line_output'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from module import something'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.wrap_length
    var_7 = var_5.line_length
    var_8 = var_6 or var_7
    assert var_8 == 80
    var_9 = None
    var_10 = 'wrap_length'
    var_11 = 'line_length'
    var_12 = {var_10: var_9, var_11: var_1}
    var_13 = module_0.Config(**var_12)
    var_14 = var_13.wrap_length
    var_15 = var_13.line_length
    var_16 = var_14 or var_15
    assert var_16 == 100
    var_17 = 50
    var_18 = 'wrap_length'
    var_19 = 'line_length'
    var_20 = {var_18: var_17, var_19: var_1}
    var_21 = module_0.Config(**var_20)
    var_22 = 52
    var_23 = 2
    var_24 = var_22 + var_23
    var_25 = var_21.wrap_length
    var_26 = var_21.line_length
    var_27 = var_25 or var_26
    var_28 = var_24 > var_27
    assert var_28 is True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import a'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #15
#--------------------------




import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 6
    var_1 = 10
    var_2 = 'multi_line_output'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = len(var_4)
    var_8 = var_3.line_length
    var_9 = var_7 <= var_8
    var_10 = bool('comment' in var_6 or var_9)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # my comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very.long.module import name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import something as alias_name_here'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something_long'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import very_long_module_name  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import something'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 3
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import very_long_function_name_here  # noqa'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = ','
    var_13 = bool(',' in var_11)
    assert var_13 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_comma_added_when_trailing_comma_enabled. Retrieved 10/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = ' #'
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'comment_prefix'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something'
    var_10 = var_8.include_trailing_comma
    var_11 = var_8.use_parentheses
    var_12 = ','
    var_13 = ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_line_17_evaluates_to_true. Retrieved 10/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = ' #'
    var_2 = 80
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'comment_prefix'
    var_6 = 'line_length'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_module import very_long_function_name_that_exceeds_line_length'
    var_10 = '\n'
    var_11 = var_8.include_trailing_comma
    var_12 = var_8.use_parentheses
    var_13 = ','



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_with_noqa_comment_exceeds_length. Retrieved 3/9 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 7/10 statements.
# Partially parsed test_line_backslash_continuation. Retrieved 9/14 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function_one, function_two'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True
    var_13 = len(var_10)
    var_14 = bool(var_13 > 0)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # comment'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = '# comment'
    var_6 = bool('# comment' in var_4)
    assert var_6 is True

def test_case_0():
    var_0 = 40
    var_1 = 'from some_very_long_module_name import function'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package.subpackage.module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import something as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_long_module_name import func1, func2, func3'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = ' #'
    var_1 = 'comment_prefix'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 50
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = 'use_parentheses'
    var_7 = 'multi_line_output'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from very_long_module_name import function_one, function_two, function_three'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)
    var_13 = len(var_12)
    var_14 = bool(var_13 > 0)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'cimport numpy as np'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import something_long'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = var_10 <= var_0
    var_12 = 3



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_line_long_content_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_noqa_mode_already_present. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_split. Retrieved 6/7 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 6/7 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/8 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 6/7 statements.
# Partially parsed test_line_without_parentheses. Retrieved 6/7 statements.
# Partially parsed test_line_custom_comment_prefix. Retrieved 6/7 statements.
# Partially parsed test_line_with_wrap_length. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = '# NOQA'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 15
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from very_long_module import name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'import'
    var_10 = bool('import' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 15
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'some.very.long.module.path.name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 15
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as very_long_alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'use_parentheses'
    var_3 = 'include_trailing_comma'
    var_4 = 'line_length'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import something'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'from module import something'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = False
    var_1 = 15
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = '\\'
    var_10 = bool('\\' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = ' #'
    var_1 = 10
    var_2 = 'comment_prefix'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 30
    var_3 = 'use_parentheses'
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == ''

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 'use_parentheses'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'abc'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'abc'



# Parsed testcases at query #22
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import a'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_line_noqa_mode_long_content. Retrieved 3/9 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 3/8 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 6/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 4/10 statements.
# Partially parsed test_line_without_parentheses. Retrieved 6/8 statements.
# Partially parsed test_line_with_cimport. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_very_long_module_name import some_function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'import'
    var_10 = bool('import' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import func  # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('#' in var_8 or 'comment' in var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 30
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 30
    var_1 = 'import very_long_module_name  # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from package.subpackage.module import func'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('.' in var_8 or 'import' in var_8)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as very_long_alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'as'
    var_10 = bool('as' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import func1, func2'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'from very_long_module_name import function'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from very_long_module_name import function'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import func  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('noqa' in var_8 or '#' in var_8)
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
    var_6 = 'import os'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_false. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_line_with_noqa_mode_exceeds_length. Retrieved 3/8 statements.
# Partially parsed test_line_splits_on_import_keyword. Retrieved 5/10 statements.
# Partially parsed test_line_with_dot_separator. Retrieved 4/10 statements.
# Partially parsed test_line_with_as_separator. Retrieved 4/10 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 4/10 statements.
# Partially parsed test_line_with_noqa_comment_preserved. Retrieved 4/9 statements.
# Partially parsed test_line_with_backslash_continuation. Retrieved 4/10 statements.
# Partially parsed test_line_with_custom_comment_prefix. Retrieved 4/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os  # comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'from very_long_module_name import something'
    var_4 = '\n'
    var_5 = 'import'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from package.subpackage.module import name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very_long_name as vln'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import function1, function2'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something  # noqa'
    var_3 = '\n'
    var_4 = 'noqa'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from very_long_module_name import something'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os, sys'
    var_1 = len(var_0)
    var_2 = 'line_length'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '\n'
    var_6 = module_1.line(var_0, var_5, var_4)
    var_7 = bool(var_6 == var_0)
    assert var_7 is True

def test_case_0():
    var_0 = 10
    var_1 = ' #'
    var_2 = 'import very_long_module_name'
    var_3 = '\n'
    var_4 = 'NOQA'



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'ab'
    var_7 = len(var_6)
    var_8 = 2
    var_9 = var_7 + var_8
    var_10 = var_5.wrap_length
    var_11 = var_5.line_length
    var_12 = var_10 or var_11
    var_13 = var_9 > var_12
    assert var_13 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_line_long_content_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_and_wrapping. Retrieved 4/8 statements.
# Partially parsed test_line_split_on_import. Retrieved 4/7 statements.
# Partially parsed test_line_split_on_dot. Retrieved 4/8 statements.
# Partially parsed test_line_split_on_as. Retrieved 4/8 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 4/8 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/8 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/8 statements.
# Partially parsed test_line_without_parentheses. Retrieved 4/8 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 3/6 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something  # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from very_long_module_name import function'
    var_3 = '\n'
    var_4 = 'import'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'module.submodule.function.method'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'import very_long_name as short'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import a, b, c, d'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something  # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from very_long_module import function'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from very_long_module import function'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from very_long_module import function'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name  # NOQA'
    var_2 = '\n'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_balanced_wrapping_predicate_line_41. Retrieved 8/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import '
    var_8 = 'very_long_import_name_one'
    var_9 = 'very_long_import_name_two'
    var_10 = 'very_long_import_name_three'
    var_11 = [var_8, var_9, var_10]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_line_with_comment_and_wrapping. Retrieved 8/10 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_comment_in_parentheses. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 7/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from x import y'
    var_1 = 88
    var_2 = 'line_length'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '\n'
    var_6 = module_1.line(var_0, var_5, var_4)
    assert var_6 == 'from x import y'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = 88
    var_2 = 'line_length'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '\n'
    var_6 = module_1.line(var_0, var_5, var_4)
    assert var_6 == 'import os  # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_that_exceeds_line_length'
    var_1 = 40
    var_2 = 6
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '\n'
    var_8 = module_1.line(var_0, var_7, var_6)
    var_9 = len(var_8)
    var_10 = var_9 > var_1
    var_11 = bool('NOQA' in var_8 or var_10)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_module import very_long_name_that_exceeds_line_length_when_combined'
    var_1 = 50
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some.very.long.module.path.name import something_here'
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)
    var_11 = bool('.' in var_10 or 'import' in var_10)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_module import some_function as very_long_alias_name_here'
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)
    var_11 = 'as'
    var_12 = bool('as' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name  # important comment'
    var_1 = 50
    var_2 = True
    var_3 = 0
    var_4 = False
    var_5 = 'line_length'
    var_6 = 'use_parentheses'
    var_7 = 'multi_line_output'
    var_8 = 'include_trailing_comma'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = '\n'
    var_12 = module_1.line(var_0, var_11, var_10)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_here'
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = 'include_trailing_comma'
    var_8 = {var_4: var_1, var_5: var_2, var_6: var_3, var_7: var_2}
    var_9 = module_0.Config(**var_8)
    var_10 = '\n'
    var_11 = module_1.line(var_0, var_10, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_1, var_4: var_2, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = '\n'
    var_9 = module_1.line(var_0, var_8, var_7)
    var_10 = len(var_9)
    var_11 = var_10 <= var_1
    var_12 = bool('\\' in var_9 or var_11)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name  # noqa'
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_here'
    var_1 = 40
    var_2 = True
    var_3 = 2
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_here'
    var_1 = 40
    var_2 = True
    var_3 = 3
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'multi_line_output'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_line_17_predicate_evaluates_to_true. Retrieved 8/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'include_trailing_comma'
    var_3 = 'use_parentheses'
    var_4 = 'line_length'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import very_long_function_name_that_exceeds_line_length'
    var_8 = var_6.include_trailing_comma
    var_9 = var_6.use_parentheses
    var_10 = ','



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/8 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # this is a comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool('# this is a comment' in var_6 or var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something_very_long'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'module.submodule.function'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something_very_long'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import something_very_long'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = var_10 > var_1
    var_12 = bool('\\' in var_9 or var_11)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something_very_long  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'noqa'
    var_12 = bool('noqa' in var_10)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long'
    var_3 = '\n'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 15/49 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'short content'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'this is a longer content'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'this is longer # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9
    var_12 = 'short'
    var_13 = len(var_12)
    var_14 = var_3 not in var_12



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 11/33 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 'import very_long_module_name_that_exceeds_line_length_significantly'
    var_6 = len(var_5)
    var_7 = var_3 not in var_5
    var_8 = 'import very_long_module_name_that_exceeds_line_length_significantly # NOQA'
    var_9 = len(var_8)
    var_10 = var_3 not in var_8



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 10/13 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 11/14 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 11/14 statements.
# Partially parsed test_import_statement_single_import. Retrieved 9/12 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 10/13 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 12/15 statements.
# Partially parsed test_import_statement_long_import_start. Retrieved 10/13 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = ()
    var_5 = '\n'
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = None
    var_9 = False
    var_10 = module_1.import_statement(var_0, var_3, var_4, var_5, var_7, var_8, var_9)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True
    var_13 = 'func2'
    var_14 = bool('func2' in var_10)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = ()
    var_6 = '\n'
    var_7 = {}
    var_8 = module_0.Config(**var_7)
    var_9 = None
    var_10 = True
    var_11 = module_1.import_statement(var_0, var_4, var_5, var_6, var_8, var_9, var_10)
    var_12 = 'func1'
    var_13 = bool('func1' in var_11)
    assert var_13 is True
    var_14 = 'func2'
    var_15 = bool('func2' in var_11)
    assert var_15 is True
    var_16 = 'func3'
    var_17 = bool('func3' in var_11)
    assert var_17 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = [var_4]
    var_6 = '\n'
    var_7 = {}
    var_8 = module_0.Config(**var_7)
    var_9 = None
    var_10 = False
    var_11 = module_1.import_statement(var_0, var_3, var_5, var_6, var_8, var_9, var_10)
    var_12 = 'func1'
    var_13 = bool('func1' in var_11)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_func'
    var_2 = [var_1]
    var_3 = ()
    var_4 = '\n'
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = None
    var_8 = False
    var_9 = module_1.import_statement(var_0, var_2, var_3, var_4, var_6, var_7, var_8)
    var_10 = 'single_func'
    var_11 = bool('single_func' in var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = ()
    var_5 = '; '
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = None
    var_9 = False
    var_10 = module_1.import_statement(var_0, var_3, var_4, var_5, var_7, var_8, var_9)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = ()
    var_3 = '\n'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = None
    var_7 = False
    var_8 = module_1.import_statement(var_0, var_1, var_2, var_3, var_5, var_6, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = 'func3'
    var_8 = [var_5, var_6, var_7]
    var_9 = ()
    var_10 = '\n'
    var_11 = None
    var_12 = False
    var_13 = module_1.import_statement(var_4, var_8, var_9, var_10, var_3, var_11, var_12)
    var_14 = 'func1'
    var_15 = bool('func1' in var_13)
    assert var_15 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from very_long_module_name_here import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = ()
    var_5 = '\n'
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = None
    var_9 = False
    var_10 = module_1.import_statement(var_0, var_3, var_4, var_5, var_7, var_8, var_9)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_true. Retrieved 13/28 statements.


def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = '    '
    var_4 = ' #'
    var_5 = 'from module import '
    var_6 = 'very_long_name_one'
    var_7 = 'very_long_name_two'
    var_8 = 'very_long_name_three'
    var_9 = [var_6, var_7, var_8]
    var_10 = '\n'
    var_11 = -1
    var_12 = -1



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'short content'
    var_2 = '\n'
    var_3 = len(var_1)
    var_4 = '# NOQA'
    var_5 = var_4 not in var_1



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 28/46 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short line'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6
    var_8 = '# NOQA'
    var_9 = var_8 not in var_4
    var_10 = 10
    var_11 = 'line_length'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = 'this is a very long line'
    var_15 = len(var_14)
    var_16 = var_13.line_length
    var_17 = var_15 > var_16
    var_18 = var_8 not in var_14
    var_19 = 'line_length'
    var_20 = {var_19: var_10}
    var_21 = module_0.Config(**var_20)
    var_22 = 'this is a very long line # NOQA'
    var_23 = len(var_22)
    var_24 = var_21.line_length
    var_25 = var_23 > var_24
    var_26 = var_8 not in var_22
    var_27 = 5
    var_28 = 'line_length'
    var_29 = {var_28: var_27}
    var_30 = module_0.Config(**var_29)
    var_31 = 'this is a very long line'
    var_32 = len(var_31)
    var_33 = var_30.line_length
    var_34 = var_32 > var_33
    var_35 = var_8 not in var_31



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'import a'
    var_3 = '\n'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 10
    var_4 = 'import very_long_module_name'
    var_5 = 'NOQA'
    var_6 = 'import very_long_module_name # NOQA'
    var_7 = 5
    var_8 = ' #'
    var_9 = 'import os # NOQA'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_import_statement_balanced_wrapping_predicate. Retrieved 16/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = 'd'
    var_12 = 'e'
    var_13 = 'f'
    var_14 = 'g'
    var_15 = 'h'
    var_16 = [var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = ()
    var_18 = '\n'
    var_19 = False
    var_20 = 'module'



# Parsed testcases at query #41
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 80
    var_2 = True
    var_3 = 'wrap_length'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_module import a_very_long_function_name_that_exceeds_wrap_length'
    var_9 = var_7.wrap_length
    var_10 = var_7.line_length
    var_11 = var_9 or var_10
    var_12 = len(var_8)
    var_13 = 2
    var_14 = var_12 + var_13
    var_15 = var_14 > var_11
    assert var_15 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'short line'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'this is a very long line that exceeds limit'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'this is a very long line # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #43
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.wrap_length
    var_7 = var_5.line_length
    var_8 = var_6 or var_7
    assert var_8 == 80
    var_9 = 'test_content'
    var_10 = len(var_9)
    var_11 = 2
    var_12 = var_10 + var_11
    var_13 = bool(var_12 > var_8)
    assert var_13 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 21/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import something'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6
    var_8 = '# NOQA'
    var_9 = var_8 not in var_4
    var_10 = 10
    var_11 = 'line_length'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = 'import something very long'
    var_15 = len(var_14)
    var_16 = var_13.line_length
    var_17 = var_15 > var_16
    var_18 = var_8 not in var_14
    var_19 = 'line_length'
    var_20 = {var_19: var_10}
    var_21 = module_0.Config(**var_20)
    var_22 = 'import something very long # NOQA'
    var_23 = len(var_22)
    var_24 = var_21.line_length
    var_25 = var_23 > var_24
    var_26 = var_8 not in var_22



# Parsed testcases at query #45
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'x'
    var_7 = 150
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 10
    var_4 = 'import very_long_module_name'
    var_5 = 'import very_long_module_name  # NOQA'



# Parsed testcases at query #47
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 40
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_module import very_long_function_name'
    var_7 = len(var_6)
    var_8 = 2
    var_9 = var_7 + var_8
    var_10 = bool(var_9 > (var_5.wrap_length or var_5.line_length))
    assert var_10 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'short line'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'this is a very long line'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'this is a very long line # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_true. Retrieved 10/25 statements.


def test_case_0():
    var_0 = True
    var_1 = 20
    var_2 = False
    var_3 = 'from module import '
    var_4 = 'very_long_import_name_one'
    var_5 = 'very_long_import_name_two'
    var_6 = [var_4, var_5]
    var_7 = '\n'
    var_8 = -1
    var_9 = -1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 14/22 statements.


def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = '\n'
    var_5 = 'from module import something'
    var_6 = 'from module import ('
    var_7 = '    something)'
    var_8 = [var_6, var_7]
    var_9 = -1
    var_10 = var_8[var_9]
    var_11 = -1
    var_12 = var_8[var_11]
    var_13 = ')'



# Parsed testcases at query #51
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 105
    var_8 = var_6 * var_7
    var_9 = '\n'
    var_10 = len(var_8)
    var_11 = 2
    var_12 = var_10 + var_11
    var_13 = var_5.wrap_length
    var_14 = var_5.line_length
    var_15 = var_13 or var_14
    var_16 = var_12 > var_15
    assert var_16 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 10/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = ' #'
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'comment_prefix'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_module import a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p'
    var_10 = '\n'
    var_11 = var_8.include_trailing_comma
    var_12 = var_8.use_parentheses
    var_13 = ','



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/8 statements.
# Partially parsed test_line_long_content_noqa_mode_existing_noqa_unchanged. Retrieved 3/8 statements.
# Partially parsed test_line_with_import_splitter_and_parentheses. Retrieved 5/11 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/10 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 4/10 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/10 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/10 statements.
# Partially parsed test_line_with_noqa_in_comment_and_parentheses. Retrieved 5/11 statements.
# Partially parsed test_line_without_parentheses_uses_backslash. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 20
    var_1 = 'from very_long_module_name import something_else'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'from very_long_module_name import something_else # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = False
    var_3 = 'from module import something'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'very_long_module.submodule.function_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 'from module import func  # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something as alias_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 'from module cimport something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'from module import something  # noqa'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import something'
    var_3 = '\n'



# Parsed testcases at query #54
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import a'
    var_7 = len(var_6)
    var_8 = 2
    var_9 = var_7 + var_8
    var_10 = bool(var_9 > (var_5.wrap_length or var_5.line_length))
    assert var_10 is True
    var_11 = var_5.wrap_length or var_5.line_length
    assert var_11 is False



# Parsed testcases at query #55
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 0
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import very_long_name_that_exceeds_line_length  # comment'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = bool(',' in var_11 or var_11 is not None)
    assert var_12 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_noqa_mode_with_existing_noqa. Retrieved 3/6 statements.
# Partially parsed test_line_with_import_splitter_and_parentheses. Retrieved 5/8 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 6/9 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma_enabled. Retrieved 4/7 statements.
# Partially parsed test_line_with_backslash_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_preserves_line_separator. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_comment_in_comment. Retrieved 5/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 4/7 statements.
# Partially parsed test_line_content_starts_with_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 5/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'from package import module'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 10
    var_1 = 'from package import module # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = 'from package import module'
    var_4 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 'from package import module # comment'
    var_5 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from package import something as alias'
    var_3 = '\n'
    var_4 = 'as'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from package import module'
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from package import module'
    var_3 = '\n'
    var_4 = '\\'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'package.module.submodule.function'
    var_3 = '\n'
    var_4 = '('

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'cimport numpy as np'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'from package import module'
    var_3 = '\r\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = ' #'
    var_3 = 'from package import module # noqa'
    var_4 = '\n'
    var_5 = ')'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from package import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from package import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'import module'
    var_3 = '\n'

def test_case_0():
    var_0 = 50
    var_1 = 30
    var_2 = True
    var_3 = 'from package import module'
    var_4 = '\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 6/9 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/9 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/7 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/6 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 8/11 statements.
# Partially parsed test_import_statement_long_import_list. Retrieved 6/11 statements.
# Partially parsed test_import_statement_with_short_line_length. Retrieved 7/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True
    var_8 = 'func2'
    var_9 = bool('func2' in var_5)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = 'comment1'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = ';'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 4
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'func1'
    var_8 = 'func2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_func'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_func'
    var_5 = bool('single_func' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'very_long_function_name_one'
    var_8 = 'very_long_function_name_two'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = [f'func{i}' for i in var_1]
    var_3 = 'from module import '
    var_4 = module_0.import_statement(var_3, var_2)
    var_5 = range(var_0)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_false. Retrieved 7/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = False
    var_8 = '\n'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/10 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/8 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/6 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/5 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 10/13 statements.
# Partially parsed test_import_statement_long_imports. Retrieved 7/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'name1'
    var_2 = 'name2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'name1'
    var_6 = bool('name1' in var_4)
    assert var_6 is True
    var_7 = 'name2'
    var_8 = bool('name2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'name1'
    var_2 = 'name2'
    var_3 = 'name3'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'name1'
    var_8 = bool('name1' in var_6)
    assert var_8 is True
    var_9 = 'name2'
    var_10 = bool('name2' in var_6)
    assert var_10 is True
    var_11 = 'name3'
    var_12 = bool('name3' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'name1'
    var_2 = 'name2'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = '# comment2'
    var_6 = [var_4, var_5]
    var_7 = module_0.import_statement(var_0, var_3, var_6)
    var_8 = 'name1'
    var_9 = bool('name1' in var_7)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'name1'
    var_2 = 'name2'
    var_3 = [var_1, var_2]
    var_4 = '; '
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 2
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'name1'
    var_8 = 'name2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_name'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_name'
    var_5 = bool('single_name' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'name1'
    var_8 = 'name2'
    var_9 = 'name3'
    var_10 = 'name4'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_1.import_statement(var_6, var_11, config=var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'very_long_name_one'
    var_6 = 'very_long_name_two'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'alpha'
    var_1 = 'beta'
    var_2 = 'gamma'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'from pkg import '
    var_5 = module_0.import_statement(var_4, var_3)
    var_6 = 'alpha'
    var_7 = bool('alpha' in var_5)
    assert var_7 is True
    var_8 = 'beta'
    var_9 = bool('beta' in var_5)
    assert var_9 is True
    var_10 = 'gamma'
    var_11 = bool('gamma' in var_5)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/9 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/8 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/10 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_multi_line_output_mode. Retrieved 4/9 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/6 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/5 statements.
# Partially parsed test_import_statement_with_trailing_comma. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 10/13 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'foo'
    var_6 = bool('foo' in var_4)
    assert var_6 is True
    var_7 = 'bar'
    var_8 = bool('bar' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'foo'
    var_8 = bool('foo' in var_6)
    assert var_8 is True
    var_9 = 'bar'
    var_10 = bool('bar' in var_6)
    assert var_10 is True
    var_11 = 'baz'
    var_12 = bool('baz' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = ';'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = '# comment2'
    var_6 = [var_4, var_5]
    var_7 = module_0.import_statement(var_0, var_3, var_6)
    var_8 = 'foo'
    var_9 = bool('foo' in var_7)
    assert var_9 is True
    var_10 = 'bar'
    var_11 = bool('bar' in var_7)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 2
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'foo'
    var_8 = 'bar'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'foo'
    var_5 = bool('foo' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'foo'
    var_6 = 'bar'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'foo'
    var_8 = 'bar'
    var_9 = 'baz'
    var_10 = 'qux'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_1.import_statement(var_6, var_11, config=var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_long_content_noqa_mode. Retrieved 6/11 statements.
# Partially parsed test_line_noqa_comment_preserved. Retrieved 10/15 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 8/13 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 8/13 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.line(var_0, var_1, var_3)
    var_5 = bool(var_4 == var_0)
    assert var_5 is True

def test_case_0():
    var_0 = 'import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = '\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.line(var_0, var_1, var_3)
    var_5 = 'comment'
    var_6 = bool('comment' in var_4)
    assert var_6 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50
    var_8 = 'use_parentheses'
    var_9 = 'line_length'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = module_1.line(var_4, var_5, var_11)
    var_13 = 'import'
    var_14 = bool('import' in var_12)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module.'
    var_1 = 'submodule'
    var_2 = 20
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = ' import something'
    var_6 = var_4 + var_5
    var_7 = '\n'
    var_8 = True
    var_9 = 50
    var_10 = 'use_parentheses'
    var_11 = 'line_length'
    var_12 = {var_10: var_8, var_11: var_9}
    var_13 = module_0.Config(**var_12)
    var_14 = module_1.line(var_6, var_7, var_13)
    var_15 = bool('.' in var_14 or 'import' in var_14)
    assert var_15 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import something as '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50
    var_8 = 'use_parentheses'
    var_9 = 'line_length'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = module_1.line(var_4, var_5, var_11)
    var_13 = 'as'
    var_14 = bool('as' in var_12)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = False
    var_7 = 50
    var_8 = 'use_parentheses'
    var_9 = 'line_length'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = module_1.line(var_4, var_5, var_11)
    var_13 = '\\'
    var_14 = bool('\\' in var_12)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50
    var_8 = 'use_parentheses'
    var_9 = 'include_trailing_comma'
    var_10 = 'line_length'
    var_11 = {var_8: var_6, var_9: var_6, var_10: var_7}
    var_12 = module_0.Config(**var_11)
    var_13 = module_1.line(var_4, var_5, var_12)
    var_14 = ','
    var_15 = bool(',' in var_13)
    assert var_15 is True

def test_case_0():
    var_0 = 'import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '  # noqa'
    var_6 = var_4 + var_5
    var_7 = '\n'
    var_8 = True
    var_9 = 50
    var_10 = 'noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50
    var_8 = '    '
    var_9 = 'use_parentheses'
    var_10 = 'line_length'
    var_11 = 'indent'
    var_12 = {var_9: var_6, var_10: var_7, var_11: var_8}
    var_13 = module_0.Config(**var_12)
    var_14 = module_1.line(var_4, var_5, var_13)
    var_15 = len(var_14)
    var_16 = len(var_4)
    var_17 = bool(var_15 > var_16)
    assert var_17 is True

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '\n'
    var_6 = True
    var_7 = 50

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = '  # noqa'
    var_6 = var_4 + var_5
    var_7 = '\n'
    var_8 = True
    var_9 = 50
    var_10 = 'use_parentheses'
    var_11 = 'include_trailing_comma'
    var_12 = 'line_length'
    var_13 = {var_10: var_8, var_11: var_8, var_12: var_9}
    var_14 = module_0.Config(**var_13)
    var_15 = module_1.line(var_6, var_7, var_14)
    var_16 = 'noqa'
    var_17 = bool('noqa' in var_15)
    assert var_17 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_line_17_evaluates_to_true. Retrieved 8/19 statements.


def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = ' #'
    var_3 = 'from module import very_long_function_name  # comment'
    var_4 = '\n'
    var_5 = 'from module import very_long_function_name'
    var_6 = ','
    var_7 = ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_line_long_content_with_noqa_mode. Retrieved 3/9 statements.
# Partially parsed test_line_with_parentheses_trailing_comma. Retrieved 5/12 statements.
# Partially parsed test_line_dot_splitter. Retrieved 6/8 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/10 statements.
# Partially parsed test_line_with_multiple_comments. Retrieved 5/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name_here'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from some_module import something_very_long  # important comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = 80
    var_12 = var_10 <= var_11
    var_13 = bool('\\' in var_9 or var_12)
    assert var_13 is True

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'from package import module1, module2, module3'
    var_3 = '\n'
    var_4 = len(var_2)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from very_long_package_name import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'import'
    var_10 = bool('import' in var_8)
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
    var_6 = 'from very.long.module.path import name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as very_long_alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'as'
    var_10 = bool('as' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import x, y, z  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from package import module1, module2'
    var_3 = '\n'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from package import module1, module2'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # comment here'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # short'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import x'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 18/34 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = False
    var_10 = '\n'
    var_11 = -1
    var_12 = min(var_6)
    var_13 = 0
    var_14 = var_3.wrap_length
    var_15 = var_3.line_length
    var_16 = var_14 or var_15
    var_17 = -1
    var_18 = 10
    var_19 = var_16 > var_18



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 9/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'a'
    var_8 = [var_7]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'use_parentheses'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'noqa'
    var_5 = var_3.use_parentheses
    var_6 = 'noqa'
    var_7 = var_6 in var_4
    var_8 = var_5 and var_7



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/9 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/6 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/7 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 8/11 statements.
# Partially parsed test_import_statement_preserves_imports. Retrieved 6/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True
    var_9 = 'func2'
    var_10 = bool('func2' in var_6)
    assert var_10 is True
    var_11 = 'func3'
    var_12 = bool('func3' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True
    var_9 = 'func2'
    var_10 = bool('func2' in var_6)
    assert var_10 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '; '
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 4
    var_2 = 'line_length'
    var_3 = 'indent'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'func1'
    var_8 = 'func2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'func1'
    var_5 = bool('func1' in var_3)
    assert var_5 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'very_long_function_name_1'
    var_8 = 'very_long_function_name_2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'very_long_function_name_1'
    var_12 = bool('very_long_function_name_1' in var_10)
    assert var_12 is True
    var_13 = 'very_long_function_name_2'
    var_14 = bool('very_long_function_name_2' in var_10)
    assert var_14 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'func1'
    var_1 = 'func2'
    var_2 = 'func3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'from mymodule import '
    var_5 = module_0.import_statement(var_4, var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_true. Retrieved 9/14 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = ' #'
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'comment_prefix'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_module import very_long_name_one, very_long_name_two  # noqa'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = var_8.comment_prefix
    var_13 = -1
    var_14 = ')'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_line_long_content_noqa_mode. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 4/8 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 4/8 statements.
# Partially parsed test_line_with_hanging_indent_mode. Retrieved 4/8 statements.
# Partially parsed test_line_with_grid_grouped_mode. Retrieved 4/8 statements.
# Partially parsed test_line_without_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_content_starts_with_splitter. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something_very_long'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something # comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'
    var_4 = 'import'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module cimport something'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'module.submodule.something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'import something as alias_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something # noqa'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import something'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'from module import something'
    var_3 = '\n'
    var_4 = '\\'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'import something'
    var_3 = '\n'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_noqa_mode_adds_noqa_comment. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = len(var_0)
    var_3 = '# NOQA'
    var_4 = bool('# NOQA' not in var_0)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 150
    var_8 = var_6 * var_7
    var_9 = '\n'
    var_10 = len(var_8)
    var_11 = 2
    var_12 = var_10 + var_11
    var_13 = var_5.wrap_length
    var_14 = var_5.line_length
    var_15 = var_13 or var_14
    var_16 = var_12 > var_15
    assert var_16 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_line_predicate_evaluates_to_false. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'import os'
    var_2 = len(var_1)



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 50
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import a, b, c'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_true. Retrieved 10/15 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 3
    var_2 = True
    var_3 = ' #'
    var_4 = 'line_length'
    var_5 = 'multi_line_output'
    var_6 = 'use_parentheses'
    var_7 = 'include_trailing_comma'
    var_8 = 'comment_prefix'
    var_9 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_2, var_8: var_3}
    var_10 = module_0.Config(**var_9)
    var_11 = 'from some_module import very_long_name_one, very_long_name_two  # noqa'
    var_12 = '\n'
    var_13 = module_1.line(var_11, var_12, var_10)
    var_14 = -1
    var_15 = var_10.comment_prefix
    var_16 = ')'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_noqa_mode_with_existing_noqa_unchanged. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_extraction. Retrieved 4/8 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/8 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 4/8 statements.
# Partially parsed test_line_without_parentheses. Retrieved 4/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 4/8 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 4/8 statements.
# Partially parsed test_line_with_custom_comment_prefix. Retrieved 5/9 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 5/9 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 5
    var_1 = 'import os  # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something  # important'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'module.submodule.function'
    var_3 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'import something as other_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from module import something  # noqa'
    var_3 = '\n'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = ' #'
    var_3 = 'from module import something'
    var_4 = '\n'

def test_case_0():
    var_0 = 80
    var_1 = 50
    var_2 = True
    var_3 = 'from module import something'
    var_4 = '\n'

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'from cython cimport something_long'
    var_3 = '\n'



# Parsed testcases at query #22
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import a'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_line_with_noqa_comment. Retrieved 4/9 statements.
# Partially parsed test_line_noqa_mode_adds_comment. Retrieved 3/8 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/9 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some.very.long.module.path import something, another, third'
    var_1 = 40
    var_2 = True
    var_3 = False
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True
    var_13 = len(var_10)
    var_14 = bool(var_13 > 0)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import something  # important comment'
    var_1 = 30
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '\n'
    var_8 = module_1.line(var_0, var_7, var_6)
    var_9 = bool('important comment' in var_8 or '#' in var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 'from very.long.module.name import something  # noqa'
    var_1 = 30
    var_2 = True
    var_3 = '\n'
    var_4 = 'noqa'

def test_case_0():
    var_0 = 'from some.very.long.module.path import something, another, third, fourth'
    var_1 = 40
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some.very.long.module.name.submodule import function'
    var_1 = 30
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '\n'
    var_8 = module_1.line(var_0, var_7, var_6)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import something as very_long_name_that_exceeds_limit'
    var_1 = 40
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '\n'
    var_8 = module_1.line(var_0, var_7, var_6)
    var_9 = 'as'
    var_10 = bool('as' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p'
    var_1 = 40
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_1, var_4: var_2, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = '\n'
    var_9 = module_1.line(var_0, var_8, var_7)
    var_10 = len(var_9)
    var_11 = 0
    var_12 = var_10 > var_11
    var_13 = bool('(' in var_9 or var_12)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some.long.module import something, another, third'
    var_1 = 30
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '\n'
    var_8 = module_1.line(var_0, var_7, var_6)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = ';'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.line(var_0, var_1, var_3)
    var_5 = bool(var_4 == var_0)
    assert var_5 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from some.very.long.module.path import something, another, third'
    var_1 = 40
    var_2 = True
    var_3 = 4
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'indent'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = '\n'
    var_10 = module_1.line(var_0, var_9, var_8)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

def test_case_0():
    var_0 = 'from some.very.long.module.path import something, another, third'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = '('

def test_case_0():
    var_0 = 'from some.very.long.module.path import something, another, third'
    var_1 = 40
    var_2 = True
    var_3 = '\n'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_line_4_evaluates_to_false. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.wrap_length
    var_7 = var_5.line_length
    var_8 = var_6 or var_7
    assert var_8 == 100



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_noqa_mode_adds_noqa_comment. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'this is a very long line'
    var_2 = '\n'
    var_3 = '# NOQA'



# Parsed testcases at query #27
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import something'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 13/20 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = True
    var_2 = False
    var_3 = ' #'
    var_4 = 3
    var_5 = '    '
    var_6 = 'line_length'
    var_7 = 'wrap_length'
    var_8 = 'use_parentheses'
    var_9 = 'include_trailing_comma'
    var_10 = 'comment_prefix'
    var_11 = 'multi_line_output'
    var_12 = 'indent'
    var_13 = {var_6: var_0, var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from some_module import (very_long_function_name_one, very_long_function_name_two)'
    var_16 = '\n'
    var_17 = module_1.line(var_15, var_16, var_14)
    var_18 = bool(var_17 is not None)
    assert var_18 is True
    var_19 = -1
    var_20 = var_14.comment_prefix
    var_21 = ')'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/9 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/7 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/6 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 9/12 statements.
# Partially parsed test_import_statement_preserves_import_start. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = 'func3'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True
    var_9 = 'func2'
    var_10 = bool('func2' in var_6)
    assert var_10 is True
    var_11 = 'func3'
    var_12 = bool('func3' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '# important'
    var_5 = '# note'
    var_6 = [var_4, var_5]
    var_7 = module_0.import_statement(var_0, var_3, var_6)
    var_8 = 'func1'
    var_9 = bool('func1' in var_7)
    assert var_9 is True
    var_10 = 'func2'
    var_11 = bool('func2' in var_7)
    assert var_11 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '; '
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'func1'
    var_8 = 'func2'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'func1'
    var_12 = bool('func1' in var_10)
    assert var_12 is True
    var_13 = 'func2'
    var_14 = bool('func2' in var_10)
    assert var_14 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_func'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_func'
    var_5 = bool('single_func' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'balanced_wrapping'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'function1'
    var_8 = 'function2'
    var_9 = 'function3'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = 'function1'
    var_13 = bool('function1' in var_11)
    assert var_13 is True
    var_14 = 'function2'
    var_15 = bool('function2' in var_11)
    assert var_15 is True
    var_16 = 'function3'
    var_17 = bool('function3' in var_11)
    assert var_17 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from mymodule import '
    var_1 = 'item'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'mymodule'
    var_5 = bool('mymodule' in var_3)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 0
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = '    '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'include_trailing_comma'
    var_10 = 'comment_prefix'
    var_11 = 'indent'
    var_12 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_0.Config(**var_12)
    var_14 = '\n'
    var_15 = 'from module import (something,\n    other)'
    var_16 = module_1.line(var_15, var_14, var_13)
    var_17 = bool(var_16 is not None)
    assert var_17 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 7/11 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 7/10 statements.
# Partially parsed test_import_statement_explode_mode. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_custom_indent. Retrieved 7/10 statements.
# Partially parsed test_import_statement_single_import. Retrieved 5/8 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 4/7 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 9/12 statements.
# Partially parsed test_import_statement_with_trailing_comma. Retrieved 8/11 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.import_statement(var_0, var_4, config=var_6)
    var_8 = 'a'
    var_9 = bool('a' in var_7)
    assert var_9 is True
    var_10 = 'b'
    var_11 = bool('b' in var_7)
    assert var_11 is True
    var_12 = 'c'
    var_13 = bool('c' in var_7)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = '# important'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.import_statement(var_0, var_3, var_5, config=var_7)
    var_9 = 'a'
    var_10 = bool('a' in var_8)
    assert var_10 is True
    var_11 = 'b'
    var_12 = bool('b' in var_8)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = '; '
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.import_statement(var_0, var_3, line_separator=var_4, config=var_6)
    var_8 = 'a'
    var_9 = bool('a' in var_7)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1.import_statement(var_0, var_4, config=var_7, explode=var_5)
    var_9 = 'a'
    var_10 = bool('a' in var_8)
    assert var_10 is True
    var_11 = 'b'
    var_12 = bool('b' in var_8)
    assert var_12 is True
    var_13 = 'c'
    var_14 = bool('c' in var_8)
    assert var_14 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 4
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_import'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.import_statement(var_0, var_2, config=var_4)
    var_6 = 'single_import'
    var_7 = bool('single_import' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.import_statement(var_0, var_1, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'very_long_name_one'
    var_8 = 'very_long_name_two'
    var_9 = 'very_long_name_three'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_1.import_statement(var_4, var_8, config=var_3)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'alpha'
    var_1 = 'beta'
    var_2 = 'gamma'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'from pkg import '
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.import_statement(var_4, var_3, config=var_6)
    var_8 = 'alpha'
    var_9 = bool('alpha' in var_7)
    assert var_9 is True
    var_10 = 'beta'
    var_11 = bool('beta' in var_7)
    assert var_11 is True
    var_12 = 'gamma'
    var_13 = bool('gamma' in var_7)
    assert var_13 is True



# Parsed testcases at query #32
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 120
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_noqa_mode_adds_noqa_comment. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'from some_module import very_long_name_that_exceeds_line_length_significantly'
    var_1 = 40
    var_2 = ' #'
    var_3 = '\n'
    var_4 = '# NOQA'



# Parsed testcases at query #34
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'x'
    var_7 = 150
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_import_statement_basic. Retrieved 5/8 statements.
# Partially parsed test_import_statement_with_explode. Retrieved 7/10 statements.
# Partially parsed test_import_statement_with_comments. Retrieved 8/11 statements.
# Partially parsed test_import_statement_with_custom_line_separator. Retrieved 6/9 statements.
# Partially parsed test_import_statement_with_custom_config. Retrieved 8/11 statements.
# Partially parsed test_import_statement_single_import. Retrieved 4/7 statements.
# Partially parsed test_import_statement_empty_imports. Retrieved 3/6 statements.
# Partially parsed test_import_statement_with_balanced_wrapping. Retrieved 9/12 statements.
# Partially parsed test_import_statement_long_import_start. Retrieved 5/8 statements.
# Partially parsed test_import_statement_multiple_comments. Retrieved 10/13 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'foo'
    var_6 = bool('foo' in var_4)
    assert var_6 is True
    var_7 = 'bar'
    var_8 = bool('bar' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = module_0.import_statement(var_0, var_4, explode=var_5)
    var_7 = 'foo'
    var_8 = bool('foo' in var_6)
    assert var_8 is True
    var_9 = 'bar'
    var_10 = bool('bar' in var_6)
    assert var_10 is True
    var_11 = 'baz'
    var_12 = bool('baz' in var_6)
    assert var_12 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = '# comment2'
    var_6 = [var_4, var_5]
    var_7 = module_0.import_statement(var_0, var_3, var_6)
    var_8 = 'foo'
    var_9 = bool('foo' in var_7)
    assert var_9 is True
    var_10 = 'bar'
    var_11 = bool('bar' in var_7)
    assert var_11 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = '; '
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = 'foo'
    var_7 = bool('foo' in var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'include_trailing_comma'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'foo'
    var_8 = 'bar'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'foo'
    var_12 = bool('foo' in var_10)
    assert var_12 is True
    var_13 = 'bar'
    var_14 = bool('bar' in var_10)
    assert var_14 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_import'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_import'
    var_5 = bool('single_import' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'very_long_name_one'
    var_8 = 'very_long_name_two'
    var_9 = 'very_long_name_three'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.import_statement(var_6, var_10, config=var_5)
    var_12 = 'very_long_name_one'
    var_13 = bool('very_long_name_one' in var_11)
    assert var_13 is True
    var_14 = 'very_long_name_two'
    var_15 = bool('very_long_name_two' in var_11)
    assert var_15 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from some_very_long_module_name import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'foo'
    var_6 = bool('foo' in var_4)
    assert var_6 is True
    var_7 = 'bar'
    var_8 = bool('bar' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = 'baz'
    var_4 = [var_1, var_2, var_3]
    var_5 = '# comment1'
    var_6 = '# comment2'
    var_7 = '# comment3'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.import_statement(var_0, var_4, var_8)
    var_10 = 'foo'
    var_11 = bool('foo' in var_9)
    assert var_11 is True



# Parsed testcases at query #36
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = bool(var_5.wrap_length or var_5.line_length)
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_line_noqa_mode_adds_comment. Retrieved 3/8 statements.
# Partially parsed test_line_noqa_mode_existing_noqa. Retrieved 3/8 statements.
# Partially parsed test_line_with_backslash_when_no_parentheses. Retrieved 7/11 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_long_module_name import something_else'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'import'
    var_10 = bool('import' in var_8)
    assert var_10 is True
    var_11 = 0
    var_12 = var_8.split(var_7)[var_11]
    var_13 = len(var_12)
    var_14 = var_5.line_length
    var_15 = var_13 <= var_14
    var_16 = bool(var_15 or '(' in var_8)
    assert var_16 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_long_module_name import something_else  # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'comment'
    var_10 = bool('comment' in var_8)
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
    var_6 = 'from package.subpackage.module import item'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
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
    var_6 = 'from module import something as alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

def test_case_0():
    var_0 = 30
    var_1 = 'from some_long_module_name import something_else'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 30
    var_1 = 'from some_long_module_name import something_else  # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from some_long_module_name import something_else'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_long_module_name import something_else'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 1

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport some_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from some_long_module_name import something_else'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_long_module_name import something_else  # noqa: E501'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'noqa'
    var_10 = bool('noqa' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import x'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #38
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 150
    var_8 = var_6 * var_7
    var_9 = '\n'
    var_10 = var_5.multi_line_output
    var_11 = len(var_8)
    var_12 = var_5.line_length
    var_13 = var_11 > var_12
    var_14 = 75
    var_15 = var_8[:var_14]
    var_16 = var_8[var_14:]
    var_17 = [var_15, var_16]
    var_18 = len(var_8)
    var_19 = 2
    var_20 = var_18 + var_19
    var_21 = bool(var_20 > (var_5.wrap_length or var_5.line_length))
    assert var_21 is True
    var_22 = bool(var_17)
    assert var_22 is True



# Parsed testcases at query #39
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 80
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 60
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_line_with_import_splitter. Retrieved 6/8 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 6/8 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 7/9 statements.
# Partially parsed test_line_cimport_splitter. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something # comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = var_11 <= var_0
    var_13 = bool('(' in var_10 or var_12 or 'import something' in var_10)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import very_long_module_name_here'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from package.subpackage.module import item'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import module as very_long_alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import something_long'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import very_long_name  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 2
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something_very_long'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 3
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import item_one, item_two'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import module'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from very_long_module import item'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = var_10 <= var_0
    var_12 = bool('\\' in var_9 or var_11)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'cimport very_long_module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_line_41_predicate_evaluates_to_false. Retrieved 14/26 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 41 evaluates to False.'
    var_1 = True
    var_2 = 80
    var_3 = 'balanced_wrapping'
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_3: var_1, var_4: var_2, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import '
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_9, var_10, var_11]
    var_13 = 'short'
    var_14 = [var_13]
    var_15 = 10
    var_16 = 'balanced_wrapping'
    var_17 = 'line_length'
    var_18 = 'wrap_length'
    var_19 = {var_16: var_1, var_17: var_15, var_18: var_15}
    var_20 = module_0.Config(**var_19)
    var_21 = [var_9, var_10]



# Parsed testcases at query #42
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = var_5.wrap_length
    var_7 = var_5.line_length
    var_8 = var_6 or var_7
    assert var_8 == 80
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #43
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 0
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = '    '
    var_6 = 'line_length'
    var_7 = 'multi_line_output'
    var_8 = 'use_parentheses'
    var_9 = 'include_trailing_comma'
    var_10 = 'comment_prefix'
    var_11 = 'indent'
    var_12 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_0.Config(**var_12)
    var_14 = 'from some_module import (very_long_name_one, very_long_name_two, very_long_name_three)'
    var_15 = '\n'
    var_16 = module_1.line(var_14, var_15, var_13)
    var_17 = bool(var_16 is not None)
    assert var_17 is True



# Parsed testcases at query #44
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 0
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from some_module import very_long_name_one, very_long_name_two, very_long_name_three'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 9/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_line_noqa_mode_adds_noqa_comment. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 50
    var_1 = ' #'
    var_2 = 'from some_very_long_module_name import some_very_long_function_name'
    var_3 = '\n'
    var_4 = len(var_2)
    var_5 = '# NOQA'
    var_6 = bool('# NOQA' not in var_2)
    assert var_6 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short line'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6



# Parsed testcases at query #48
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = None
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'x'
    var_7 = 100
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_with_existing_noqa_comment_unchanged. Retrieved 3/6 statements.
# Partially parsed test_line_with_import_splitter_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_preserved. Retrieved 4/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_as_splitter_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_backslash_wrapping. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma_included. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_noqa_in_comment_special_handling. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 5/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something_very_long'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import something_very_long  # NOQA'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long_name'
    var_3 = '\n'
    var_4 = 'import'
    var_5 = '('
    var_6 = ')'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from module import something  # important comment'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'some_module.submodule.very_long_function_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something as very_long_alias'
    var_3 = '\n'
    var_4 = 'as'

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from module import something_very_long_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long  # noqa: E501'
    var_3 = '\n'

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module cimport something_very_long_name'
    var_3 = '\n'

def test_case_0():
    var_0 = 50
    var_1 = 30
    var_2 = True
    var_3 = 'from module import something_very_long_function_name'
    var_4 = '\n'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_line_with_noqa_mode_adds_noqa_comment. Retrieved 3/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 10
    var_1 = 'from os import path'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function_one, function_two'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from os import path  # important comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = len(var_6)
    var_11 = var_9 > var_10
    var_12 = bool('important comment' in var_8 or var_11)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from package.subpackage.module import func'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
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
    var_6 = 'from module import very_long_name as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

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
    var_7 = 'from module import function_one, function_two'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from libc cimport stdlib, stdio'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
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
    var_6 = 'from os import path, sys  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'noqa'
    var_10 = bool('noqa' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 50
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import sys'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import function_one, function_two'
    var_3 = '\n'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_line_noqa_mode_long_content. Retrieved 3/8 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 6/8 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 7/9 statements.
# Partially parsed test_line_multiple_comments. Retrieved 6/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'import very_long_module_name'
    var_2 = '\n'
    var_3 = 'NOQA'

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
    var_8 = 'from module import something  # comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'comment'
    var_12 = bool('comment' in var_10)
    assert var_12 is True

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
    var_8 = 'from very_long_module_name import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module.submodule.item import name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import very_long_name as alias'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

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
    var_7 = 'from module import item1, item2'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

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
    var_8 = 'from module import something  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from very_long_module import item1, item2'
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
    var_6 = 'from very_long_module import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('\\' in var_8 or 'from' in var_8)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 40
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_module_name import something_else'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    assert var_8 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import x  # first comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_line_with_noqa_mode_exceeds_length. Retrieved 3/8 statements.
# Partially parsed test_line_split_on_import. Retrieved 4/9 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/10 statements.
# Partially parsed test_line_trailing_comma_enabled. Retrieved 4/10 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 4/10 statements.
# Partially parsed test_line_exact_length. Retrieved 5/7 statements.
# Partially parsed test_line_with_backslash_continuation. Retrieved 4/10 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from os import path'

def test_case_0():
    var_0 = 20
    var_1 = 'from os import path'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from os import path  # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'from os import path  # comment'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from os import path, sep'
    var_3 = '\n'
    var_4 = 'import'

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'import numpy as np'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from package.module.submodule import func'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from os import path, sep'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from os import path  # noqa'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from os import path'
    var_1 = '\n'
    var_2 = 19
    var_3 = 'line_length'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.line(var_0, var_1, var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'from os import path, sep'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from os import path, sep, environ'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from os import path, sep, environ'
    var_3 = '\n'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_line_with_noqa_comment_preserves_it. Retrieved 7/9 statements.
# Partially parsed test_line_empty_content_after_split. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 4
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os, sys, json'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something  # important'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool('# important' in var_10 or 'important' in var_10)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_module_name import function_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module.submodule.another import item'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'module'
    var_12 = bool('module' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'as'
    var_12 = bool('as' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 0
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_module import item'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = len(var_10)
    var_12 = var_11 <= var_0
    var_13 = bool('\\' in var_10 or var_12 or 'import' in var_10)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import a, b, c, d'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = bool(var_11 is not None)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 2
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import item1, item2'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import os'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    assert var_10 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import something  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 0
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import os'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 4
    var_2 = ' #'
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'comment_prefix'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import os, sys, json, collections'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool(var_10 is not None)
    assert var_11 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_line_long_content_with_noqa_mode. Retrieved 7/12 statements.
# Partially parsed test_line_noqa_comment_handling. Retrieved 7/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os, sys'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 1

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
    var_7 = 'from module import something  # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = 0
    var_12 = var_10 > var_11
    var_13 = bool('comment' in var_9 or var_12)
    assert var_13 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from package.module import func'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import something as alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
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
    var_6 = 'from x import y  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 2
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module1, module2'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module1, module2'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool(var_10 is not None)
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
    var_6 = 'from very_long_package_name import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('\\' in var_8 or var_8 == var_6)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import a, b, c'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = bool(var_9 is not None)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = 3
    var_6 = var_4 * var_5
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_3)
    var_9 = bool(var_8 is not None)
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
    var_6 = 'from x import y  # important comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_line_long_content_noqa_mode. Retrieved 3/8 statements.
# Partially parsed test_line_with_comment_split. Retrieved 4/9 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 4/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/9 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/9 statements.
# Partially parsed test_line_with_trailing_comma. Retrieved 4/10 statements.
# Partially parsed test_line_with_noqa_comment. Retrieved 4/9 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 5/11 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import func'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 20
    var_1 = 'from very_long_module_name import very_long_function_name'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from some_module import function # important comment'
    var_3 = '\n'
    var_4 = 'import'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_long'
    var_3 = '\n'
    var_4 = 'import'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module.submodule.another import func'
    var_3 = '\n'
    var_4 = 'module'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_very_long as alias_name'
    var_3 = '\n'
    var_4 = 'as'

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 'from module import function'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something_long # noqa'
    var_3 = '\n'
    var_4 = 'noqa'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 'from module import func'
    var_1 = len(var_0)
    var_2 = 'line_length'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '\n'
    var_6 = module_1.line(var_0, var_5, var_4)
    var_7 = bool(var_6 == var_0)
    assert var_7 is True

def test_case_0():
    var_0 = 80
    var_1 = 40
    var_2 = True
    var_3 = 'from some_module import some_function_with_long_name'
    var_4 = '\n'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_line_long_content_with_import_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_long_content_noqa_mode_no_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_noqa_mode_with_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment. Retrieved 4/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/7 statements.
# Partially parsed test_line_without_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True
    var_4 = 'import'

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length # NOQA'
    var_1 = '\n'
    var_2 = 40

def test_case_0():
    var_0 = 'from module import something  # this is a comment'
    var_1 = '\n'
    var_2 = 20
    var_3 = True
    var_4 = '#'

def test_case_0():
    var_0 = 'from some.very.long.module.path.that.exceeds.line.length import something'
    var_1 = '\n'
    var_2 = 40
    var_3 = True

def test_case_0():
    var_0 = 'from module import very_long_function_name as very_long_alias_name_exceeding_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True
    var_4 = 'as'

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True

def test_case_0():
    var_0 = 'from some_very_long_cython_module cimport very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = True

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length'
    var_1 = '\n'
    var_2 = 40
    var_3 = False

def test_case_0():
    var_0 = 'from some_very_long_module_name import very_long_function_name_that_exceeds_line_length  # noqa'
    var_1 = '\n'
    var_2 = 40
    var_3 = True
    var_4 = 'noqa'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 80
    var_1 = 60
    var_2 = 'import very_long_module_name_that_exceeds_wrap_length'
    var_3 = 50
    var_4 = 'import very_long_module_name_that_exceeds_wrap_length'
    var_5 = len(var_4)
    var_6 = 2
    var_7 = var_5 + var_6



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 9/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]
    var_11 = False



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 5/15 statements.


def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'from some_module import very_long_name_that_exceeds_line_length'
    var_3 = ','



# Parsed testcases at query #60
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 88
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 90
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_line_with_comment_preservation. Retrieved 6/8 statements.
# Partially parsed test_line_with_import_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 6/8 statements.
# Partially parsed test_line_with_as_keyword. Retrieved 6/8 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 6/8 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 7/9 statements.
# Partially parsed test_line_with_noqa_in_comment. Retrieved 6/8 statements.
# Partially parsed test_line_with_backslash_continuation. Retrieved 6/8 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 6/8 statements.
# Partially parsed test_line_with_multiple_comments. Retrieved 6/8 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 7
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'x = 1'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something  # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_module_name import function_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module.submodule.nested import x'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import function_one, function_two'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 2
    var_2 = True
    var_3 = 'line_length'
    var_4 = 'multi_line_output'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import name_one, name_two, name_three'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import x  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport numpy as np'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import x  # test comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_line_long_content_noqa_mode_adds_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_noqa_mode_existing_noqa_unchanged. Retrieved 3/6 statements.
# Partially parsed test_line_with_comment_splits_correctly. Retrieved 6/7 statements.
# Partially parsed test_line_import_splitter_with_parentheses. Retrieved 5/11 statements.
# Partially parsed test_line_as_splitter_with_parentheses. Retrieved 6/7 statements.
# Partially parsed test_line_dot_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 6/7 statements.
# Partially parsed test_line_cimport_splitter. Retrieved 6/7 statements.
# Partially parsed test_line_without_splitters_long_content. Retrieved 3/6 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/8 statements.
# Partially parsed test_line_with_noqa_comment_and_parentheses. Retrieved 6/7 statements.
# Partially parsed test_line_custom_indent. Retrieved 7/8 statements.
# Partially parsed test_line_custom_wrap_length. Retrieved 7/8 statements.
# Partially parsed test_line_carriage_return_separator. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

def test_case_0():
    var_0 = 10
    var_1 = 'from package import module'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 10
    var_1 = 'from package import module  # NOQA'
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
    var_6 = 'from very_long_package import module  # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'from package import module'
    var_3 = '\n'
    var_4 = len(var_2)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_name as alias'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from package.subpackage.module import func'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

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
    var_7 = 'from package import module'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'cimport numpy'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

def test_case_0():
    var_0 = 5
    var_1 = 'verylongword'
    var_2 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from package import module'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from package import module  # noqa'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = '    '
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'indent'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = 80
    var_2 = True
    var_3 = 'wrap_length'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from package import module'
    var_7 = '\r\n'
    var_8 = module_1.line(var_6, var_7, var_5)



# Parsed testcases at query #63
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 150
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import a'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 16/50 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'import very_long_module_name'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'import very_long_module_name # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9
    var_12 = 5
    var_13 = 'import os'
    var_14 = len(var_13)
    var_15 = var_3 not in var_13



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 5/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 50
    var_2 = 'from some_module import very_long_function_name_here'
    var_3 = '\n'
    var_4 = ','



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'import very_long_module_name'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'import very_long_module_name # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_line_with_as_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_noqa_mode_exceeds_length. Retrieved 3/7 statements.
# Partially parsed test_line_noqa_already_present. Retrieved 3/7 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 7/9 statements.
# Partially parsed test_line_with_comment_and_parentheses. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/9 statements.
# Partially parsed test_line_custom_line_separator. Retrieved 6/8 statements.
# Partially parsed test_line_indent_configuration. Retrieved 8/10 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 88
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import func'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'from module import func'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'verylongimportname'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'verylongimportname'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 88
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os  # comment'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import verylongfunction'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = 'import'
    var_11 = bool('import' in var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import verylongfunction'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = bool('(' in var_10 and ')' in var_10)
    assert var_11 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import verylongmodulename as vln'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module.submodule import func'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 20
    var_1 = 'from module import verylongfunction'
    var_2 = '\n'
    var_3 = 'NOQA'

def test_case_0():
    var_0 = 20
    var_1 = 'from module import verylongfunction  # NOQA'
    var_2 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import verylongfunction'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import func  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'from module import verylongfunction'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import verylongfunction'
    var_8 = ';'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 4
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'indent'
    var_7 = 'multi_line_output'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import verylongfunction'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 21/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short content'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6
    var_8 = '# NOQA'
    var_9 = var_8 not in var_4
    var_10 = 10
    var_11 = 'line_length'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = 'this is longer content'
    var_15 = len(var_14)
    var_16 = var_13.line_length
    var_17 = var_15 > var_16
    var_18 = var_8 not in var_14
    var_19 = 'line_length'
    var_20 = {var_19: var_10}
    var_21 = module_0.Config(**var_20)
    var_22 = 'this is longer content # NOQA'
    var_23 = len(var_22)
    var_24 = var_21.line_length
    var_25 = var_23 > var_24
    var_26 = var_8 not in var_22



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 8/45 statements.


def test_case_0():
    var_0 = 'short'
    var_1 = '\n'
    var_2 = len(var_0)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_0
    var_5 = 'a'
    var_6 = 1
    var_7 = ' # NOQA'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'import very_long_module_name'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'import very_long_module_name # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 80
    var_1 = 'import short'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'import this is a very long line'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'import this is a very long line # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #72
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 3
    var_3 = ' #'
    var_4 = 'line_length'
    var_5 = 'use_parentheses'
    var_6 = 'include_trailing_comma'
    var_7 = 'multi_line_output'
    var_8 = 'comment_prefix'
    var_9 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Config(**var_9)
    var_11 = 'from module import (very_long_function_name)'
    var_12 = '\n'
    var_13 = module_1.line(var_11, var_12, var_10)
    var_14 = bool(var_13 is not None)
    assert var_14 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_line_41_predicate_evaluates_to_false. Retrieved 7/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = 'balanced_wrapping'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #74
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 0
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import very_long_function_name'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)
    var_12 = ','
    var_13 = bool(',' in var_11)
    assert var_13 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 21/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short content'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6
    var_8 = '# NOQA'
    var_9 = var_8 not in var_4
    var_10 = 10
    var_11 = 'line_length'
    var_12 = {var_11: var_10}
    var_13 = module_0.Config(**var_12)
    var_14 = 'this is a very long content'
    var_15 = len(var_14)
    var_16 = var_13.line_length
    var_17 = var_15 > var_16
    var_18 = var_8 not in var_14
    var_19 = 'line_length'
    var_20 = {var_19: var_10}
    var_21 = module_0.Config(**var_20)
    var_22 = 'this is a very long content # NOQA'
    var_23 = len(var_22)
    var_24 = var_21.line_length
    var_25 = var_23 > var_24
    var_26 = var_8 not in var_22



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_line_noqa_mode. Retrieved 3/9 statements.
# Partially parsed test_line_already_has_noqa. Retrieved 3/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = '\n'
    var_4 = module_1.line(var_2, var_3, var_1)
    assert var_4 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from some_very_long_module_name import function_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True
    var_13 = var_10.split(var_9)[var_2]
    var_14 = len(var_13)
    var_15 = bool(var_14 <= var_7.line_length + 10)
    assert var_15 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something  # comment'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('#' in var_8 or 'comment' in var_8)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some.very.long.module.path import name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 40
    var_1 = 'from very_long_module_name import something_else'
    var_2 = '\n'
    var_3 = 'NOQA'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something as very_long_alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'include_trailing_comma'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import a, b, c, d, e, f'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = bool(var_9 is not None)
    assert var_10 is True

def test_case_0():
    var_0 = 40
    var_1 = 'from module import something  # noqa'
    var_2 = '\n'

def test_case_0():
    var_0 = 40
    var_1 = True
    var_2 = 'from some_module import func1, func2, func3'
    var_3 = '\n'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from some_module import something_very_long'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('\\' in var_8 or var_8 == var_6)
    assert var_9 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module cimport something_long'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #77
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 105
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_import_statement_predicate_line_41_false. Retrieved 10/17 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 41 evaluates to False when len(lines) != line_count.'
    var_1 = True
    var_2 = 80
    var_3 = 'balanced_wrapping'
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = {var_3: var_1, var_4: var_2, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import '
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_9, var_10, var_11]
    var_13 = False



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_predicate_line_71_evaluates_to_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 100
    var_1 = 'short content'
    var_2 = len(var_1)
    var_3 = '# NOQA'
    var_4 = var_3 not in var_1
    var_5 = 10
    var_6 = 'this is a longer content'
    var_7 = len(var_6)
    var_8 = var_3 not in var_6
    var_9 = 'long content # NOQA'
    var_10 = len(var_9)
    var_11 = var_3 not in var_9



# Parsed testcases at query #80
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 50
    var_1 = 100
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 49
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True

import isort.settings as module_0

def test_case_0():
    var_0 = None
    var_1 = 80
    var_2 = 'wrap_length'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 79
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = var_5.wrap_length
    var_13 = var_5.line_length
    var_14 = var_12 or var_13
    var_15 = var_11 > var_14
    assert var_15 is True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_import_statement_empty_imports. Retrieved 3/6 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = module_0.import_statement(var_0, var_3)
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = '# comment1'
    var_5 = [var_4]
    var_6 = module_0.import_statement(var_0, var_3, var_5)
    var_7 = 'func1'
    var_8 = bool('func1' in var_6)
    assert var_8 is True
    var_9 = 'func2'
    var_10 = bool('func2' in var_6)
    assert var_10 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = ';'
    var_5 = module_0.import_statement(var_0, var_3, line_separator=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True
    var_8 = 'func2'
    var_9 = bool('func2' in var_5)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'func1'
    var_2 = 'func2'
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.import_statement(var_0, var_3, explode=var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_5)
    assert var_7 is True
    var_8 = 'func2'
    var_9 = bool('func2' in var_5)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = 'single_func'
    var_2 = [var_1]
    var_3 = module_0.import_statement(var_0, var_2)
    var_4 = 'single_func'
    var_5 = bool('single_func' in var_3)
    assert var_5 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import '
    var_1 = []
    var_2 = module_0.import_statement(var_0, var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = 'func1'
    var_10 = bool('func1' in var_8)
    assert var_10 is True
    var_11 = 'func2'
    var_12 = bool('func2' in var_8)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 'include_trailing_comma'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import '
    var_5 = 'func1'
    var_6 = 'func2'
    var_7 = [var_5, var_6]
    var_8 = module_1.import_statement(var_4, var_7, config=var_3)
    var_9 = 'func1'
    var_10 = bool('func1' in var_8)
    assert var_10 is True
    var_11 = 'func2'
    var_12 = bool('func2' in var_8)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = True
    var_1 = 40
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'very_long_function_name_one'
    var_8 = 'very_long_function_name_two'
    var_9 = [var_7, var_8]
    var_10 = module_1.import_statement(var_6, var_9, config=var_5)
    var_11 = 'very_long_function_name_one'
    var_12 = bool('very_long_function_name_one' in var_10)
    assert var_12 is True
    var_13 = 'very_long_function_name_two'
    var_14 = bool('very_long_function_name_two' in var_10)
    assert var_14 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_line_with_trailing_comma. Retrieved 7/9 statements.
# Partially parsed test_line_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_as_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 7/9 statements.
# Partially parsed test_line_noqa_comment_handling. Retrieved 7/9 statements.
# Partially parsed test_line_empty_line_parts. Retrieved 7/9 statements.
# Partially parsed test_line_wrap_length_config. Retrieved 8/10 statements.
# Partially parsed test_line_include_trailing_comma_with_comment. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 6
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name_that_exceeds_line_length'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = '# NOQA'
    var_10 = bool('# NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module  # important comment'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = '# important comment'
    var_12 = bool('# important comment' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very_long_package_name import module'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from package import item1, item2'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very.long.module.path import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import very_long_name as alias_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1  # short'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    assert var_6 == 'x = 1  # short'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from package import module_name'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = bool('\\' in var_9 or '\n' in var_9)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 2
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import item1, item2'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 25
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from package import module  # noqa'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 30
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os  # comment with # hash'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'os'
    var_10 = bool('os' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from x import y'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = 'use_parentheses'
    var_7 = 'multi_line_output'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from very_long_package_name import module_name'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from pkg import a  # c'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 9/12 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 3
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from some_module import very_long_name_one, very_long_name_two, very_long_name_three'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 10/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 0
    var_3 = 'include_trailing_comma'
    var_4 = 'use_parentheses'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from some_module import very_long_name_that_exceeds_line_length'
    var_10 = '\n'
    var_11 = var_8.include_trailing_comma
    var_12 = var_8.use_parentheses
    var_13 = ','



# Parsed testcases at query #85
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 150
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a'
    var_7 = 140
    var_8 = var_6 * var_7
    var_9 = len(var_8)
    var_10 = 2
    var_11 = var_9 + var_10
    var_12 = bool(var_11 > (var_5.wrap_length or var_5.line_length))
    assert var_12 is True
    var_13 = var_5.wrap_length or var_5.line_length
    assert var_13 is False



# Parsed testcases at query #86
#--------------------------




import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 100
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something'
    var_7 = '\n'
    var_8 = 'line_length'
    var_9 = 'wrap_length'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.Config(**var_10)
    var_12 = 'short'
    var_13 = module_1.line(var_12, var_7, var_11)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True



# Parsed testcases at query #87
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 150
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'short'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 8/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'short content'
    var_5 = len(var_4)
    var_6 = var_3.line_length
    var_7 = var_5 > var_6
    var_8 = '# NOQA'
    var_9 = var_8 not in var_4



# Parsed testcases at query #89
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 100
    var_1 = 120
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'short'
    var_7 = '\n'
    var_8 = len(var_6)
    var_9 = 2
    var_10 = var_8 + var_9
    var_11 = var_5.wrap_length
    var_12 = var_5.line_length
    var_13 = var_11 or var_12
    var_14 = var_10 > var_13
    assert var_14 is False



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_line_with_noqa_comment. Retrieved 6/8 statements.
# Partially parsed test_line_noqa_mode_adds_noqa_suffix. Retrieved 4/11 statements.
# Partially parsed test_line_vertical_hanging_indent_mode. Retrieved 4/10 statements.
# Partially parsed test_line_vertical_grid_grouped_mode. Retrieved 4/10 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'from module import something'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = 3
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = bool('NOQA' in var_8 or '\\' in var_8 or '(' in var_8)
    assert var_9 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'import x  # comment'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

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
    var_8 = 'from very_long_module_name import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)
    var_11 = 'import'
    var_12 = bool('import' in var_10)
    assert var_12 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module.submodule import item'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import very_long_name as alias_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
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
    var_6 = 'from module import something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
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
    var_7 = 'from module import something'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
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
    var_6 = 'from module import something  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'noqa'

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
    var_7 = 'from module import something  # comment'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True

def test_case_0():
    var_0 = 20
    var_1 = 'from very_long_module_name import something'
    var_2 = '\n'
    var_3 = len(var_1)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module cimport something'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'from module import something'
    var_3 = '\n'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_import_statement_predicate_line_41_evaluates_to_false. Retrieved 8/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = 'balanced_wrapping'
    var_3 = 'line_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from module import '
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_false. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 50
    var_1 = 'import os'
    var_2 = '\n'
    var_3 = 10
    var_4 = 'from module import something'
    var_5 = 'from module import something  # NOQA'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_line_long_content_with_noqa_mode_adds_noqa_comment. Retrieved 3/6 statements.
# Partially parsed test_line_long_content_with_noqa_mode_existing_noqa_returns_unchanged. Retrieved 3/6 statements.
# Partially parsed test_line_with_import_splitter_and_parentheses. Retrieved 4/7 statements.
# Partially parsed test_line_with_comment_preserves_comment. Retrieved 4/7 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 4/7 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_as_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 4/7 statements.
# Partially parsed test_line_without_parentheses_uses_backslash. Retrieved 4/7 statements.
# Partially parsed test_line_with_noqa_in_comment_preserves_formatting. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 4/7 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 4/7 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name_that_exceeds_line_length'
    var_1 = 50
    var_2 = '\n'
    var_3 = '# NOQA'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name # NOQA'
    var_1 = 50
    var_2 = '\n'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = '('
    var_5 = ')'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name # important comment'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = 'important comment'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = ','

def test_case_0():
    var_0 = 'some_module.some_very_long_submodule.some_very_long_function_name_exceeding_limit'
    var_1 = 40
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name as very_long_alias_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = 'as'

def test_case_0():
    var_0 = 'from some_very_long_module_name cimport some_very_long_function_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = False
    var_3 = '\n'
    var_4 = '\\'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name # noqa: E501'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = 'noqa'

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = '('

def test_case_0():
    var_0 = 'from some_very_long_module_name import some_very_long_function_name'
    var_1 = 40
    var_2 = True
    var_3 = '\n'
    var_4 = '('



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_line_with_import_splitter_and_parentheses. Retrieved 7/9 statements.
# Partially parsed test_line_with_dot_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_with_as_keyword. Retrieved 7/9 statements.
# Partially parsed test_line_with_trailing_comma_config. Retrieved 7/9 statements.
# Partially parsed test_line_with_vertical_hanging_indent_mode. Retrieved 7/9 statements.
# Partially parsed test_line_with_vertical_grid_grouped_mode. Retrieved 7/9 statements.
# Partially parsed test_line_with_cimport_splitter. Retrieved 7/9 statements.
# Partially parsed test_line_without_parentheses_uses_backslash. Retrieved 6/8 statements.
# Partially parsed test_line_with_wrap_length_config. Retrieved 8/10 statements.
# Partially parsed test_line_with_custom_comment_prefix. Retrieved 6/8 statements.
# Partially parsed test_line_with_multiple_hashes. Retrieved 7/9 statements.


import isort.wrap as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == 'import os'

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 4
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'NOQA'
    var_10 = bool('NOQA' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # some comment'
    var_5 = '\n'
    var_6 = module_1.line(var_4, var_5, var_3)
    var_7 = '# some comment'
    var_8 = bool('# some comment' in var_6)
    assert var_8 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from very.long.module import something'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function as fn'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'include_trailing_comma'
    var_6 = 'multi_line_output'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_1, var_6: var_2}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from module import function'
    var_10 = '\n'
    var_11 = module_1.line(var_9, var_10, var_8)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = 4
    var_2 = 'line_length'
    var_3 = 'multi_line_output'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name  # noqa'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)
    var_9 = 'noqa'
    var_10 = bool('noqa' in var_8)
    assert var_10 is True

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 2
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 3
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import function'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 15
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'cimport very_long_module_name'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = False
    var_2 = 'line_length'
    var_3 = 'use_parentheses'
    var_4 = 'multi_line_output'
    var_5 = {var_2: var_0, var_3: var_1, var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from module import function'
    var_8 = '\n'
    var_9 = module_1.line(var_7, var_8, var_6)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 80
    var_1 = 40
    var_2 = True
    var_3 = 0
    var_4 = 'line_length'
    var_5 = 'wrap_length'
    var_6 = 'use_parentheses'
    var_7 = 'multi_line_output'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from module import function'
    var_11 = '\n'
    var_12 = module_1.line(var_10, var_11, var_9)

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 10
    var_1 = ' #'
    var_2 = 'line_length'
    var_3 = 'comment_prefix'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import very_long_module_name'
    var_7 = '\n'
    var_8 = module_1.line(var_6, var_7, var_5)

import isort.wrap as module_0

def test_case_0():
    var_0 = ''
    var_1 = '\n'
    var_2 = module_0.line(var_0, var_1)
    assert var_2 == ''

import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 0
    var_3 = 'line_length'
    var_4 = 'use_parentheses'
    var_5 = 'multi_line_output'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Config(**var_6)
    var_8 = 'from module import func  # comment with # hash'
    var_9 = '\n'
    var_10 = module_1.line(var_8, var_9, var_7)



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_import_statement_predicate_line_41_false. Retrieved 27/42 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 41 evaluates to False.'
    var_1 = True
    var_2 = 'balanced_wrapping'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'from module import '
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = [var_6, var_7, var_8]
    var_10 = '\n'
    var_11 = 'from x import '
    var_12 = 'item1'
    var_13 = 'item2'
    var_14 = [var_12, var_13]
    var_15 = 80
    var_16 = 'balanced_wrapping'
    var_17 = 'line_length'
    var_18 = {var_16: var_1, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = 'from mod import '
    var_21 = 'x'
    var_22 = [var_21]
    var_23 = 10
    var_24 = 'balanced_wrapping'
    var_25 = 'line_length'
    var_26 = 'wrap_length'
    var_27 = {var_24: var_1, var_25: var_23, var_26: var_23}
    var_28 = module_0.Config(**var_27)
    var_29 = 'from package import '
    var_30 = 'func1'
    var_31 = 'func2'
    var_32 = 'func3'
    var_33 = [var_30, var_31, var_32]
    var_34 = False
    var_35 = 'balanced_wrapping'
    var_36 = {var_35: var_34}
    var_37 = module_0.Config(**var_36)



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 11/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 80
    var_2 = None
    var_3 = ' #'
    var_4 = 'include_trailing_comma'
    var_5 = 'use_parentheses'
    var_6 = 'line_length'
    var_7 = 'wrap_length'
    var_8 = 'comment_prefix'
    var_9 = {var_4: var_0, var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Config(**var_9)
    var_11 = 'from some_module import very_long_name_that_exceeds_line_length  # comment'
    var_12 = '\n'
    var_13 = 'from some_module import very_long_name_that_exceeds_line_length  '
    var_14 = var_10.include_trailing_comma
    var_15 = var_10.use_parentheses
    var_16 = ','



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_predicate_at_line_65_evaluates_to_false. Retrieved 13/21 statements.


import isort.settings as module_0
import isort.wrap as module_1

def test_case_0():
    var_0 = 40
    var_1 = 3
    var_2 = True
    var_3 = False
    var_4 = ' #'
    var_5 = 'line_length'
    var_6 = 'multi_line_output'
    var_7 = 'use_parentheses'
    var_8 = 'include_trailing_comma'
    var_9 = 'comment_prefix'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = 'from some_module import something_long'
    var_13 = '\n'
    var_14 = module_1.line(var_12, var_13, var_11)
    var_15 = var_11.comment_prefix
    var_16 = -1
    var_17 = -1
    var_18 = ')'



