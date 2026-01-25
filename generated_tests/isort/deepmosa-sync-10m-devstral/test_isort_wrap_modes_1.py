# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.formatter_from_string(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.from_string(var_0)

def test_case_3():
    var_0 = 'CLAMP'
    var_1 = None
    var_2 = True
    var_3 = module_0._wrap_mode_interface(var_1, var_1, var_0, var_0, var_1, var_1, var_1, var_1, var_2, var_1)
    assert var_3 == ''

def test_case_4():
    var_0 = 'test '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_5():
    var_0 = 'test'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'imports'
    var_1 = 'line_separator'
    var_2 = 'indent'
    var_3 = [var_1]
    var_4 = ''
    var_5 = '\n'
    var_6 = '    '
    var_7 = True
    var_8 = '  # '
    var_9 = 88
    var_10 = '3'
    var_11 = module_0.from_string(var_10)
    assert var_11 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.GRID == module_0.WrapModes.GRID
    assert module_0.WrapModes.VERTICAL == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.HANGING_INDENT == module_0.WrapModes.HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_GRID == module_0.WrapModes.VERTICAL_GRID
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED == module_0.WrapModes.VERTICAL_GRID_GROUPED
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    assert module_0.WrapModes.NOQA == module_0.WrapModes.NOQA
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    assert module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    assert module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    assert module_0.WrapModes.BACKSLASH_GRID == module_0.WrapModes.BACKSLASH_GRID
    var_12 = 'imports'
    var_13 = 'statement'
    var_14 = 'remove_comments'
    var_15 = 'comment_prefix'
    var_16 = 'comments'
    var_17 = 'include_trailing_comma'
    var_18 = {var_12: var_3, var_13: var_4, var_17: var_5, var_2: var_6, var_14: var_7, var_15: var_8, var_16: var_3, var_0: var_9, var_17: var_7}
    module_0._vertical_grid_common(var_16, **var_18)
    assert var_19 == '(    os'

def test_case_7():
    var_0 = []
    var_1 = ''
    var_2 = '\n'
    var_3 = '    '
    var_4 = False
    var_5 = '  # '
    var_6 = None
    var_7 = 88
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'remove_comments'
    var_13 = 'comment_prefix'
    var_14 = 'comments'
    var_15 = 'line_length'
    var_16 = 'include_trailing_comma'
    var_17 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_4}
    var_18 = module_0._vertical_grid_common(var_4, **var_17)
    assert var_18 == ''

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'line_separator'
    var_1 = 'indent'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_length'
    var_5 = 'include_trailing_comma'
    var_6 = 'os'
    var_7 = [var_6]
    var_8 = ''
    var_9 = ''
    var_10 = '    '
    var_11 = True
    var_12 = '  # '
    var_13 = '3'
    var_14 = module_0.from_string(var_13)
    assert var_14 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.GRID == module_0.WrapModes.GRID
    assert module_0.WrapModes.VERTICAL == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.HANGING_INDENT == module_0.WrapModes.HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_GRID == module_0.WrapModes.VERTICAL_GRID
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED == module_0.WrapModes.VERTICAL_GRID_GROUPED
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    assert module_0.WrapModes.NOQA == module_0.WrapModes.NOQA
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    assert module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    assert module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    assert module_0.WrapModes.BACKSLASH_GRID == module_0.WrapModes.BACKSLASH_GRID
    var_15 = True
    var_16 = {var_4: var_7, var_0: var_8, var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_6: var_7, var_4: var_11, var_5: var_15}
    var_17 = 'imports'
    var_18 = 'statement'
    var_19 = 'line_separator'
    var_20 = 'indent'
    var_21 = 'remove_comments'
    var_22 = 'comment_prefix'
    var_23 = 'comments'
    var_24 = {var_17: var_7, var_18: var_8, var_19: var_9, var_20: var_10, var_21: var_11, var_22: var_12, var_23: var_7, var_3: var_16, var_2: var_15}
    module_0._vertical_grid_common(var_15, **var_24)
    assert var_25 == '(    os'

def test_case_9():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = ''
    var_3 = '\n'
    var_4 = '    '
    var_5 = False
    var_6 = '  # '
    var_7 = None
    var_8 = 88
    var_9 = 'imports'
    var_10 = 'statement'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'comments'
    var_16 = 'line_length'
    var_17 = 'include_trailing_comma'
    var_18 = {var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_5}
    var_19 = module_0._vertical_grid_common(var_5, **var_18)
    assert var_19 == '(\n    os'

def test_case_10():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'remove_comments'
    var_5 = 'comment_prefix'
    var_6 = 'comments'
    var_7 = 'line_length'
    var_8 = 'include_trailing_comma'
    var_9 = 'os'
    var_10 = [var_9]
    var_11 = ''
    var_12 = ''
    var_13 = '    '
    var_14 = True
    var_15 = 'Comment 1'
    var_16 = [var_15, var_11]
    var_17 = 88
    var_18 = '3'
    var_19 = module_0.from_string(var_18)
    assert var_19 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.GRID == module_0.WrapModes.GRID
    assert module_0.WrapModes.VERTICAL == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.HANGING_INDENT == module_0.WrapModes.HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_GRID == module_0.WrapModes.VERTICAL_GRID
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED == module_0.WrapModes.VERTICAL_GRID_GROUPED
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    assert module_0.WrapModes.NOQA == module_0.WrapModes.NOQA
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    assert module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    assert module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    assert module_0.WrapModes.BACKSLASH_GRID == module_0.WrapModes.BACKSLASH_GRID
    var_20 = True
    var_21 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_3, var_6: var_16, var_7: var_17, var_8: var_20}
    var_22 = module_0._vertical_grid_common(var_20, **var_21)
    assert var_22 == '(    os,'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'remove_comments'
    var_5 = 'comment_prefix'
    var_6 = 'comments'
    var_7 = '\n'
    var_8 = '    '
    var_9 = '  # '
    var_10 = 'Comment 1'
    var_11 = [var_10, var_10]
    var_12 = 88
    var_13 = '3'
    var_14 = module_0.from_string(var_13)
    assert var_14 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.GRID == module_0.WrapModes.GRID
    assert module_0.WrapModes.VERTICAL == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.HANGING_INDENT == module_0.WrapModes.HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_GRID == module_0.WrapModes.VERTICAL_GRID
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED == module_0.WrapModes.VERTICAL_GRID_GROUPED
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    assert module_0.WrapModes.NOQA == module_0.WrapModes.NOQA
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    assert module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    assert module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    assert module_0.WrapModes.BACKSLASH_GRID == module_0.WrapModes.BACKSLASH_GRID
    var_15 = {var_0: var_11, var_1: var_1, var_2: var_7, var_3: var_8, var_5: var_9, var_4: var_2, var_5: var_9, var_6: var_11, var_9: var_12, var_9: var_2}
    var_16 = None
    module_0._vertical_grid_common(var_16, **var_15)

def test_case_12():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = ''
    var_4 = '\n'
    var_5 = '    '
    var_6 = False
    var_7 = '  # '
    var_8 = None
    var_9 = 88
    var_10 = True
    var_11 = 'imports'
    var_12 = 'statement'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'comments'
    var_18 = 'line_length'
    var_19 = 'include_trailing_comma'
    var_20 = {var_11: var_2, var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_10}
    var_21 = module_0._vertical_grid_common(var_6, **var_20)
    assert var_21 == '(\n    os, sys,'

def test_case_13():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = ''
    var_4 = '\n'
    var_5 = '    '
    var_6 = False
    var_7 = '  # '
    var_8 = None
    var_9 = 10
    var_10 = True
    var_11 = 'imports'
    var_12 = 'statement'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'comments'
    var_18 = 'line_length'
    var_19 = 'include_trailing_comma'
    var_20 = {var_11: var_2, var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_6}
    var_21 = module_0._vertical_grid_common(var_10, **var_20)
    assert var_21 == '(\n    os,\n    sys'

def test_case_14():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'json'
    var_3 = 'datetime'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = ''
    var_6 = '\n'
    var_7 = '    '
    var_8 = False
    var_9 = '  # '
    var_10 = None
    var_11 = 20
    var_12 = 'imports'
    var_13 = 'statement'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'remove_comments'
    var_17 = 'comment_prefix'
    var_18 = 'comments'
    var_19 = 'line_length'
    var_20 = 'include_trailing_comma'
    var_21 = {var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_9, var_18: var_10, var_19: var_11, var_20: var_8}
    var_22 = module_0._vertical_grid_common(var_8, **var_21)
    assert var_22 == '(\n    os, sys, json,\n    datetime'