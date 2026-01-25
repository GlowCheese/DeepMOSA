# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '^J"|f\rROJ!$=c'
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = 'n-Q>V\\{%&'
    var_1 = module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = 'Hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'Hello \\'

def test_case_4():
    var_0 = 'Hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'Hello \\'

def test_case_5():
    var_0 = ''
    var_1 = []
    var_2 = 0
    var_3 = []
    var_4 = True
    var_5 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_4)
    assert var_5 == ''

def test_case_6():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = ''
    var_3 = '\n'
    var_4 = '    '
    var_5 = False
    var_6 = '  # '
    var_7 = 88
    var_8 = None
    var_9 = 'imports'
    var_10 = 'statement'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'line_length'
    var_16 = 'include_trailing_comma'
    var_17 = 'comments'
    var_18 = {var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_5, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_5, **var_18)
    assert var_19 == '(\n    import os'

def test_case_7():
    var_0 = []
    var_1 = ''
    var_2 = '\n'
    var_3 = '    '
    var_4 = False
    var_5 = '  # '
    var_6 = 88
    var_7 = None
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'remove_comments'
    var_13 = 'comment_prefix'
    var_14 = 'line_length'
    var_15 = 'include_trailing_comma'
    var_16 = 'comments'
    var_17 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_4, var_16: var_7}
    var_18 = module_0._vertical_grid_common(var_4, **var_17)
    assert var_18 == ''

def test_case_8():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = ''
    var_4 = '\n'
    var_5 = '    '
    var_6 = False
    var_7 = '  # '
    var_8 = 88
    var_9 = None
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'include_trailing_comma'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'line_length'
    var_18 = 'comments'
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9}
    var_20 = module_0._vertical_grid_common(var_6, **var_19)
    assert var_20 == '(\n    os, sys'

def test_case_9():
    var_0 = 'import1'
    var_1 = 'import2'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = False
    var_5 = ''
    var_6 = '\n'
    var_7 = '    '
    var_8 = 100
    var_9 = True
    var_10 = 'imports'
    var_11 = 'comments'
    var_12 = 'remove_comments'
    var_13 = 'comment_prefix'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'include_trailing_comma'
    var_17 = 'line_length'
    var_18 = 'statement'
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_4, var_17: var_8, var_18: var_5}
    var_20 = module_0._vertical_grid_common(var_9, **var_19)
    assert var_20 == '(\n    import1, import2'
    var_21 = bool(var_20)
    assert var_21 is True

def test_case_10():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = ''
    var_4 = '\n'
    var_5 = '    '
    var_6 = True
    var_7 = False
    var_8 = '  # '
    var_9 = 88
    var_10 = None
    var_11 = 'imports'
    var_12 = 'statement'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'include_trailing_comma'
    var_16 = 'remove_comments'
    var_17 = 'comment_prefix'
    var_18 = 'line_length'
    var_19 = 'comments'
    var_20 = {var_11: var_2, var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_10}
    var_21 = module_0._vertical_grid_common(var_7, **var_20)
    assert var_21 == '(\n    os, sys,'

def test_case_11():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'very_long_module_name'
    var_3 = [var_0, var_1, var_2]
    var_4 = ''
    var_5 = '\n'
    var_6 = '    '
    var_7 = False
    var_8 = '  # '
    var_9 = 20
    var_10 = None
    var_11 = 'imports'
    var_12 = 'statement'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'include_trailing_comma'
    var_16 = 'remove_comments'
    var_17 = 'comment_prefix'
    var_18 = 'line_length'
    var_19 = 'comments'
    var_20 = {var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_10}
    var_21 = module_0._vertical_grid_common(var_7, **var_20)
    assert var_21 == '(\n    os, sys,\n    very_long_module_name'