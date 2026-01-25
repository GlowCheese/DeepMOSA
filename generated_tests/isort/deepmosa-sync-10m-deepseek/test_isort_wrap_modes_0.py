# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


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
    var_0 = ' '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_4():
    var_0 = ''
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_5():
    var_0 = ''
    var_1 = []
    var_2 = 0
    var_3 = []
    var_4 = False
    var_5 = False
    var_6 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_5)
    assert var_6 == ''

def test_case_6():
    var_0 = 'line_separator'
    var_1 = 'include_trailing_comma'
    var_2 = 'remove_comments'
    var_3 = 'import os'
    var_4 = [var_2, var_0, var_3]
    var_5 = '\n'
    var_6 = 80
    var_7 = False
    var_8 = None
    var_9 = 'imports'
    var_10 = 'statement'
    var_11 = 'indent'
    var_12 = 'line_length'
    var_13 = 'include_trailing_comma'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = {var_9: var_4, var_10: var_1, var_0: var_5, var_11: var_1, var_12: var_6, var_13: var_7, var_14: var_8, var_15: var_7, var_16: var_16}
    var_18 = module_0._vertical_grid_common(var_7, **var_17)
    assert var_18 == 'include_trailing_comma(\ninclude_trailing_commaremove_comments, line_separator, import os'

def test_case_7():
    var_0 = []
    var_1 = ''
    var_2 = '\n'
    var_3 = '    '
    var_4 = 80
    var_5 = False
    var_6 = None
    var_7 = '#'
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'line_length'
    var_13 = 'include_trailing_comma'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_5, var_16: var_7}
    var_18 = module_0._vertical_grid_common(var_5, **var_17)
    assert var_18 == ''

def test_case_8():
    var_0 = 'remove_comments'
    var_1 = 'import os'
    var_2 = [var_1]
    var_3 = 'from x import'
    var_4 = '4{p&M6;g<z'
    var_5 = 80
    var_6 = True
    var_7 = None
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'line_lengt<h'
    var_13 = 'include_trailing_comma'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = {var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_10, var_12: var_5, var_13: var_6, var_14: var_7, var_15: var_6, var_16: var_0}
    var_18 = module_0._vertical_grid_common(var_6, **var_17)
    assert var_18 == 'from x import(4{p&M6;g<zline_separatorimport os,'
    var_19 = 'from x import (\n    import os)'
    var_20 = bool(var_18 == var_19)

def test_case_9():
    var_0 = 'statement'
    var_1 = 'line_separator'
    var_2 = 'include_trailing_comma'
    var_3 = 'import os'
    var_4 = [var_0, var_0, var_3]
    var_5 = '\n'
    var_6 = 80
    var_7 = True
    var_8 = None
    var_9 = '#'
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'indent'
    var_13 = 'line_length'
    var_14 = 'include_trailing_comma'
    var_15 = 'comments'
    var_16 = 'remove_comments'
    var_17 = 'comment_prefix'
    var_18 = {var_10: var_4, var_11: var_2, var_1: var_5, var_12: var_2, var_13: var_6, var_14: var_7, var_15: var_8, var_16: var_7, var_17: var_9}
    var_19 = module_0._vertical_grid_common(var_7, **var_18)
    assert var_19 == 'include_trailing_comma(\ninclude_trailing_commastatement, statement, import os,'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'line_length'
    var_4 = '[U^j"^:+a*|x!U>eX'
    var_5 = 'import sys'
    var_6 = [var_4, var_5]
    var_7 = 'from x import'
    var_8 = '\n'
    var_9 = False
    var_10 = None
    var_11 = "'&+N-aZt"
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_7: var_9}
    var_13 = 'imports'
    var_14 = 'statement'
    var_15 = 'line_separator'
    var_16 = 'indent'
    var_17 = 'line_length'
    var_18 = 'include_trailing_comma'
    var_19 = 'comments'
    var_20 = 'remove_comments'
    var_21 = 'comment_prefix'
    var_22 = {var_13: var_6, var_14: var_7, var_15: var_8, var_16: var_21, var_17: var_9, var_18: var_9, var_19: var_10, var_20: var_9, var_21: var_11}
    var_23 = module_0._vertical_grid_common(var_10, **var_22)
    assert var_23 == 'from x import(\ncomment_prefix[U^j"^:+a*|x!U>eX,\ncomment_prefiximport sys'
    module_0.from_string(var_12)