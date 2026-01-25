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
    var_0 = ''
    var_1 = []
    var_2 = 0
    var_3 = []
    var_4 = True
    var_5 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_4)
    assert var_5 == ''

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'include_trailing_comma'
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
    assert var_19 == '(\n    include_trailing_comma'
    module_0.vertical_hanging_indent(var_7, var_1, var_7, var_7, var_7, var_1, var_1, var_7, var_7, var_7)

def test_case_5():
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
def test_case_6():
    var_0 = 'statement'
    var_1 = [var_0]
    var_2 = ''
    var_3 = '\n'
    var_4 = '    '
    var_5 = True
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
    var_17 = {var_8: var_1, var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_8, var_14: var_6, var_15: var_7, var_16: var_5}
    var_18 = module_0._vertical_grid_common(var_5, **var_17)
    assert var_18 == '(\n    statement,'
    module_0.from_string(var_6)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'statement'
    var_1 = 'AA7ij6Nf\nxRm'
    var_2 = 'import os'
    var_3 = [var_2, var_2, var_1]
    var_4 = ''
    var_5 = 'K'
    var_6 = ' \r Q'
    var_7 = False
    var_8 = 'Z # '
    var_9 = None
    var_10 = 'imports'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'comments'
    var_16 = 'line_length'
    var_17 = 'include_trailing_comma'
    var_18 = {var_10: var_3, var_0: var_4, var_11: var_5, var_12: var_6, var_13: var_7, var_14: var_8, var_15: var_9, var_16: var_7, var_17: var_7}
    var_19 = module_0._vertical_grid_common(var_7, **var_18)
    assert var_19 == '(K \r Qimport os,K \r Qimport os,K \r QAA7ij6Nf\nxRm'
    module_0.from_string(var_8)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'AA7ij6Nf\nxRm'
    var_1 = 'import os'
    var_2 = [var_1, var_1, var_0]
    var_3 = '?'
    var_4 = ' \r Q'
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
    var_18 = {var_9: var_2, var_10: var_3, var_11: var_15, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_5}
    var_19 = module_0._vertical_grid_common(var_5, **var_18)
    assert var_19 == '?(comments \r Qimport os, import os, AA7ij6Nf\nxRm'
    module_0.from_string(var_7)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'AA7ij6Nf\nxRm'
    var_1 = 'import os'
    var_2 = [var_1, var_1, var_0]
    var_3 = ''
    var_4 = '\n'
    var_5 = ' \r Q'
    var_6 = True
    var_7 = '  # '
    var_8 = None
    var_9 = 88
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'remove_comments'
    var_15 = 'comment_prefix'
    var_16 = 'comments'
    var_17 = 'line_length'
    var_18 = 'include_trailing_comma'
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_9, var_18: var_6}
    var_20 = module_0._vertical_grid_common(var_6, **var_19)
    assert var_20 == '(\n \r Qimport os, import os, AA7ij6Nf\nxRm,'
    module_0.from_string(var_12)

def test_case_10():
    var_0 = 'Hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'Hello \\'

def test_case_11():
    var_0 = 'Hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'Hello \\'