# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


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
    var_4 = False
    var_5 = True
    var_6 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_5)
    assert var_6 == ''

def test_case_4():
    var_0 = 'test '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_5():
    var_0 = ''
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_6():
    var_0 = 'comments'
    var_1 = 'import os'
    var_2 = [var_1]
    var_3 = 'from x import'
    var_4 = 'fN\n'
    var_5 = '    '
    var_6 = 80
    var_7 = True
    var_8 = [var_0]
    var_9 = False
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'remove_comments'
    var_15 = 'comments'
    var_16 = 'comment_prefix'
    var_17 = 'include_trailing_comma'
    var_18 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_5: var_6, var_14: var_7, var_15: var_8, var_16: var_0, var_17: var_9}
    var_19 = module_0._vertical_grid_common(var_7, **var_18)
    assert var_19 == 'from x import(fN\n    import os'

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
    var_13 = 'remove_comments'
    var_14 = 'comments'
    var_15 = 'comment_prefix'
    var_16 = 'include_trailing_comma'
    var_17 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_5}
    var_18 = module_0._vertical_grid_common(var_5, **var_17)
    assert var_18 == ''

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'import a'
    var_1 = 'import b'
    var_2 = [var_0, var_1]
    var_3 = ''
    var_4 = '\n'
    var_5 = '    '
    var_6 = True
    var_7 = False
    var_8 = None
    var_9 = 'imports'
    var_10 = 'statement'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'include_trailing_comma'
    var_14 = 'remove_comments'
    var_15 = 'comments'
    var_16 = 'comment_prefix'
    var_17 = {var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_6, var_14: var_7, var_15: var_8, var_16: var_3}
    module_0._vertical_grid_common(var_7, **var_17)

def test_case_9():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = 'from x import'
    var_4 = '\n'
    var_5 = '    '
    var_6 = 80
    var_7 = False
    var_8 = None
    var_9 = '#'
    var_10 = True
    var_11 = 'imports'
    var_12 = 'statement'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'line_length'
    var_16 = 'remove_comments'
    var_17 = 'comments'
    var_18 = 'comment_prefix'
    var_19 = 'include_trailing_comma'
    var_20 = {var_11: var_2, var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_10}
    var_21 = module_0._vertical_grid_common(var_10, **var_20)
    assert var_21 == 'from x import(\n    import os, import sys,'
    var_22 = 'from x import(\n    import os, import sys,)'
    var_23 = bool(var_21 == var_22)

def test_case_10():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import'
    var_4 = None
    var_5 = False
    var_6 = '#'
    var_7 = '\n'
    var_8 = '    '
    var_9 = 80
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'comments'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'line_separator'
    var_16 = 'indent'
    var_17 = 'include_trailing_comma'
    var_18 = 'line_length'
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_5, var_18: var_9}
    var_20 = module_0._vertical_grid_common(var_5, **var_19)
    assert var_20 == 'import(\n    os, sys'
    var_21 = 'import(\n    os, sys)'
    var_22 = bool(var_20 == var_21)

def test_case_11():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'json'
    var_3 = 'math'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'import'
    var_6 = None
    var_7 = False
    var_8 = '#'
    var_9 = '\n'
    var_10 = '    '
    var_11 = 20
    var_12 = True
    var_13 = 'imports'
    var_14 = 'statement'
    var_15 = 'comments'
    var_16 = 'remove_comments'
    var_17 = 'comment_prefix'
    var_18 = 'line_separator'
    var_19 = 'indent'
    var_20 = 'include_trailing_comma'
    var_21 = 'line_length'
    var_22 = {var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_9, var_19: var_10, var_20: var_7, var_21: var_11}
    var_23 = module_0._vertical_grid_common(var_12, **var_22)
    assert var_23 == 'import(\n    os, sys, json,\n    math'
    var_24 = 'import(\n    os, sys,\n    json, math)'
    var_25 = bool(var_23 == var_24)