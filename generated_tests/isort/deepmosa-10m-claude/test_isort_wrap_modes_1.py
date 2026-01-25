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
    var_0 = 'test   '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test   \\'

def test_case_4():
    var_0 = 'test'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_5():
    var_0 = ''
    var_1 = []
    var_2 = 0
    var_3 = []
    var_4 = False
    var_5 = False
    var_6 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_5)
    assert var_6 == ''

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = False
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = None
    var_4 = ''
    var_5 = '    '
    var_6 = 'imports'
    var_7 = 'statement'
    var_8 = 'comments'
    var_9 = 'remove_comments'
    var_10 = 'comment_prefix'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'jV.7^t14'
    var_14 = {var_6: var_2, var_7: var_7, var_8: var_3, var_9: var_0, var_10: var_4, var_11: var_13, var_12: var_5, var_5: var_0, var_13: var_0}
    module_0._vertical_grid_common(var_0, **var_14)

def test_case_7():
    var_0 = True
    var_1 = []
    var_2 = 'from module import'
    var_3 = None
    var_4 = False
    var_5 = ''
    var_6 = '\n'
    var_7 = '    '
    var_8 = 80
    var_9 = 'imports'
    var_10 = 'statement'
    var_11 = 'comments'
    var_12 = 'remove_comments'
    var_13 = 'comment_prefix'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'include_trailing_comma'
    var_17 = 'line_length'
    var_18 = {var_9: var_1, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_4, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_0, **var_18)
    assert var_19 == ''

def test_case_8():
    var_0 = True
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = 'from module import'
    var_4 = None
    var_5 = False
    var_6 = ''
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
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_0, var_18: var_9}
    var_20 = module_0._vertical_grid_common(var_0, **var_19)
    assert var_20 == 'from module import(\n    foo,'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = False
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = None
    var_4 = '8\n'
    var_5 = '    '
    var_6 = 'imports'
    var_7 = 'statement'
    var_8 = 'comments'
    var_9 = 'remove_comments'
    var_10 = 'comment_prefix'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'include_trailing_comma'
    var_14 = 'PV.o^t 4'
    var_15 = {var_6: var_2, var_7: var_7, var_8: var_3, var_9: var_0, var_10: var_1, var_11: var_4, var_12: var_5, var_13: var_0, var_14: var_0}
    var_16 = module_0._vertical_grid_common(var_0, **var_15)
    assert var_16 == 'statement(8\n    foo'
    module_0.from_string(var_12)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = False
    var_1 = 'foo'
    var_2 = [var_1, var_1]
    var_3 = None
    var_4 = 'v?'
    var_5 = '8\n'
    var_6 = '    '
    var_7 = 'imports'
    var_8 = 'statement'
    var_9 = 'comments'
    var_10 = 'remove_comments'
    var_11 = 'comment_prefix'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'include_trailing_comma'
    var_15 = 'PV.o^t 4'
    var_16 = {var_7: var_2, var_8: var_8, var_9: var_3, var_10: var_0, var_11: var_4, var_12: var_5, var_13: var_6, var_14: var_0, var_15: var_0}
    module_0._vertical_grid_common(var_0, **var_16)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = True
    var_1 = 'foo'
    var_2 = [var_1, var_1]
    var_3 = 'from module import'
    var_4 = None
    var_5 = '\n'
    var_6 = '    '
    var_7 = 80
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'comments'
    var_11 = 'remove_comments'
    var_12 = 'comment_prefix'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'include_trailing_comma'
    var_16 = {var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_0, var_12: var_3, var_13: var_5, var_14: var_6, var_15: var_0, var_11: var_7}
    module_0._vertical_grid_common(var_0, **var_16)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = False
    var_1 = 'foo'
    var_2 = [var_1, var_1]
    var_3 = 'from module import'
    var_4 = None
    var_5 = ''
    var_6 = '    '
    var_7 = 80
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'comments'
    var_11 = 'remove_comments'
    var_12 = 'comment_prefix'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'include_trailing_comma'
    var_16 = 'line_length'
    var_17 = {var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_0, var_12: var_5, var_13: var_15, var_14: var_6, var_15: var_0, var_16: var_7}
    var_18 = module_0._vertical_grid_common(var_0, **var_17)
    assert var_18 == 'from module import(include_trailing_comma    foo, foo'
    module_0.from_string(var_4)

def test_case_13():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = False
    var_5 = ' #'
    var_6 = '\n'
    var_7 = '    '
    var_8 = 'from module import ('
    var_9 = 80
    var_10 = True
    var_11 = 'imports'
    var_12 = 'comments'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'line_separator'
    var_16 = 'indent'
    var_17 = 'statement'
    var_18 = 'include_trailing_comma'
    var_19 = 'line_length'
    var_20 = {var_11: var_2, var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_18: var_4, var_19: var_9}
    var_21 = module_0._vertical_grid_common(var_10, **var_20)
    assert var_21 == 'from module import ((\n    os, sys'
    var_22 = bool(var_21 is not None)
    assert var_22 is True
    var_23 = [var_0]
    var_24 = 'imports'
    var_25 = 'comments'
    var_26 = 'remove_comments'
    var_27 = 'comment_prefix'
    var_28 = 'line_separator'
    var_29 = 'indent'
    var_30 = 'statement'
    var_31 = 'include_trailing_comma'
    var_32 = 'line_length'
    var_33 = {var_24: var_23, var_25: var_3, var_26: var_4, var_27: var_5, var_28: var_6, var_29: var_7, var_30: var_8, var_31: var_10, var_32: var_9}
    var_34 = module_0._vertical_grid_common(var_10, **var_33)
    assert var_34 == 'from module import ((\n    os,'
    var_35 = bool(var_34 is not None)
    assert var_35 is True
    var_36 = 'json'
    var_37 = [var_0, var_1, var_36]
    var_38 = 'imports'
    var_39 = 'comments'
    var_40 = 'remove_comments'
    var_41 = 'comment_prefix'
    var_42 = 'line_separator'
    var_43 = 'indent'
    var_44 = 'statement'
    var_45 = 'include_trailing_comma'
    var_46 = 'line_length'
    var_47 = {var_38: var_37, var_39: var_3, var_40: var_4, var_41: var_5, var_42: var_6, var_43: var_7, var_44: var_8, var_45: var_10, var_46: var_9}
    var_48 = module_0._vertical_grid_common(var_10, **var_47)
    assert var_48 == 'from module import ((\n    os, sys, json,'
    var_49 = bool(var_48 is not None)
    assert var_49 is True

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = False
    var_1 = 'foo'
    var_2 = [var_1, var_1]
    var_3 = 'from module import'
    var_4 = None
    var_5 = ''
    var_6 = '\n'
    var_7 = '    '
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'comments'
    var_11 = 'remove_comments'
    var_12 = 'comment_prefix'
    var_13 = 'line_separator'
    var_14 = 'indent'
    var_15 = 'include_trailing_comma'
    var_16 = 'line_length'
    var_17 = {var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_0, var_12: var_5, var_13: var_6, var_14: var_7, var_15: var_0, var_16: var_0}
    var_18 = module_0._vertical_grid_common(var_0, **var_17)
    assert var_18 == 'from module import(\n    foo,\n    foo'
    module_0.from_string(var_4)