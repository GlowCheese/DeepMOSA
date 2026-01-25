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
    var_0 = ''
    var_1 = []
    var_2 = 0
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_5)
    assert var_6 == ''

def test_case_4():
    var_0 = 'hello   '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello   \\'

def test_case_5():
    var_0 = 'hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'func'
    var_1 = 'from module import '
    var_2 = [var_0]
    var_3 = ' #'
    var_4 = False
    var_5 = 'imports'
    var_6 = 'statement'
    var_7 = 'comments'
    var_8 = 'remove_comments'
    var_9 = 'comment_prefix'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'include_trailing_comma'
    var_13 = 'line_length'
    var_14 = {var_5: var_2, var_6: var_1, var_7: var_2, var_8: var_4, var_9: var_3, var_10: var_3, var_11: var_10, var_12: var_4, var_13: var_4}
    var_15 = module_0._vertical_grid_common(var_4, **var_14)
    assert var_15 == 'from module import ( # func #line_separatorfunc'
    module_0.from_string(var_12)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'func'
    var_1 = [var_0, var_0]
    var_2 = 'from module import '
    var_3 = '    '
    var_4 = False
    var_5 = 'imports'
    var_6 = 'statement'
    var_7 = 'comments'
    var_8 = 'remove_comments'
    var_9 = 'comment_prefix'
    var_10 = 'indent'
    var_11 = module_0.formatter_from_string(var_10)
    var_12 = 'line_length'
    var_13 = {var_5: var_1, var_6: var_2, var_7: var_1, var_8: var_4, var_9: var_10, var_12: var_9, var_10: var_3, var_5: var_4, var_12: var_4}
    var_14 = module_0._vertical_grid_common(var_4, **var_13)
    assert var_14 == ''
    module_0.from_string(var_10)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ';?M.z-JXe`eZZ}fqt'
    var_1 = [var_0, var_0]
    var_2 = ' #'
    var_3 = '\n'
    var_4 = '    '
    var_5 = True
    var_6 = 'imports'
    var_7 = 'statement'
    var_8 = 'comments'
    var_9 = 'remove_comments'
    var_10 = 'comment_prefix'
    var_11 = 'line_separator'
    var_12 = 'indent'
    var_13 = 'include_trailing_comma'
    var_14 = module_0.formatter_from_string(var_12)
    var_15 = 'line_length'
    var_16 = {var_6: var_1, var_7: var_2, var_8: var_1, var_9: var_5, var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_15: var_5}
    var_17 = module_0._vertical_grid_common(var_5, **var_16)
    assert var_17 == ' #(\n    ;?M.z-JXe`eZZ}fqt,\n    ;?M.z-JXe`eZZ}fqt,'
    module_0.from_string(var_13)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'func'
    var_1 = [var_0, var_0]
    var_2 = '    '
    var_3 = False
    var_4 = 'imports'
    var_5 = 'statement'
    var_6 = 'comments'
    var_7 = 'remove_comments'
    var_8 = 'comment_prefix'
    var_9 = 'line_separator'
    var_10 = 'indent'
    var_11 = 'include_trailing_comma'
    var_12 = 'line_length'
    var_13 = {var_4: var_1, var_5: var_8, var_6: var_1, var_7: var_3, var_8: var_10, var_9: var_8, var_10: var_2, var_11: var_3, var_12: var_3}
    var_14 = module_0._vertical_grid_common(var_3, **var_13)
    assert var_14 == 'comment_prefix(indent funccomment_prefix    func,comment_prefix    func'
    module_0.from_string(var_11)

def test_case_10():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'from m import '
    var_5 = None
    var_6 = ''
    var_7 = '\n'
    var_8 = '    '
    var_9 = 79
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'comments'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'line_separator'
    var_16 = 'indent'
    var_17 = 'include_trailing_comma'
    var_18 = 'line_length'
    var_19 = {var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_0, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_0, var_18: var_9}
    var_20 = module_0._vertical_grid_common(var_0, **var_19)
    assert var_20 == 'from m import (\n    a, b'
    var_21 = bool('a' in var_20)
    assert var_21 is True
    var_22 = bool('b' in var_20)
    assert var_22 is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'func'
    var_1 = [var_0, var_0, var_0]
    var_2 = 'from module import '
    var_3 = '    '
    var_4 = False
    var_5 = 'imports'
    var_6 = 'statement'
    var_7 = 'comments'
    var_8 = 'remove_comments'
    var_9 = 'comment_prefix'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'include_trailing_comma'
    var_13 = module_0.formatter_from_string(var_11)
    var_14 = ''
    var_15 = {var_5: var_1, var_6: var_2, var_7: var_1, var_8: var_4, var_9: var_11, var_10: var_9, var_11: var_3, var_12: var_4, var_14: var_4}
    module_0._vertical_grid_common(var_4, **var_15)

def test_case_12():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = 'module1'
    var_10 = 'module2'
    var_11 = 'module3'
    var_12 = [var_9, var_10, var_11]
    var_13 = None
    var_14 = False
    var_15 = ' #'
    var_16 = '\n'
    var_17 = '    '
    var_18 = 'from package import ('
    var_19 = 79
    var_20 = {var_0: var_12, var_1: var_13, var_2: var_14, var_3: var_15, var_4: var_16, var_5: var_17, var_6: var_18, var_7: var_14, var_8: var_19}
    var_21 = True
    var_22 = 'imports'
    var_23 = 'comments'
    var_24 = 'remove_comments'
    var_25 = 'comment_prefix'
    var_26 = 'line_separator'
    var_27 = 'indent'
    var_28 = 'statement'
    var_29 = 'include_trailing_comma'
    var_30 = 'line_length'
    var_31 = {var_22: var_12, var_23: var_13, var_24: var_14, var_25: var_15, var_26: var_16, var_27: var_17, var_28: var_18, var_29: var_14, var_30: var_19}
    var_32 = module_0._vertical_grid_common(var_21, **var_31)
    assert var_32 == 'from package import ((\n    module1, module2, module3'
    var_33 = bool('module1' in var_32)
    assert var_33 is True
    var_34 = bool('module2' in var_32)
    assert var_34 is True
    var_35 = bool('module3' in var_32)
    assert var_35 is True
    var_36 = var_20[var_0]
    var_37 = len(var_36)
    assert var_37 == 0