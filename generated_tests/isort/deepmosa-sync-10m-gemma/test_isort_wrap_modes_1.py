# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'SOME_EXIS]TING_ATTR'
    module_0.from_string(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Ea5+M\n1a\t+I1/d{+'
    var_1 = module_0.formatter_from_string(var_0)
    var_2 = None
    var_3 = '<g8'
    var_4 = '"a`^'
    var_5 = 'm2\x0b_ap[*#-!o1V8m1'
    module_0.vertical_grid(var_0, var_2, var_3, var_4, var_2, var_2, var_2, var_5, var_2, var_2)

def test_case_3():
    var_0 = 'hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

def test_case_4():
    var_0 = 'hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'include_trailing_comma'
    var_1 = '6y/$o}0SxMSryY)j0['
    var_2 = '2khNVNgu\\h\tbL'
    var_3 = [var_2, var_2]
    var_4 = []
    var_5 = True
    var_6 = 100
    var_7 = True
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'comments'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'line_length'
    var_16 = {var_8: var_3, var_9: var_13, var_10: var_11, var_11: var_1, var_12: var_4, var_13: var_5, var_14: var_14, var_0: var_5, var_15: var_6}
    var_17 = module_0._vertical_grid_common(var_7, **var_16)
    assert var_17 == 'remove_comments(indent6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL, 2khNVNgu\\h\tbL,'
    var_18 = None
    module_0.from_string(var_18)

def test_case_6():
    var_0 = False
    var_1 = []
    var_2 = 'import'
    var_3 = '\n'
    var_4 = '    '
    var_5 = 'imports'
    var_6 = 'statement'
    var_7 = 'line_separator'
    var_8 = 'indent'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0._vertical_grid_common(var_0, **var_9)
    assert var_10 == ''

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'statement'
    var_1 = [var_0]
    var_2 = '\n'
    var_3 = '  '
    var_4 = False
    var_5 = ''
    var_6 = 100
    var_7 = False
    var_8 = 'imports'
    var_9 = 'statement'
    var_10 = 'line_separator'
    var_11 = 'indent'
    var_12 = 'comments'
    var_13 = 'remove_comments'
    var_14 = 'comment_prefix'
    var_15 = 'include_trailing_comma'
    var_16 = 'line_length'
    var_17 = {var_8: var_1, var_9: var_9, var_10: var_2, var_11: var_3, var_12: var_1, var_13: var_4, var_14: var_5, var_15: var_4, var_16: var_6}
    var_18 = module_0._vertical_grid_common(var_7, **var_17)
    assert var_18 == 'statement( statement\n  statement'
    var_19 = None
    module_0.from_string(var_19)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'include_trailing_comma'
    var_1 = '6y/$o}0SxMSryY)j0['
    var_2 = '2khNVNgu\\h\tbL'
    var_3 = [var_2, var_2]
    var_4 = 'from '
    var_5 = []
    var_6 = False
    var_7 = ''
    var_8 = 100
    var_9 = True
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'line_length'
    var_18 = {var_10: var_3, var_11: var_4, var_12: var_10, var_13: var_1, var_14: var_5, var_15: var_6, var_16: var_7, var_0: var_6, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_9, **var_18)
    assert var_19 == 'from (imports6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL, 2khNVNgu\\h\tbL'
    module_0.from_string(var_12)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'include_trailing_comma'
    var_1 = '6y/$o}0SxMSryY)j0['
    var_2 = '2khNVNgu\\h\tbL'
    var_3 = [var_2, var_2, var_2]
    var_4 = 'from '
    var_5 = '?-'
    var_6 = []
    var_7 = True
    var_8 = 100
    var_9 = True
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'line_length'
    var_18 = {var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_1, var_14: var_6, var_15: var_7, var_16: var_16, var_0: var_7, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_9, **var_18)
    assert var_19 == 'from (?-6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL, 2khNVNgu\\h\tbL, 2khNVNgu\\h\tbL,'
    var_20 = None
    module_0.from_string(var_20)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'include_trailing_comma'
    var_1 = '6y/$o}0SxMSryY)j0['
    var_2 = '2khNVNgu\\h\tbL'
    var_3 = [var_2, var_2]
    var_4 = 'from '
    var_5 = '?-'
    var_6 = []
    var_7 = True
    var_8 = 100
    var_9 = False
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'line_length'
    var_18 = {var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_1, var_14: var_6, var_15: var_7, var_16: var_16, var_0: var_7, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_9, **var_18)
    assert var_19 == 'from (?-6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL, 2khNVNgu\\h\tbL,'
    var_20 = None
    module_0.from_string(var_20)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'include_trailing_comma'
    var_1 = '6y/$o}0SxMSryY)j0['
    var_2 = '2khNVNgu\\h\tbL'
    var_3 = [var_2, var_2]
    var_4 = 'from '
    var_5 = '?-'
    var_6 = []
    var_7 = True
    var_8 = -555
    var_9 = True
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'comments'
    var_15 = 'remove_comments'
    var_16 = 'comment_prefix'
    var_17 = 'line_length'
    var_18 = {var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_1, var_14: var_6, var_15: var_7, var_16: var_16, var_0: var_7, var_17: var_8}
    var_19 = module_0._vertical_grid_common(var_9, **var_18)
    assert var_19 == 'from (?-6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL,?-6y/$o}0SxMSryY)j0[2khNVNgu\\h\tbL,'
    module_0.from_string(var_19)

def test_case_12():
    var_0 = None
    var_1 = False
    var_2 = '    '
    var_3 = True
    var_4 = 'c?\x0c'
    var_5 = 'w@BmOX0:RL- ki|t'
    var_6 = module_0._wrap_mode_interface(var_0, var_0, var_4, var_2, var_0, var_0, var_0, var_5, var_1, var_3)
    assert var_6 == ''