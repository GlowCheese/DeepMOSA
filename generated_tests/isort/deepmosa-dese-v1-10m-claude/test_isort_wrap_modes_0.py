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
    var_0 = 'import os'
    var_1 = []
    var_2 = ''
    var_3 = 100
    var_4 = []
    var_5 = '\n'
    var_6 = '# '
    var_7 = False
    var_8 = True
    var_9 = module_0._wrap_mode_interface(var_0, var_1, var_2, var_2, var_3, var_4, var_5, var_6, var_7, var_8)
    assert var_9 == ''

def test_case_4():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_5():
    var_0 = 'some text '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'some text \\'

def test_case_6():
    var_0 = 'some text'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'some text \\'

def test_case_7():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = []
    var_10 = None
    var_11 = False
    var_12 = ''
    var_13 = '\n'
    var_14 = '    '
    var_15 = 'from module'
    var_16 = 79
    var_17 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_11, var_8: var_16}
    var_18 = module_0._vertical_grid_common(var_11, **var_17)
    assert var_18 == ''

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = [var_4, var_4, var_4]
    var_9 = None
    var_10 = False
    var_11 = ''
    var_12 = '\n'
    var_13 = '    '
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_7, var_7: var_10, var_3: var_10}
    module_0._vertical_grid_common(var_10, **var_14)

def test_case_9():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_8, var_8, var_4]
    var_10 = None
    var_11 = False
    var_12 = ''
    var_13 = '\n'
    var_14 = '    '
    var_15 = 79
    var_16 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_7, var_7: var_11, var_8: var_15}
    var_17 = module_0._vertical_grid_common(var_11, **var_16)
    assert var_17 == 'include_trailing_comma(\n    line_length, line_length, line_separator'

def test_case_10():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_8, var_8, var_4]
    var_10 = None
    var_11 = True
    var_12 = '\n'
    var_13 = '    '
    var_14 = 79
    var_15 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_8, var_4: var_12, var_5: var_13, var_6: var_7, var_7: var_11, var_8: var_14}
    var_16 = module_0._vertical_grid_common(var_11, **var_15)
    assert var_16 == 'include_trailing_comma(\n    line_length, line_length, line_separator,'

def test_case_11():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_4]
    var_10 = None
    var_11 = True
    var_12 = 'qoo7D)~=7[&c|&hr8'
    var_13 = '\n'
    var_14 = '    '
    var_15 = 79
    var_16 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_7, var_7: var_11, var_8: var_15}
    var_17 = module_0._vertical_grid_common(var_11, **var_16)
    assert var_17 == 'include_trailing_comma(\n    line_separator,'

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
    var_9 = [var_8, var_8, var_0, var_4]
    var_10 = None
    var_11 = True
    var_12 = '!hVLi5<\nd^2A\r2J'
    var_13 = '\n'
    var_14 = '    '
    var_15 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_7, var_7: var_11, var_8: var_11}
    var_16 = module_0._vertical_grid_common(var_11, **var_15)
    assert var_16 == 'include_trailing_comma(\n    line_length,\n    line_length,\n    imports,\n    line_separator,'