# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_string(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = 'x = 1'
    var_1 = 'import os'
    var_2 = [var_1]
    var_3 = ' '
    var_4 = '    '
    var_5 = 80
    var_6 = '# comment'
    var_7 = [var_6]
    var_8 = '\n'
    var_9 = '#'
    var_10 = True
    var_11 = False
    var_12 = module_0._wrap_mode_interface(var_0, var_2, var_3, var_4, var_5, var_7, var_8, var_9, var_10, var_11)
    assert var_12 == ''

def test_case_4():
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
    var_11 = 'from'
    var_12 = '\n'
    var_13 = '    '
    var_14 = False
    var_15 = ''
    var_16 = []
    var_17 = 100
    var_18 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_15, var_6: var_16, var_7: var_17, var_8: var_14}
    var_19 = True
    var_20 = module_0._vertical_grid_common(var_19, **var_18)
    assert var_20 == 'from(\n    os'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'remove_comments'
    var_5 = 'comment_prefix'
    var_6 = 'comments'
    var_7 = 'include_trailing_comma'
    var_8 = '\n'
    var_9 = '    @'
    var_10 = True
    var_11 = [var_5]
    var_12 = 100
    var_13 = {var_0: var_11, var_1: var_9, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_9, var_6: var_11, var_9: var_12, var_7: var_10}
    var_14 = module_0._vertical_grid_common(var_10, **var_13)
    assert var_14 == '    @(\n    @comment_prefix,'
    var_15 = None
    var_16 = var_11.__contains__(var_8)
    assert var_16 is False
    var_17 = False
    var_18 = '/!,4"1(gJ!s6V:5"'
    module_0.vertical_grid_grouped_no_comma(var_5, var_15, var_15, var_15, var_17, var_11, var_18, var_15, var_16, var_15)

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_10 = [var_7, var_9]
    var_11 = '    @'
    var_12 = True
    var_13 = [var_11]
    var_14 = 100
    var_15 = {var_0: var_10, var_1: var_11, var_2: var_7, var_3: var_11, var_4: var_12, var_5: var_11, var_6: var_13, var_7: var_14, var_8: var_12}
    var_16 = module_0._vertical_grid_common(var_12, **var_15)
    assert var_16 == '    @(line_length    @line_length, os,'
    var_17 = var_13.__contains__(var_11)
    assert var_17 is True
    var_18 = False
    var_19 = '/!,4"1(gJ!s6V:5"'
    module_0.vertical_grid_grouped_no_comma(var_5, var_17, var_17, var_17, var_18, var_13, var_19, var_17, var_18, var_17)

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_10 = [var_7, var_9, var_9]
    var_11 = 'G'
    var_12 = '    @'
    var_13 = False
    var_14 = 2915
    var_15 = {var_0: var_10, var_1: var_12, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_12, var_6: var_10, var_7: var_14, var_8: var_13}
    var_16 = module_0._vertical_grid_common(var_13, **var_15)
    assert var_16 == '    @(    @ line_length; osG    @line_length, os, os'
    var_17 = None
    module_0.from_string(var_17)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'imports'
    var_1 = 'b(KyC_w%}-l'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'remove_comments'
    var_5 = '\x0comment_prefix'
    var_6 = 'comments'
    var_7 = 'line_length'
    var_8 = 'include_trailing_comma'
    var_9 = []
    var_10 = '\n'
    var_11 = '    '
    var_12 = ''
    var_13 = 100
    var_14 = False
    var_15 = {var_0: var_9, var_1: var_11, var_2: var_10, var_3: var_11, var_4: var_14, var_5: var_12, var_6: var_9, var_7: var_13, var_8: var_14}
    var_16 = module_0._vertical_grid_common(var_14, **var_15)
    assert var_16 == ''
    var_17 = None
    module_0.from_string(var_17)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_10 = [var_7, var_9]
    var_11 = 'G'
    var_12 = '    '
    var_13 = True
    var_14 = {var_0: var_10, var_1: var_12, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_12, var_6: var_10, var_7: var_13, var_8: var_13}
    var_15 = module_0._vertical_grid_common(var_13, **var_14)
    assert var_15 == '    (G    line_length,G    os,'
    var_16 = var_10.__contains__(var_11)
    assert var_16 is False
    var_17 = False
    var_18 = '/!,4"1(gJ!s6V:5"'
    module_0.vertical_grid_grouped_no_comma(var_5, var_16, var_16, var_16, var_17, var_10, var_18, var_16, var_17, var_16)

@pytest.mark.xfail(strict=True)
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
    var_10 = [var_7, var_9, var_9]
    var_11 = 'G'
    var_12 = 2915
    var_13 = True
    var_14 = {var_0: var_10, var_1: var_2, var_2: var_11, var_3: var_2, var_4: var_13, var_5: var_2, var_6: var_10, var_7: var_12, var_8: var_13}
    var_15 = module_0._vertical_grid_common(var_13, **var_14)
    assert var_15 == 'line_separator(Gline_separatorline_length, os, os,'
    var_16 = None
    module_0.from_string(var_16)

def test_case_11():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_12():
    var_0 = ' '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_13():
    var_0 = 'hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'