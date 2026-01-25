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
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_4():
    var_0 = None
    var_1 = '1\x0cp4'
    var_2 = True
    var_3 = module_0._wrap_mode_interface(var_0, var_0, var_1, var_0, var_0, var_0, var_1, var_1, var_2, var_0)
    assert var_3 == ''

def test_case_5():
    var_0 = 'test '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_6():
    var_0 = ''
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_7():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = 'os'
    var_10 = [var_9]
    var_11 = None
    var_12 = False
    var_13 = ''
    var_14 = '\n'
    var_15 = '    '
    var_16 = 80
    var_17 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_12, var_8: var_16}
    var_18 = module_0._vertical_grid_common(var_12, **var_17)
    assert var_18 == '(\n    os'

def test_case_8():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = []
    var_10 = None
    var_11 = False
    var_12 = ''
    var_13 = '\n'
    var_14 = '    '
    var_15 = 80
    var_16 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_12, var_5: var_13, var_6: var_14, var_7: var_11, var_8: var_15}
    var_17 = module_0._vertical_grid_common(var_11, **var_16)
    assert var_17 == ''

def test_case_9():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = 'os'
    var_10 = [var_9]
    var_11 = None
    var_12 = '1'
    var_13 = '\n'
    var_14 = '    '
    var_15 = 80
    var_16 = {var_0: var_10, var_1: var_11, var_2: var_14, var_3: var_12, var_4: var_12, var_5: var_13, var_6: var_14, var_7: var_14, var_8: var_15}
    var_17 = True
    var_18 = module_0._vertical_grid_common(var_17, **var_16)
    assert var_18 == '1(\n    os,'

def test_case_10():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_1, var_2, var_4]
    var_10 = 'comment'
    var_11 = [var_10]
    var_12 = '# '
    var_13 = ''
    var_14 = '\n'
    var_15 = '    '
    var_16 = False
    var_17 = 80
    var_18 = {var_0: var_9, var_1: var_11, var_2: var_16, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17}
    var_19 = module_0._vertical_grid_common(var_16, **var_18)
    assert var_19 == '(#  comment\n    comments, remove_comments, statement'

def test_case_11():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'include_trailing_comma'
    var_7 = 'line_length'
    var_8 = 'comments'
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = 'from module import'
    var_13 = False
    var_14 = '#'
    var_15 = '\n'
    var_16 = '    '
    var_17 = True
    var_18 = 80
    var_19 = 'comment1'
    var_20 = [var_19]
    var_21 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_18, var_8: var_20}
    var_22 = module_0._vertical_grid_common(var_13, **var_21)
    assert var_22 == 'from module import(# comment1\n    import1, import2,'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_1, var_2, var_4]
    var_10 = 'comment'
    var_11 = [var_10]
    var_12 = True
    var_13 = ''
    var_14 = '\n'
    var_15 = '    '
    var_16 = True
    var_17 = 89
    var_18 = {var_0: var_9, var_1: var_11, var_2: var_12, var_3: var_5, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17}
    var_19 = module_0._vertical_grid_common(var_16, **var_18)
    assert var_19 == '(\n    comments, remove_comments, statement,'
    var_20 = None
    module_0.from_string(var_20)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'statement'
    var_5 = 'line_separator'
    var_6 = 'indent'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = [var_1, var_2, var_4]
    var_10 = 'comment'
    var_11 = [var_10]
    var_12 = True
    var_13 = '# '
    var_14 = ''
    var_15 = '\n'
    var_16 = '    '
    var_17 = False
    var_18 = {var_0: var_9, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_15, var_6: var_16, var_7: var_17, var_8: var_17}
    var_19 = module_0._vertical_grid_common(var_17, **var_18)
    assert var_19 == '(\n    comments,\n    remove_comments,\n    statement'
    var_20 = None
    module_0.from_string(var_20)