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
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'remove_comments'
    var_6 = 'comment_prefix'
    var_7 = 'include_trailing_comma'
    var_8 = 'comments'
    var_9 = 'import os'
    var_10 = [var_9]
    var_11 = ''
    var_12 = '\n'
    var_13 = '    '
    var_14 = 88
    var_15 = False
    var_16 = '  # '
    var_17 = None
    var_18 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_15, var_6: var_16, var_7: var_15, var_8: var_17}
    var_19 = module_0._vertical_grid_common(var_15, **var_18)
    assert var_19 == '(\n    import os'

def test_case_4():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'remove_comments'
    var_6 = 'comment_prefix'
    var_7 = 'include_trailing_comma'
    var_8 = 'comments'
    var_9 = []
    var_10 = ''
    var_11 = '\n'
    var_12 = '    '
    var_13 = 88
    var_14 = False
    var_15 = '  # '
    var_16 = None
    var_17 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_14, var_8: var_16}
    var_18 = module_0._vertical_grid_common(var_14, **var_17)
    assert var_18 == ''

def test_case_5():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'remove_comments'
    var_6 = 'comment_prefix'
    var_7 = 'include_trailing_comma'
    var_8 = 'comments'
    var_9 = [var_2, var_2, var_8]
    var_10 = '\n'
    var_11 = '    '
    var_12 = 74
    var_13 = False
    var_14 = '  #'
    var_15 = {var_0: var_9, var_1: var_3, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_14, var_7: var_13, var_8: var_9}
    var_16 = 'BWb*tzUjO/oW'
    var_17 = module_0.formatter_from_string(var_16)
    var_18 = module_0._vertical_grid_common(var_13, **var_15)
    assert var_18 == 'indent(  # line_separator; comments\n    line_separator, line_separator, comments'

def test_case_6():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'remove_comments'
    var_6 = 'comment_prefix'
    var_7 = 'include_trailing_comma'
    var_8 = 'comments'
    var_9 = [var_2, var_1, var_8]
    var_10 = ''
    var_11 = 88
    var_12 = False
    var_13 = '  #'
    var_14 = 'comment2'
    var_15 = [var_4, var_4, var_14]
    var_16 = {var_0: var_9, var_1: var_10, var_2: var_0, var_3: var_0, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_12, var_5: var_9, var_7: var_8, var_8: var_15, var_14: var_4}
    var_17 = 'BWb*tzUjO/oW'
    var_18 = module_0.formatter_from_string(var_17)
    var_19 = module_0._vertical_grid_common(var_12, **var_16)
    assert var_19 == '(importsimportsline_separator, statement, comments,'

def test_case_7():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'line_separator'
    var_3 = 'indent'
    var_4 = 'line_length'
    var_5 = 'remove_comments'
    var_6 = 'comment_prefix'
    var_7 = 'include_trailing_comma'
    var_8 = 'comments'
    var_9 = 'import os'
    var_10 = [var_2, var_9, var_8]
    var_11 = ''
    var_12 = '\n'
    var_13 = '    '
    var_14 = True
    var_15 = '  #'
    var_16 = 'comment2'
    var_17 = [var_4, var_4, var_16]
    var_18 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_13, var_4: var_14, var_5: var_14, var_6: var_15, var_7: var_14, var_8: var_17}
    var_19 = 'BWb*tzUjO/oW'
    var_20 = module_0.formatter_from_string(var_19)
    var_21 = module_0._vertical_grid_common(var_14, **var_18)
    assert var_21 == '(\n    line_separator,\n    import os,\n    comments,'

def test_case_8():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_9():
    var_0 = 'test '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_10():
    var_0 = 'test'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_11():
    var_0 = 'INVALID'
    var_1 = '\n'
    var_2 = False
    var_3 = None
    var_4 = [var_1]
    var_5 = module_0._wrap_mode_interface(var_3, var_4, var_2, var_3, var_3, var_4, var_0, var_3, var_2, var_3)
    assert var_5 == ''