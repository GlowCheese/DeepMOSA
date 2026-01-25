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
    var_2 = 80
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = module_0._wrap_mode_interface(var_0, var_1, var_0, var_0, var_2, var_3, var_0, var_0, var_4, var_5)
    assert var_6 == ''

def test_case_4():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_5():
    var_0 = 'hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

def test_case_6():
    var_0 = 'hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

def test_case_7():
    var_0 = 'imports'
    var_1 = 'remove_comments_'
    var_2 = module_0.formatter_from_string(var_1)
    var_3 = 'statement'
    var_4 = 'line_length'
    var_5 = [var_1]
    var_6 = None
    var_7 = True
    var_8 = '\n'
    var_9 = '    '
    var_10 = 80
    var_11 = {var_0: var_5, var_0: var_6, var_1: var_7, var_1: var_3, var_1: var_8, var_8: var_9, var_3: var_1, var_4: var_10, var_8: var_7}
    var_12 = module_0._vertical_grid_common(var_7, **var_11)
    assert var_12 == ''

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'imports'
    var_1 = 'remove_commnts_'
    var_2 = module_0.formatter_from_string(var_1)
    var_3 = 'W~TR'
    var_4 = 'line_length'
    var_5 = [var_1]
    var_6 = None
    var_7 = True
    var_8 = '\n'
    var_9 = '    '
    var_10 = 80
    var_11 = {var_0: var_5, var_9: var_6, var_1: var_7, var_1: var_3, var_1: var_8, var_8: var_9, var_3: var_1, var_4: var_10, var_8: var_7}
    module_0._vertical_grid_common(var_6, **var_11)

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
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = False
    var_14 = ' #'
    var_15 = '\n'
    var_16 = '    '
    var_17 = 'from module import ('
    var_18 = 80
    var_19 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_13, var_8: var_18}
    var_20 = module_0._vertical_grid_common(var_13, **var_19)
    assert var_20 == 'from module import ((\n    import1, import2'
    var_21 = var_19[var_0]
    var_22 = len(var_21)
    assert var_22 == 0

@pytest.mark.xfail(strict=True)
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
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = False
    var_14 = '\n'
    var_15 = '    '
    var_16 = 80
    var_17 = True
    var_18 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_0, var_4: var_14, var_5: var_15, var_6: var_10, var_7: var_17, var_8: var_16}
    var_19 = module_0._vertical_grid_common(var_13, **var_18)
    assert var_19 == 'import2(\n    import1, import2,'
    var_20 = 'pnv6B`S&'
    module_0.from_string(var_20)

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
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = False
    var_14 = ' #'
    var_15 = '\n'
    var_16 = '    '
    var_17 = 'from module import'
    var_18 = 80
    var_19 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_13, var_8: var_18}
    var_20 = module_0._vertical_grid_common(var_13, **var_19)
    assert var_20 == 'from module import(\n    import1, import2'
    var_21 = []
    var_22 = True
    var_23 = {var_0: var_21, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_22, var_8: var_18}
    var_24 = module_0._vertical_grid_common(var_13, **var_23)
    assert var_24 == ''
    var_25 = 'import3'
    var_26 = [var_9, var_10, var_25]
    var_27 = {var_0: var_26, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_22, var_8: var_18}
    var_28 = module_0._vertical_grid_common(var_13, **var_27)
    assert var_28 == 'from module import(\n    import1, import2, import3,'

@pytest.mark.xfail(strict=True)
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
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = True
    var_14 = '\n'
    var_15 = '   _'
    var_16 = 80
    var_17 = True
    var_18 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_0, var_4: var_14, var_5: var_15, var_6: var_10, var_7: var_17, var_8: var_16}
    var_19 = module_0._vertical_grid_common(var_13, **var_18)
    assert var_19 == 'import2(\n   _import1, import2,'
    var_20 = 'pnv6B`S&'
    module_0.from_string(var_20)

def test_case_13():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = 'import1'
    var_10 = 'import2'
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = False
    var_14 = ' #'
    var_15 = '\n'
    var_16 = '    '
    var_17 = 'from module import'
    var_18 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_13, var_8: var_13}
    var_19 = module_0._vertical_grid_common(var_13, **var_18)
    assert var_19 == 'from module import(\n    import1,\n    import2'
    var_20 = []
    var_21 = True
    var_22 = {var_0: var_20, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_21, var_8: var_21}
    var_23 = module_0._vertical_grid_common(var_13, **var_22)
    assert var_23 == ''
    var_24 = 'import3'
    var_25 = [var_9, var_10, var_24]
    var_26 = {var_0: var_25, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_21, var_8: var_21}
    var_27 = module_0._vertical_grid_common(var_12, **var_26)
    assert var_27 == 'from module import(\n    import1,\n    import2,\n    import3,'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'imports'
    var_1 = 'comments'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'line_separator'
    var_5 = 'indent'
    var_6 = 'statement'
    var_7 = 'include_trailing_comma'
    var_8 = 'line_length'
    var_9 = 'import2'
    var_10 = [var_3, var_9]
    var_11 = None
    var_12 = True
    var_13 = '\n'
    var_14 = '    '
    var_15 = 'from module import'
    var_16 = 80
    var_17 = {var_0: var_10, var_1: var_11, var_2: var_12, var_3: var_0, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_12, var_8: var_16}
    var_18 = module_0._vertical_grid_common(var_12, **var_17)
    assert var_18 == 'from module import(\n    comment_prefix, import2,'
    var_19 = [var_3]
    var_20 = True
    var_21 = {var_0: var_19, var_1: var_11, var_2: var_12, var_3: var_0, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_20, var_8: var_16}
    var_22 = module_0._vertical_grid_common(var_12, **var_21)
    assert var_22 == 'from module import(\n    comment_prefix,'
    var_23 = '8m\tdnC+8u^-pQ5Dk M'
    var_24 = 'AWY6lxwbi#n,~rQVgLi'
    var_25 = '@T\\t`'
    var_26 = [var_24, var_25]
    var_27 = "I|O>5k5$jE2vjs':"
    module_0.vertical_hanging_indent_bracket(var_22, var_11, var_9, var_23, var_11, var_26, var_27, var_18, var_12, var_11)