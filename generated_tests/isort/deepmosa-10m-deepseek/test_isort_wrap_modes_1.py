# Check out: https://github.com/GlowCheese/deepmosa
import email._header_value_parser as module_1

import isort.wrap_modes as module_0
import pytest


def test_case_0():
    pass

def test_case_1():
    var_0 = ''
    var_1 = module_0.formatter_from_string(var_0)
    var_2 = module_0._hanging_indent_end_line(var_0)
    assert var_2 == ' \\'

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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'Sk'
    var_1 = [var_0, var_0, var_0]
    var_2 = False
    var_3 = None
    var_4 = 'imports'
    var_5 = 'statement'
    var_6 = 'indent'
    var_7 = 'comments'
    var_8 = {var_4: var_1, var_5: var_7, var_7: var_6, var_6: var_5, var_0: var_2, var_6: var_2, var_7: var_3, var_4: var_6, var_7: var_2}
    module_0._vertical_grid_common(var_2, **var_8)

def test_case_5():
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
def test_case_6():
    var_0 = 'SaxI'
    var_1 = [var_0]
    var_2 = False
    var_3 = None
    var_4 = 'O'
    var_5 = 'imports'
    var_6 = 'statement'
    var_7 = 'line_separator'
    var_8 = 'indent'
    var_9 = 'remove_comments'
    var_10 = 'comments'
    var_11 = 'comment_prefix'
    var_12 = {var_5: var_1, var_6: var_7, var_7: var_4, var_8: var_6, var_0: var_2, var_9: var_2, var_10: var_3, var_11: var_4, var_9: var_2}
    module_0._vertical_grid_common(var_2, **var_12)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'remove_comments'
    var_1 = 'comments'
    var_2 = [var_0]
    var_3 = ''
    var_4 = False
    var_5 = None
    var_6 = module_1.get_fws(var_0)
    assert module_1.hexdigits == '0123456789abcdefABCDEF'
    assert module_1.WSP == {' ', '\t'}
    assert module_1.CFWS_LEADER == {' ', '\t', '('}
    assert module_1.SPECIALS == {':', '\\', ',', '>', ']', '@', ')', '[', '(', '<', ';', '"', '.'}
    assert module_1.ATOM_ENDS == {':', '\\', ',', '>', '\t', ']', '@', ')', '[', '(', '<', ' ', ';', '"', '.'}
    assert module_1.DOT_ATOM_ENDS == {':', '\\', ',', '>', '\t', ']', '@', ')', '[', '(', '<', ' ', ';', '"'}
    assert module_1.PHRASE_ENDS == {':', '\\', '>', ']', '@', ';', ')', '[', '<', ','}
    assert module_1.TSPECIALS == {',', ':', '\\', '>', ']', '/', '?', '=', '@', '[', ')', '(', '<', ';', '"'}
    assert module_1.TOKEN_ENDS == {':', '\\', '>', '\t', ']', '/', '?', '@', '=', ';', '[', ')', '(', '<', ' ', ',', '"'}
    assert module_1.ASPECIALS == {':', '\\', '%', '>', ']', '/', '?', '*', '@', '=', ';', '[', ')', '(', '<', "'", ',', '"'}
    assert module_1.ATTRIBUTE_ENDS == {'%', '>', '@', '[', '<', "'", ';', ',', '"', ':', '\\', '\t', '/', '?', '=', '(', ']', '*', ')', ' '}
    assert module_1.EXTENDED_ATTRIBUTE_ENDS == {'>', '@', '[', '<', "'", ';', ',', '"', ':', '\\', '\t', '/', '?', '=', '(', ']', '*', ')', ' '}
    assert module_1.NLSET == {'\r', '\n'}
    assert module_1.SPECIALSNL == {':', '\\', ',', '>', ']', '@', '\r', ')', '[', '(', '<', ';', '"', '\n', '.'}
    assert f'{type(module_1.rfc2047_matcher).__module__}.{type(module_1.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DOT).__module__}.{type(module_1.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_1.DOT) == 1
    assert f'{type(module_1.ListSeparator).__module__}.{type(module_1.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_1.ListSeparator) == 1
    assert f'{type(module_1.RouteComponentMarker).__module__}.{type(module_1.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_1.RouteComponentMarker) == 1
    var_7 = 'imports'
    var_8 = 'statement'
    var_9 = 'line_separator'
    var_10 = 'indent'
    var_11 = 'remove_comments'
    var_12 = 'comments'
    var_13 = 'comment_prefix'
    var_14 = 'include_trailing_comma'
    var_15 = {var_7: var_2, var_8: var_13, var_9: var_3, var_10: var_11, var_13: var_9, var_11: var_4, var_12: var_5, var_13: var_1, var_14: var_4}
    var_16 = module_0._vertical_grid_common(var_4, **var_15)
    assert var_16 == 'comment_prefix(remove_commentsremove_comments'
    module_0.backslash_grid(var_5, var_14, var_5, var_5, var_5, var_6, var_5, var_5, var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'comments'
    var_1 = 'comment_prefix'
    var_2 = 'SaxI'
    var_3 = [var_2]
    var_4 = 'from x import '
    var_5 = '\n'
    var_6 = 80
    var_7 = True
    var_8 = None
    var_9 = '#'
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = '=f2-\n'
    var_15 = 'remove_comments'
    var_16 = 'comments'
    var_17 = 'comment_prefix'
    var_18 = 'include_trailing_comma'
    var_19 = {var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_0, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_9, var_18: var_7}
    var_20 = module_0._vertical_grid_common(var_7, **var_19)
    assert var_20 == 'from x import (\ncommentsSaxI,'
    module_0.backslash_grid(var_8, var_18, var_8, var_8, var_8, var_1, var_8, var_8, var_8, var_8)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'SaxI'
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = None
    var_4 = 'imports'
    var_5 = 'statement'
    var_6 = 'line_separator'
    var_7 = 'indent'
    var_8 = 'remove_comments'
    var_9 = 'comments'
    var_10 = 'comment_prefix'
    var_11 = {var_4: var_1, var_5: var_6, var_6: var_4, var_7: var_5, var_0: var_2, var_8: var_2, var_9: var_3, var_10: var_4, var_9: var_2}
    module_0._vertical_grid_common(var_2, **var_11)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'SaxI'
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = None
    var_4 = 'imports'
    var_5 = 'statement'
    var_6 = 'line_separator'
    var_7 = 'indent'
    var_8 = 'remove_comments'
    var_9 = 'comments'
    var_10 = 'comment_prefix'
    var_11 = 'inclue_trailing_comma'
    var_12 = {var_4: var_1, var_5: var_6, var_6: var_6, var_7: var_5, var_0: var_2, var_8: var_2, var_9: var_3, var_10: var_6, var_11: var_2}
    module_0._vertical_grid_common(var_2, **var_12)

def test_case_11():
    var_0 = 'imports'
    var_1 = 'line_length&'
    var_2 = 'include_trailing_comma'
    var_3 = 'import os'
    var_4 = module_0.formatter_from_string(var_2)
    var_5 = 'import sys'
    var_6 = [var_3, var_5]
    var_7 = 'from x import '
    var_8 = '\n'
    var_9 = 30
    var_10 = False
    var_11 = None
    var_12 = '$'
    var_13 = 'statement'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'line_length'
    var_17 = 'remove_comments'
    var_18 = 'comments'
    var_19 = 'comment_prefix'
    var_20 = 'include_trailing_comma'
    var_21 = {var_0: var_6, var_13: var_7, var_14: var_8, var_15: var_1, var_16: var_9, var_17: var_10, var_18: var_11, var_19: var_12, var_20: var_10}
    var_22 = module_0._vertical_grid_common(var_10, **var_21)
    assert var_22 == 'from x import (\nline_length&import os,\nline_length&import sys'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'imports'
    var_1 = 'include_trailing_cmma'
    var_2 = 'import os'
    var_3 = module_0.formatter_from_string(var_1)
    var_4 = 'import sys'
    var_5 = [var_2, var_4]
    var_6 = 'from x import '
    var_7 = '\n'
    var_8 = '    '
    var_9 = 30
    var_10 = False
    var_11 = None
    var_12 = '#'
    var_13 = 'statement'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'line_length'
    var_17 = 'remove_comments'
    var_18 = 'comments'
    var_19 = 'comment_prefix'
    var_20 = 'include_trailing_comma'
    var_21 = {var_0: var_5, var_13: var_6, var_14: var_7, var_15: var_8, var_16: var_9, var_17: var_10, var_18: var_11, var_19: var_12, var_20: var_10}
    var_22 = module_0._vertical_grid_common(var_10, **var_21)
    assert var_22 == 'from x import (\n    import os, import sys'
    var_23 = "nW9Wz?hF;y^'\x0c J,,"
    module_0.from_string(var_23)

def test_case_13():
    var_0 = 'comment_prefix'
    var_1 = 'import os'
    var_2 = 'import sys'
    var_3 = [var_1, var_2]
    var_4 = 'from x import '
    var_5 = '\n'
    var_6 = '    '
    var_7 = 80
    var_8 = False
    var_9 = None
    var_10 = '#'
    var_11 = True
    var_12 = 'imports'
    var_13 = 'statement'
    var_14 = 'line_separator'
    var_15 = 'indent'
    var_16 = 'line_length'
    var_17 = 'remove_comments'
    var_18 = module_0.formatter_from_string(var_0)
    var_19 = 'comments'
    var_20 = 'comment_prefix'
    var_21 = 'include_trailing_comma'
    var_22 = {var_12: var_3, var_13: var_4, var_14: var_5, var_15: var_6, var_16: var_7, var_17: var_8, var_19: var_9, var_20: var_10, var_21: var_11}
    var_23 = module_0._vertical_grid_common(var_8, **var_22)
    assert var_23 == 'from x import (\n    import os, import sys,'
    var_24 = 'from x import (    import os, import sys,)'
    var_25 = bool(var_23 == var_24)

def test_case_14():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = 'from x import '
    var_4 = '\n'
    var_5 = '    '
    var_6 = 30
    var_7 = True
    var_8 = None
    var_9 = '#'
    var_10 = 'imports'
    var_11 = 'statement'
    var_12 = 'line_separator'
    var_13 = 'indent'
    var_14 = 'line_length'
    var_15 = 'remove_comments'
    var_16 = 'comments'
    var_17 = 'comment_prefix'
    var_18 = 'include_trailing_comma'
    var_19 = {var_10: var_2, var_11: var_3, var_12: var_4, var_13: var_5, var_14: var_6, var_15: var_7, var_16: var_8, var_17: var_9, var_18: var_7}
    var_20 = module_0._vertical_grid_common(var_7, **var_19)
    assert var_20 == 'from x import (\n    import os, import sys,'
    var_21 = 'from x import (    import os,\n    import sys)'
    var_22 = bool(var_20 == var_21)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'imports'
    var_1 = 'statement'
    var_2 = 'remove_comments'
    var_3 = 'comment_prefix'
    var_4 = 'vWMrKD'
    var_5 = 'import os'
    var_6 = module_0.formatter_from_string(var_4)
    var_7 = 'import sys'
    var_8 = [var_1, var_5, var_7, var_3]
    var_9 = 'from x import '
    var_10 = '\n'
    var_11 = '    '
    var_12 = 30
    var_13 = False
    var_14 = None
    var_15 = '#'
    var_16 = 'statement'
    var_17 = 'line_separator'
    var_18 = 'indent'
    var_19 = 'line_length'
    var_20 = 'remove_comments'
    var_21 = 'comments'
    var_22 = 'comment_prefix'
    var_23 = 'include_trailing_comma'
    var_24 = {var_0: var_8, var_16: var_9, var_17: var_10, var_18: var_11, var_19: var_12, var_20: var_13, var_21: var_14, var_22: var_15, var_2: var_7, var_10: var_9, var_23: var_13}
    var_25 = module_0._vertical_grid_common(var_13, **var_24)
    assert var_25 == 'from x import (\n    statement, import os,\n    import sys, comment_prefix'
    module_0.from_string(var_14)

def test_case_16():
    var_0 = ' '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'

def test_case_17():
    var_0 = ''
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'