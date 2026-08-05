# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.text as module_0

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.squash_html_whitespace(var_0)

def test_case_2():
    var_0 = None
    var_1 = True
    var_2 = False
    var_3 = [var_0, var_1, var_2, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [None, True, False, 0])
    assert var_5 is True

def test_case_3():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_4():
    var_0 = '  leading'
    var_1 = 'middle   '
    var_2 = 'trailing  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == ['leading middle trailing'])

def test_case_5():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_6():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_7():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == [None, 1, None])

def test_case_8():
    var_0 = '  '
    var_1 = '\n\t'
    var_2 = ' '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_9():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = [var_0, var_0, var_1, var_2, var_3, var_0, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_6 = bool(var_5 == ['start', 1, 'end'])
    assert var_6 is True

def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [None])
    assert var_3 is True

def test_case_11():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_12():
    var_0 = 'a'
    var_1 = None
    var_2 = [var_0, var_1, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'dfn', 'samp', 'button', 'bdo', 'kbd', 'br', 'cite', 'tt', 'big', 'acronym', 'select', 'object', 'em', 'input', 'img', 'textarea', 'var', 'map', 'time', 'strong', 'a', 'sup', 'sub', 'abbr', 'script', 'code', 'b', 'span', 'q', 'small', 'i', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == ['a', None])
    assert var_4 is True