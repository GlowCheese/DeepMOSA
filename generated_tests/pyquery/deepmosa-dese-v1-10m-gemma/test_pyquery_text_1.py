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
    var_0 = 'Start '
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_3():
    var_0 = None
    var_1 = 1
    var_2 = False
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_4():
    var_0 = ' '
    var_1 = '\n'
    var_2 = '\t  \n'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = lambda x: x
    module_0._strip_artifical_nl(var_0)

def test_case_6():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_7():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._strip_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_8():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_9():
    var_0 = None
    var_1 = 'start'
    var_2 = 1
    var_3 = 'end'
    var_4 = [var_0, var_1, var_2, var_3, var_0]
    var_5 = module_0._strip_artifical_nl(var_4)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_11():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_12():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1, var_0]
    var_3 = module_0._squash_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'sub', 'time', 'br', 'img', 'var', 'span', 'button', 'samp', 'dfn', 'big', 'b', 'strong', 'script', 'label', 'bdo', 'input', 'object', 'select', 'em', 'a', 'kbd', 'textarea', 'i', 'acronym', 'tt', 'code', 'q', 'sup', 'map', 'small', 'cite', 'abbr'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'