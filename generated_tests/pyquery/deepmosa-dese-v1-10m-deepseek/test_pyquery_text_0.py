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
    var_0 = 'span'
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_3():
    var_0 = 1
    var_1 = '  hello  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_4():
    var_0 = '  '
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_5():
    var_0 = 'div'
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_6():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_7():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_8():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_9():
    var_0 = 42
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_10():
    var_0 = 'hello'
    var_1 = 42
    var_2 = [var_0, var_1]
    var_3 = module_0._strip_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_11():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'big', 'sup', 'br', 'dfn', 'code', 'abbr', 'i', 'textarea', 'q', 'tt', 'span', 'a', 'cite', 'bdo', 'label', 'b', 'sub', 'button', 'samp', 'small', 'strong', 'acronym', 'time', 'map', 'object', 'select', 'var', 'input', 'em', 'script', 'kbd', 'img'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'