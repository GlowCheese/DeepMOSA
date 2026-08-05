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
    var_0 = '  More Text  '
    var_1 = module_0.squash_html_whitespace(var_0)
    assert var_1 == ' More Text '
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = module_0._merge_original_parts(var_1)

def test_case_3():
    var_0 = True
    var_1 = False
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = ''
    var_2 = -32.821877089841465
    var_3 = [var_0, var_1, var_2, var_1]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    module_0.squash_html_whitespace(var_4)

def test_case_5():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_6():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_7():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_8():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_0, var_1, var_2, var_0, var_0]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_9():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_10():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

def test_case_11():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'var', 'big', 'strong', 'label', 'i', 'select', 'em', 'b', 'object', 'textarea', 'samp', 'abbr', 'br', 'time', 'button', 'img', 'input', 'sup', 'span', 'q', 'a', 'sub', 'small', 'acronym', 'bdo', 'script', 'map', 'code', 'kbd', 'tt', 'cite', 'dfn'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'