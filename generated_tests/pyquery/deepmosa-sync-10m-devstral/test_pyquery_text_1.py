# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.text as module_0
import builtins as module_1

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.squash_html_whitespace(var_0)

def test_case_2():
    var_0 = 'hello '
    var_1 = ' world  '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == ['hello world'])
    assert var_4 is True

def test_case_3():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_4():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

def test_case_5():
    var_0 = '   '
    var_1 = '  \n  '
    var_2 = '  \t  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._merge_original_parts(var_3)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_6():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_7():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_8():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, 3])

def test_case_9():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._strip_artifical_nl(var_4)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_6 = bool(var_5 == ['a', 'b'])
    assert var_6 is True

def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [None])
    assert var_3 is True

def test_case_11():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_12():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._squash_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, None])
    assert var_5 is True

def test_case_13():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'getchildren'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {var_1: var_4, var_2: var_2}
    var_6 = [var_2, var_0, var_5]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_0.extract_text_array(var_9)
    assert var_10 == ''
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = ()
    var_1 = 'tag'
    var_2 = {var_1: var_1}
    var_3 = [var_1, var_0, var_2]
    var_4 = {}
    var_5 = module_1.type(*var_3, **var_4)
    module_0.extract_text_array(var_5, strip_artifical_nl=var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'MockDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = '=|'
    var_5 = None
    var_6 = lambda : var_1
    var_7 = {var_2: var_4, var_3: var_5, var_0: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = var_10()
    module_0.extract_text_array(var_11)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'MockDom'
    var_1 = module_0.squash_html_whitespace(var_0)
    assert var_1 == 'MockDom'
    assert module_0.INLINE_TAGS == {'bdo', 'br', 'script', 'dfn', 'select', 'img', 'code', 'q', 'span', 'a', 'label', 'object', 'kbd', 'samp', 'button', 'input', 'big', 'sup', 'textarea', 'abbr', 'small', 'b', 'strong', 'var', 'tt', 'cite', 'acronym', 'map', 'em', 'i', 'time', 'sub'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = ()
    var_3 = 'tag'
    var_4 = 'text'
    var_5 = '=|'
    var_6 = lambda : var_2
    var_7 = {var_3: var_5, var_4: var_1, var_1: var_6}
    var_8 = [var_0, var_2, var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = var_10()
    module_0.extract_text_array(var_11)