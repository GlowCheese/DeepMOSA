# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.text as module_0
import re as module_1
import builtins as module_2
import enum as module_3

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.squash_html_whitespace(var_0)

def test_case_2():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_3():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_4():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._merge_original_parts(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [1])
    assert var_3 is True

def test_case_5():
    var_0 = '   '
    var_1 = ' '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_6():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [1])
    assert var_3 is True

def test_case_7():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
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
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, 3])

def test_case_9():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == ['a', 'b', 1])

def test_case_10():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [None])
    assert var_3 is True

def test_case_12():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_13():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['a'])
    assert var_3 is True

def test_case_14():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = module_1.RegexFlag.DEBUG
    var_4 = lambda : var_3
    var_5 = {var_2: var_2, var_4: var_4, var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_2.type(*var_6, **var_7)
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_9 = var_8()
    var_10 = module_0.extract_text_array(var_9, var_4, var_9)
    assert var_10 == ''
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = ()
    var_1 = 'tag'
    var_2 = {var_1: var_1, var_1: var_0, var_1: var_1, var_1: var_0, var_1: var_1, var_1: var_0}
    var_3 = [var_1, var_0, var_2]
    var_4 = module_2.type(*var_3)
    module_0.extract_text_array(var_4)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = None
    var_5 = {var_2: var_3, var_3: var_4, var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_2.type(*var_6, **var_7)
    module_0.extract_text_array(var_8)
    assert var_9 == ''

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'FakeDom'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'text'
    var_4 = module_0._merge_original_parts(var_2)
    assert module_0.INLINE_TAGS == {'img', 'q', 'abbr', 'script', 'i', 'big', 'br', 'time', 'strong', 'code', 'a', 'cite', 'b', 'map', 'em', 'bdo', 'tt', 'small', 'select', 'sub', 'object', 'textarea', 'var', 'sup', 'samp', 'kbd', 'input', 'button', 'acronym', 'label', 'dfn', 'span'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = lambda : var_4
    var_6 = {var_2: var_2, var_3: var_5, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_2.type(*var_7, **var_8)
    var_10 = var_9()
    module_0.extract_text_array(var_10)
    assert var_11 == ''

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'tag'
    var_3 = 'getchildren'
    var_4 = 'br'
    var_5 = None
    var_6 = []
    var_7 = lambda self: var_6
    var_8 = {var_2: var_4, var_4: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_2.type(*var_9, **var_10)
    var_12 = var_11()
    module_0.extract_text_array(var_12)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'a'
    var_3 = {var_1: var_2, var_2: var_0, var_2: var_0}
    var_4 = [var_2, var_0, var_3]
    var_5 = module_3._EnumDict()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'enum._EnumDict'
    assert len(var_5) == 0
    var_6 = module_2.type(*var_4, **var_5)
    module_0.extract_text_array(var_6)