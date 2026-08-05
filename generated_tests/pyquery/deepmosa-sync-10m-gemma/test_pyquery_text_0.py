# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.text as module_0
import re as module_1

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.squash_html_whitespace(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'f$85'
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    module_0.extract_text(var_1)

def test_case_3():
    var_0 = []
    var_1 = []
    var_2 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '$85'
    var_2 = module_1.RegexFlag.DOTALL
    var_3 = {var_0: var_1, var_1: var_1, var_0: var_1, var_2: var_2}
    var_4 = module_0._merge_original_parts(var_3)
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
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    module_0.extract_text_array(var_0, strip_artifical_nl=var_4)

def test_case_5():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_6():
    var_0 = []
    var_1 = module_0._strip_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_7():
    var_0 = None
    var_1 = 1
    var_2 = False
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [None, 1, False])

def test_case_8():
    var_0 = 'start'
    var_1 = None
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == ['start', None, 1])

def test_case_9():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [None])
    assert var_3 is True

def test_case_10():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_11():
    var_0 = None
    var_1 = 'a'
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._squash_artifical_nl(var_2)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == [None, 'a'])
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'a'
    var_1 = None
    var_2 = ''
    var_3 = True
    var_4 = 'c'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = [var_0, var_1, var_2, var_3, var_4]
    var_7 = module_0._merge_original_parts(var_5)
    assert module_0.INLINE_TAGS == {'i', 'dfn', 'button', 'bdo', 'sup', 'abbr', 'time', 'big', 'code', 'em', 'q', 'map', 'kbd', 'script', 'cite', 'small', 'textarea', 'samp', 'b', 'acronym', 'tt', 'object', 'br', 'input', 'img', 'var', 'strong', 'select', 'a', 'sub', 'span', 'label'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    module_0.extract_text(var_6, sep_symbol=var_7)