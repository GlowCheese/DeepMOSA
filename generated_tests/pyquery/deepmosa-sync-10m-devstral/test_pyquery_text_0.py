# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.text as module_0
import enum as module_1
import builtins as module_2

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
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_3():
    var_0 = []
    var_1 = module_0._merge_original_parts(var_0)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
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
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, 3])
    assert var_5 is True

def test_case_5():
    var_0 = '  '
    var_1 = '   '
    var_2 = [var_0, var_1]
    var_3 = module_0._merge_original_parts(var_2)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_6():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['a'])
    assert var_3 is True

def test_case_7():
    var_0 = []
    var_1 = module_0._squash_artifical_nl(var_0)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_8():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._squash_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [None])
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ()
    var_1 = 'tag'
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = {var_1: var_1, var_0: var_0, var_1: var_0}
    var_4 = [var_1, var_0, var_3]
    var_5 = module_2.type(*var_4, **var_2)
    var_6 = var_5()
    module_0.extract_text_array(var_6, var_6)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = None
    var_4 = []
    var_5 = lambda : var_4
    var_6 = {var_1: var_2, var_2: var_3, var_5: var_5}
    var_7 = [var_1, var_0, var_6]
    var_8 = {}
    var_9 = module_2.type(*var_7, **var_8)
    var_10 = var_9()
    module_0.extract_text_array(var_10)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'text'
    var_3 = 'div'
    var_4 = []
    var_5 = lambda : var_4
    var_6 = {var_1: var_3, var_2: var_5, var_3: var_5}
    var_7 = [var_2, var_0, var_6]
    var_8 = {}
    var_9 = module_2.type(*var_7, **var_8)
    var_10 = var_9()
    module_0.extract_text_array(var_10)

def test_case_12():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'r:0)PS5:&'
    var_3 = lambda : var_1
    var_4 = {var_1: var_2, var_2: var_3, var_1: var_3}
    var_5 = [var_2, var_0, var_4]
    var_6 = {}
    var_7 = module_2.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = module_0.extract_text(var_8, var_3, squash_space=var_3)
    assert var_9 == ''
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = ()
    var_1 = 'tag'
    var_2 = 'teQt'
    var_3 = ''
    var_4 = lambda : var_1
    var_5 = {var_1: var_3, var_2: var_4, var_1: var_4}
    var_6 = [var_3, var_0, var_5]
    var_7 = {}
    var_8 = module_2.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = None
    var_11 = module_0.extract_text(var_9, var_4, squash_space=var_10)
    assert var_11 == ''
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    module_0.extract_text_array(var_11)

def test_case_14():
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == ['hello'])
    assert var_3 is True

def test_case_15():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == [1, 2, 3])

def test_case_16():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._strip_artifical_nl(var_3)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_5 = bool(var_4 == ['hello'])
    assert var_5 is True

def test_case_17():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0._strip_artifical_nl(var_1)
    assert module_0.INLINE_TAGS == {'sub', 'code', 'abbr', 'big', 'strong', 'button', 'b', 'samp', 'cite', 'object', 'dfn', 'select', 'span', 'tt', 'script', 'a', 'map', 'q', 'em', 'input', 'var', 'bdo', 'sup', 'label', 'kbd', 'acronym', 'i', 'img', 'small', 'textarea', 'br', 'time'}
    assert module_0.SEPARATORS == {'br'}
    assert f'{type(module_0.WHITESPACE_RE).__module__}.{type(module_0.WHITESPACE_RE).__qualname__}' == 're.Pattern'
    var_3 = bool(var_2 == [1])
    assert var_3 is True