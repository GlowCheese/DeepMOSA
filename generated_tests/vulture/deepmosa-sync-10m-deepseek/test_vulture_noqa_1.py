# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import collections as module_1
import re as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.parse_noqa(var_0)

def test_case_1():
    var_0 = 'x = 1  # noqa: E501\ny = 2  # noqa: W503\n'
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = '9'
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = False
    var_4 = module_0.ignore_line(var_2, var_3, var_0)
    assert var_4 is False
    assert len(var_2) == 2
    var_5 = None
    var_2.__setattr__(var_5, var_4, var_4)

def test_case_3():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 1
    var_3 = {var_2}
    var_4 = {var_2}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_0.ignore_line(var_5, var_2, var_0)
    assert var_6 is True
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}

def test_case_4():
    var_0 = '(?P<codes>.*)'
    var_1 = 'all'
    var_2 = module_2.match(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 're.Match'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    assert f'{type(module_2.Match.string).__module__}.{type(module_2.Match.string).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Match.re).__module__}.{type(module_2.Match.re).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Match.pos).__module__}.{type(module_2.Match.pos).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Match.endpos).__module__}.{type(module_2.Match.endpos).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Match.lastindex).__module__}.{type(module_2.Match.lastindex).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Match.lastgroup).__module__}.{type(module_2.Match.lastgroup).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Match.regs).__module__}.{type(module_2.Match.regs).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0._parse_error_codes(var_2)
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    var_4 = bool(var_3 == ['all'])
    assert var_4 is True

def test_case_5():
    var_0 = 'import os  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = bool(var_2)
    assert var_3 is True