# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import enum as module_1
import collections as module_2
import re as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.parse_noqa(var_0)

def test_case_1():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_2():
    var_0 = "|!\x0b6 !f0'1,?\x0c"
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_3():
    var_0 = 'Wk\x0bk'
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2 = module_0.ignore_line(var_1, var_0, var_0)
    assert var_2 is False
    assert len(var_1) == 2

def test_case_4():
    var_0 = '(?P<code>(?P<codes>.*))'
    var_1 = 'codes: all'
    var_2 = module_3.search(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 're.Match'
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    assert f'{type(module_3.Match.string).__module__}.{type(module_3.Match.string).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Match.re).__module__}.{type(module_3.Match.re).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Match.pos).__module__}.{type(module_3.Match.pos).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Match.endpos).__module__}.{type(module_3.Match.endpos).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Match.lastindex).__module__}.{type(module_3.Match.lastindex).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Match.lastgroup).__module__}.{type(module_3.Match.lastgroup).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Match.regs).__module__}.{type(module_3.Match.regs).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0._parse_error_codes(var_2)
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}

def test_case_5():
    var_0 = 'tO!'
    var_1 = 'x = 1  # noqa: '
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_6():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]
    var_5 = {var_0: var_4, var_1: var_4}
    var_6 = module_0.ignore_line(var_5, var_2, var_0)
    assert var_6 is True
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}