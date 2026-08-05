# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import enum as module_1
import collections as module_2
import builtins as module_3
import re as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.parse_noqa(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(AttributeError):
        var_2.__getattr__(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'\xa2\xc6\xaf1\r'
    module_0.parse_noqa(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_3.object(*var_2)
    var_4 = module_0.ignore_line(var_2, var_0, var_0)
    assert var_4 is False
    assert len(var_2) == 2
    module_0.ignore_line(var_0, var_4, var_2)

def test_case_4():
    var_0 = 'E402'
    var_1 = 'Import error'
    var_2 = {var_0: var_1}
    var_3 = 'import os'
    var_4 = 'import sys  # noqa: E402, F401'
    var_5 = "print('hello')  # noqa"
    var_6 = 'x = 1  # noqa: E701'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.parse_noqa(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.defaultdict'
    assert len(var_8) == 4
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_9 = var_8['Import error']
    var_10 = bool(var_8['Import error'] == {2})
    assert var_10 is True
    var_11 = var_8['F401']
    var_12 = bool(var_8['F401'] == {2})
    assert var_12 is True
    var_13 = var_8['all']
    with pytest.raises(KeyError):
        var_14 = bool(var_2['all'] == {3})
    assert var_14 is True

def test_case_5():
    var_0 = '# noqa:? (?P<code>.*)'
    var_1 = module_4.compile(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 're.Pattern'
    assert module_4.ASCII == module_4.RegexFlag.ASCII
    assert module_4.A == module_4.RegexFlag.ASCII
    assert module_4.IGNORECASE == module_4.RegexFlag.IGNORECASE
    assert module_4.I == module_4.RegexFlag.IGNORECASE
    assert module_4.LOCALE == module_4.RegexFlag.LOCALE
    assert module_4.L == module_4.RegexFlag.LOCALE
    assert module_4.UNICODE == module_4.RegexFlag.UNICODE
    assert module_4.U == module_4.RegexFlag.UNICODE
    assert module_4.MULTILINE == module_4.RegexFlag.MULTILINE
    assert module_4.M == module_4.RegexFlag.MULTILINE
    assert module_4.DOTALL == module_4.RegexFlag.DOTALL
    assert module_4.S == module_4.RegexFlag.DOTALL
    assert module_4.VERBOSE == module_4.RegexFlag.VERBOSE
    assert module_4.X == module_4.RegexFlag.VERBOSE
    assert module_4.TEMPLATE == module_4.RegexFlag.TEMPLATE
    assert module_4.T == module_4.RegexFlag.TEMPLATE
    assert module_4.DEBUG == module_4.RegexFlag.DEBUG
    assert f'{type(module_4.Pattern.pattern).__module__}.{type(module_4.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.flags).__module__}.{type(module_4.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groups).__module__}.{type(module_4.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.Pattern.groupindex).__module__}.{type(module_4.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = [var_0]
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_4 = bool(var_3 == {})

def test_case_6():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]
    var_5 = 5
    var_6 = [var_5]
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}