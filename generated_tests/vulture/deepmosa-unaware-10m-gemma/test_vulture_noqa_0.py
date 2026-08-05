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
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

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
    var_0 = module_4.RegexFlag.DOTALL
    var_1 = var_0.__dir__()
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
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    var_3 = var_2.__setitem__(var_0, var_2)
    assert len(var_2) == 1
    var_4 = module_0.ignore_line(var_2, var_0, var_0)
    assert var_4 is True
    module_0.parse_noqa(var_2)

def test_case_6():
    var_0 = 'import os  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = 'import a  # noqa: E1, E2'
    var_4 = 'import b  # noqa: E2, E3'
    var_5 = 'import c  # noqa'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.parse_noqa(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.defaultdict'
    assert len(var_7) == 4
    var_8 = 'import os'
    var_9 = "print('hello')"
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_noqa(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.defaultdict'
    assert len(var_11) == 0
    var_12 = 'import sys  # noqa: F401'
    var_13 = [var_12]
    var_14 = module_0.parse_noqa(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.defaultdict'
    assert len(var_14) == 1