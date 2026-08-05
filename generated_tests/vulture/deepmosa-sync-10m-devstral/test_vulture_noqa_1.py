# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import enum as module_1
import collections as module_2

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
    var_0 = 'x = 1  # noqa: E123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_3():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '9'
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = False
    var_4 = module_0.ignore_line(var_2, var_3, var_0)
    assert var_4 is False
    assert len(var_2) == 2
    var_5 = None
    var_2.__setattr__(var_5, var_4, var_4)

def test_case_5():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 1
    var_6 = 2
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_2, var_0)
    assert var_9 is True
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}

def test_case_6():
    var_0 = 'x = 1  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2['all']
    var_4 = bool(var_2['all'] == {1})
    assert var_4 is True