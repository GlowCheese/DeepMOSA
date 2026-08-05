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
    var_0 = '9'
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_3():
    var_0 = None
    var_1 = 'Hx/\ti!} ){'
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
    var_3 = var_2.__setitem__(var_0, var_2)
    assert len(var_2) == 1
    var_4 = module_0.ignore_line(var_2, var_0, var_0)
    assert var_4 is True
    module_0.parse_noqa(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = "print('hello')"
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_4 = "print('hello')  # noqa"
    var_5 = [var_4, var_1]
    var_6 = module_0.parse_noqa(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.defaultdict'
    assert len(var_6) == 1
    var_7 = "print('hello')  # noqa: F401"
    var_8 = 'x = 1  # noqa: F841'
    var_9 = [var_7, var_8]
    var_10 = module_0.parse_noqa(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.defaultdict'
    assert len(var_10) == 2
    var_11 = "print('hello')  # noqa: F401, F841"
    var_12 = [var_11, var_1]
    var_13 = module_0.parse_noqa(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.defaultdict'
    assert len(var_13) == 2
    var_14 = 'x = 1  # NoQa: F841'
    var_15 = [var_3, var_14]
    module_0.parse_noqa(var_15)