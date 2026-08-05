# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import collections as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.parse_noqa(var_0)

def test_case_1():
    var_0 = 'x = 1  # noqa: F401'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_2():
    var_0 = 'x = 1  # noqa: E301, E302'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 2
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = None
    var_4 = module_0.ignore_line(var_2, var_3, var_3)
    assert var_4 is False
    assert len(var_2) == 4
    var_5 = bool(var_2 == {'E301': {1}, 'E302': {1}})

def test_case_3():
    var_0 = 'x = 1  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_4():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_5():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}