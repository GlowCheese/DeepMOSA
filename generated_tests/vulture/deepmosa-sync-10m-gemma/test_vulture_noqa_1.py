# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import enum as module_0
import vulture.noqa as module_1
import collections as module_2
import re as module_3

def test_case_0():
    pass

def test_case_1():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_1.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'

def test_case_2():
    var_0 = 'import os  # noqa: E401 ,  F401 '
    var_1 = [var_0]
    var_2 = module_1.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 2
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = bool(var_2['Import error'] == {1})
    var_4 = var_2['Unused import']
    var_5 = bool(var_2['Unused import'] == {1})

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'm9rN\nXh'
    var_1 = module_1.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    module_3.escape(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '9'
    var_2 = module_1.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = False
    var_4 = module_1.ignore_line(var_2, var_3, var_0)
    assert var_4 is False
    assert len(var_2) == 2
    var_5 = None
    var_2.__setattr__(var_5, var_4, var_4)

def test_case_5():
    var_0 = 'F401'
    var_1 = 'all'
    var_2 = 1
    var_3 = [var_2]
    var_4 = {var_0: var_3, var_1: var_3}
    var_5 = module_1.ignore_line(var_4, var_2, var_0)
    assert var_5 is True
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}

def test_case_6():
    var_0 = 'import os  # noqa'
    var_1 = 'import sys  # noqa: all'
    var_2 = 'iNmport mh'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_1.parse_noqa(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 1
    assert f'{type(module_1.NOQA_REGEXP).__module__}.{type(module_1.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_1.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4['all']
    var_6 = bool(var_4['all'] == {1, 2})
    assert var_6 is True
    var_7 = len(var_4)
    assert var_7 == 1