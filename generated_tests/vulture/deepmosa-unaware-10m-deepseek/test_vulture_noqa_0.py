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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.parse_noqa(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.__dir__()
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0
    module_4.compile(var_3)

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
    var_0 = 'x = 1'
    var_1 = 'import module'
    var_2 = [var_1, var_0]
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_4 = 'a = 1  # noqa: F401 F841'
    var_5 = 'b = 2  # noqa: F401,F841'
    var_6 = 'c = 3  # noqa: F401, F841'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.parse_noqa(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.defaultdict'
    assert len(var_8) == 3
    var_9 = []
    var_10 = module_0.parse_noqa(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.defaultdict'
    assert len(var_10) == 0

def test_case_7():
    var_0 = 'import unused_module  # noqa'
    var_1 = 'x = 1'
    var_2 = 'y = 2  # noqa: F401'
    var_3 = 'w = 4  # noqa: F401,F841'
    var_4 = 'v = 5  # NoQA: F401'
    var_5 = 'u = 6  # noqa: E501'
    var_6 = [var_0, var_1, var_2, var_1, var_3, var_4, var_5]
    var_7 = module_0.parse_noqa(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.defaultdict'
    assert len(var_7) == 4
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_8 = 'import module'
    var_9 = [var_8, var_1]
    var_10 = module_0.parse_noqa(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.defaultdict'
    assert len(var_10) == 0
    var_11 = 'a = 1  # noqa: F401 F841'
    var_12 = 'b = 2  # noqa: F401,F841'
    var_13 = 'c = 3  # noqa: F401, F841'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.parse_noqa(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.defaultdict'
    assert len(var_15) == 3
    var_16 = []
    var_17 = module_0.parse_noqa(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.defaultdict'
    assert len(var_17) == 0