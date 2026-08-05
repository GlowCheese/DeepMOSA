# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.noqa as module_0
import enum as module_1
import collections as module_2
import builtins as module_3

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
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_0.__dir__()
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2.scan(var_0)

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
    var_0 = "print('hello')"
    var_1 = 'x = 1'
    var_2 = 'y = 2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_noqa(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 0
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_5 = set()
    var_6 = "print('hello')  # noqa"
    var_7 = [var_6, var_1, var_2]
    var_8 = module_0.parse_noqa(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.defaultdict'
    assert len(var_8) == 1
    var_9 = "print('hello')  # noqa: F401"
    var_10 = 'x = 1  # noqa: F841'
    var_11 = [var_9, var_10, var_2]
    var_12 = module_0.parse_noqa(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.defaultdict'
    assert len(var_12) == 2
    var_13 = "print('hello')  # noqa: F401, F841"
    var_14 = [var_13, var_1, var_2]
    var_15 = module_0.parse_noqa(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.defaultdict'
    assert len(var_15) == 2
    var_16 = "print('hello')  # NoQa: F401"
    var_17 = 'x = 1  # NOQA'
    var_18 = [var_16, var_17, var_2]
    var_19 = module_0.parse_noqa(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.defaultdict'
    assert len(var_19) == 2
    var_20 = 'y = 2  # noqa'
    var_21 = [var_6, var_10, var_20]
    var_22 = module_0.parse_noqa(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.defaultdict'
    assert len(var_22) == 2
    var_23 = "print('hello')  # noqa: E123"
    var_24 = [var_23, var_1, var_2]
    var_25 = module_0.parse_noqa(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'collections.defaultdict'
    assert len(var_25) == 1