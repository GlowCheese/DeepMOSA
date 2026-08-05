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

def test_case_4():
    var_0 = 'x = 1  # noqa'
    var_1 = 'y = 2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_4 = 'import os  # noqa: F401'
    var_5 = 'import sys  # noqa: W123'
    var_6 = [var_4, var_5]
    var_7 = module_0.parse_noqa(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.defaultdict'
    assert len(var_7) == 2
    var_8 = 'x = 1  # noqa: E123,W451,F921'
    var_9 = [var_8]
    var_10 = module_0.parse_noqa(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.defaultdict'
    assert len(var_10) == 3
    var_11 = 'x = 1  # NoQA: E123'
    var_12 = [var_11]
    var_13 = module_0.parse_noqa(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.defaultdict'
    assert len(var_13) == 1
    var_14 = 'x = 1  # noqa: E123, W451'
    var_15 = [var_14]
    var_16 = module_0.parse_noqa(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.defaultdict'
    assert len(var_16) == 2
    var_17 = 'import unused  # noqa: F401'
    var_18 = 'y = 1  # noqa: F841'
    var_19 = [var_17, var_18]
    var_20 = module_0.parse_noqa(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.defaultdict'
    assert len(var_20) == 2
    var_21 = 'y = 2  # noqa: E123'
    var_22 = [var_4, var_21]
    var_23 = module_0.parse_noqa(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.defaultdict'
    assert len(var_23) == 2
    var_24 = 'x = 1'
    var_25 = [var_24, var_1]
    var_26 = module_0.parse_noqa(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'collections.defaultdict'
    assert len(var_26) == 0
    var_27 = []
    var_28 = module_0.parse_noqa(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'collections.defaultdict'
    assert len(var_28) == 0

def test_case_5():
    var_0 = 'x = 1  # noqa: F401'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 1
    assert f'{type(module_0.NOQA_REGEXP).__module__}.{type(module_0.NOQA_REGEXP).__qualname__}' == 're.Pattern'
    assert module_0.NOQA_CODE_MAP == {'F401': 'V104', 'F841': 'V107'}
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = 1
    var_4 = 'V104'
    var_5 = module_0.ignore_line(var_2, var_3, var_4)
    assert var_5 is True
    var_6 = 'F401'
    var_7 = module_0.ignore_line(var_2, var_3, var_6)
    assert var_7 is False
    assert len(var_2) == 3
    var_8 = 'y = 2  # noqa'
    var_9 = [var_8]
    var_10 = module_0.parse_noqa(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.defaultdict'
    assert len(var_10) == 1
    var_11 = module_0.ignore_line(var_10, var_3, var_4)
    assert var_11 is True
    assert len(var_10) == 2
    var_12 = 'V107'
    var_13 = module_0.ignore_line(var_10, var_3, var_12)
    assert var_13 is True
    assert len(var_10) == 3
    var_14 = 'AnyCode'
    var_15 = module_0.ignore_line(var_10, var_3, var_14)
    assert var_15 is True
    assert len(var_10) == 4
    var_16 = 'z = 3'
    var_17 = [var_16]
    var_18 = module_0.parse_noqa(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.defaultdict'
    assert len(var_18) == 0
    var_19 = module_0.ignore_line(var_18, var_3, var_4)
    assert var_19 is False
    assert len(var_18) == 2
    var_20 = 'a = 1  # noqa: F401, F841'
    var_21 = [var_20]
    var_22 = module_0.parse_noqa(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.defaultdict'
    assert len(var_22) == 2
    var_23 = module_0.ignore_line(var_22, var_3, var_4)
    assert var_23 is True
    var_24 = module_0.ignore_line(var_22, var_3, var_12)
    assert var_24 is True
    var_25 = module_0.ignore_line(var_22, var_3, var_6)
    assert var_25 is False
    assert len(var_22) == 4
    var_26 = 'E123'
    var_27 = module_0.ignore_line(var_22, var_3, var_26)
    assert var_27 is False
    assert len(var_22) == 5
    var_28 = 'import os  # noqa: F401'
    var_29 = 'unused_var = 5  # noqa: F841'
    var_30 = [var_28, var_29]
    var_31 = module_0.parse_noqa(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'collections.defaultdict'
    assert len(var_31) == 2
    var_32 = module_0.ignore_line(var_31, var_3, var_4)
    assert var_32 is True
    var_33 = module_0.ignore_line(var_31, var_3, var_12)
    assert var_33 is False
    assert len(var_31) == 3
    var_34 = 2
    var_35 = module_0.ignore_line(var_31, var_34, var_12)
    assert var_35 is True
    var_36 = module_0.ignore_line(var_31, var_34, var_4)
    assert var_36 is False
    var_37 = 'import sys  # NoQA: F401'
    var_38 = [var_37]
    var_39 = module_0.parse_noqa(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'collections.defaultdict'
    assert len(var_39) == 1
    var_40 = module_0.ignore_line(var_39, var_3, var_4)
    assert var_40 is True
    var_41 = [var_0]
    var_42 = module_0.parse_noqa(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'collections.defaultdict'
    assert len(var_42) == 1
    var_43 = module_0.ignore_line(var_42, var_34, var_4)
    assert var_43 is False
    assert len(var_42) == 2
    var_44 = 'x = 1  # noqa: F401 , F841'
    var_45 = [var_44]
    var_46 = module_0.parse_noqa(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'collections.defaultdict'
    assert len(var_46) == 2
    var_47 = module_0.ignore_line(var_46, var_3, var_4)
    assert var_47 is True
    var_48 = module_0.ignore_line(var_46, var_3, var_12)
    assert var_48 is True
    var_49 = ''
    var_50 = 'x = 1  # noqa'
    var_51 = [var_49, var_50, var_49]
    var_52 = module_0.parse_noqa(var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'collections.defaultdict'
    assert len(var_52) == 1
    var_53 = module_0.ignore_line(var_52, var_3, var_4)
    assert var_53 is False
    assert len(var_52) == 2
    var_54 = module_0.ignore_line(var_52, var_34, var_4)
    assert var_54 is True
    var_55 = 3
    var_56 = module_0.ignore_line(var_52, var_55, var_4)
    assert var_56 is False
    var_57 = 'y = 2  # noqa: F401'
    var_58 = [var_0, var_57]
    var_59 = module_0.parse_noqa(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'collections.defaultdict'
    assert len(var_59) == 1
    var_60 = module_0.ignore_line(var_59, var_3, var_4)
    assert var_60 is True
    var_61 = module_0.ignore_line(var_59, var_34, var_4)
    assert var_61 is True