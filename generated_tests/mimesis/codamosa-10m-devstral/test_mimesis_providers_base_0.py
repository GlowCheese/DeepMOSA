# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.base as module_0
import re as module_1
import mimesis.exceptions as module_2
import mimesis.random as module_3

def test_case_0():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_1():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = b'\x99\xb1\x04\x1cH[\xc6\x90\x9c\xfaks\x06\xfea\x99-\x10\x9dW'
    var_2 = var_0.reseed(var_1)
    assert var_0.seed == b'\x99\xb1\x04\x1cH[\xc6\x90\x9c\xfaks\x06\xfea\x99-\x10\x9dW'

def test_case_2():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = var_0.update_dataset(var_1)
    var_0.validate_enum(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_0.validate_enum(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.__str__()
    assert var_1 == 'BaseDataProvider <Locale.EN>'
    var_2 = {var_1: var_1, var_1: var_0, var_1: var_1, var_1: var_1}
    var_0.validate_enum(var_2, var_2)

def test_case_6():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.__str__()
    assert var_1 == 'BaseProvider'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.get_current_locale()
    assert var_1 == 'en'
    var_2 = {var_1: var_1, var_1: var_0, var_1: var_1, var_1: var_1}
    var_0.validate_enum(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = None
    var_0.validate_enum(var_1, var_1)

def test_case_9():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    with pytest.raises(TypeError):
        var_0.update_dataset(var_1)

def test_case_10():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.reseed()
    var_2 = ''
    var_3 = var_0.reseed()
    var_4 = var_0.reseed()
    with pytest.raises(module_2.NonEnumerableError):
        var_0.validate_enum(var_2, var_1)

def test_case_11():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed == 42
    var_4 = var_3.random
    var_5 = module_3.Random()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.random.Random'
    assert var_5.gauss_next is None
    assert f'{type(module_3.MissingSeed).__module__}.{type(module_3.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_3.Seed).__module__}.{type(module_3.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_3.global_seed).__module__}.{type(module_3.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_3.random).__module__}.{type(module_3.random).__qualname__}' == 'mimesis.random.Random'
    assert module_3.random.gauss_next is None
    var_6 = module_0.BaseProvider(random=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = 'not_a_random_object'
    with pytest.raises(TypeError):
        module_0.BaseProvider(random=var_7)