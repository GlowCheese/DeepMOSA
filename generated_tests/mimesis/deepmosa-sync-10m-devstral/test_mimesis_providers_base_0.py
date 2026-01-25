# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.base as module_0
import mimesis.exceptions as module_1
import mimesis.random as module_2
import re as module_3

def test_case_0():
    pass

def test_case_1():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

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
    var_1 = var_0.reseed()
    var_2 = ()
    var_3 = module_0.BaseDataProvider(seed=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed is None
    assert var_3.locale == 'en'
    with pytest.raises(module_1.NonEnumerableError):
        var_0.validate_enum(var_0, var_2)

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
    var_1 = {}
    var_2 = var_0.update_dataset(var_1)
    var_3 = var_0.override_locale(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_3.gen).__module__}.{type(var_3.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(var_3.args).__module__}.{type(var_3.args).__qualname__}' == 'builtins.tuple'
    assert len(var_3.args) == 2
    assert var_3.kwds == {}
    var_3.__enter__()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_0.validate_enum(var_0, var_0)

def test_case_6():
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
    var_2 = var_0.__str__()
    assert var_2 == 'BaseDataProvider <Locale.EN>'
    var_3 = '5'
    var_4 = {var_3: var_3}
    var_5 = module_0.BaseProvider(seed=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed == 'en'
    var_6 = var_5.__str__()
    assert var_6 == 'BaseProvider'
    var_7 = var_0.update_dataset(var_4)
    var_0.validate_enum(var_3, var_7)

def test_case_8():
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
    var_2 = ()
    with pytest.raises(module_1.NonEnumerableError):
        var_0.validate_enum(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    with pytest.raises(TypeError):
        var_0.update_dataset(var_1)

def test_case_11():
    var_0 = module_2.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_2.global_seed).__module__}.{type(module_2.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.random).__module__}.{type(module_2.random).__qualname__}' == 'mimesis.random.Random'
    assert module_2.random.gauss_next is None
    var_1 = module_0.BaseProvider(random=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

def test_case_12():
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
    var_2 = var_0.__str__()
    assert var_2 == 'BaseDataProvider <Locale.EN>'
    var_3 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_4 = {var_1: var_3}
    var_5 = var_0.update_dataset(var_4)
    var_6 = var_0.reseed()
    with pytest.raises(TypeError):
        module_0.BaseProvider(random=var_1)

def test_case_13():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = ()
    with pytest.raises(module_1.NonEnumerableError):
        var_0.validate_enum(var_1, var_1)

def test_case_14():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = ()
    with pytest.raises(module_1.NonEnumerableError):
        var_0.validate_enum(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.override_locale(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(var_1.args).__module__}.{type(var_1.args).__qualname__}' == 'builtins.tuple'
    assert len(var_1.args) == 2
    assert var_1.kwds == {}
    var_1.__enter__()