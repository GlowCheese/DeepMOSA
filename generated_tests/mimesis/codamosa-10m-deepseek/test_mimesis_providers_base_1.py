# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.base as module_0
import mimesis.exceptions as module_1
import mimesis.random as module_2

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_1():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.reseed()
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.BaseDataProvider()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1.validate_enum(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.__str__()
    assert var_2 == 'BaseDataProvider <Locale.EN>'
    var_1.validate_enum(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.BaseProvider()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = 'gN\x0bMtC'
    var_3 = {var_2: var_2, var_2: var_0}
    var_1.validate_enum(var_3, var_0)

def test_case_5():
    var_0 = module_0.BaseProvider()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = {}
    with pytest.raises(module_1.NonEnumerableError):
        var_0.validate_enum(var_1, var_1)

def test_case_6():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.reseed()
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'en'
    var_4 = module_0.BaseDataProvider()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_4.locale == 'en'
    var_5 = var_4.reseed(var_0)
    assert var_4.seed is None

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = 'x7->Q=8B#H'
    var_3 = "U\rOs-u'"
    var_4 = {var_2: var_1, var_2: var_2, var_3: var_0}
    var_5 = var_1.update_dataset(var_4)
    var_1.validate_enum(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = module_0.BaseProvider()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = 'gN\x0bMC'
    var_4 = {var_3: var_3, var_3: var_0}
    var_5 = var_2.__str__()
    assert var_5 == 'BaseProvider'
    var_6 = 'x7->Q=8B#H'
    var_7 = "U\rOs-u'"
    var_8 = {var_7: var_1, var_6: var_6, var_7: var_0}
    var_9 = var_1.update_dataset(var_8)
    var_2.validate_enum(var_4, var_9)

def test_case_9():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.__str__()
    assert var_2 == 'BaseDataProvider <Locale.EN>'
    with pytest.raises(TypeError):
        var_1.update_dataset(var_0)

def test_case_10():
    var_0 = 'BGJJK"'
    with pytest.raises(TypeError):
        module_0.BaseProvider(random=var_0)

def test_case_11():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed == 42
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = module_2.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_2.global_seed).__module__}.{type(module_2.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.random).__module__}.{type(module_2.random).__qualname__}' == 'mimesis.random.Random'
    assert module_2.random.gauss_next is None
    var_3 = module_0.BaseProvider(random=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_4 = module_0.BaseProvider(seed=var_0, random=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed == 42

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.override_locale(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_2.gen).__module__}.{type(var_2.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(var_2.args).__module__}.{type(var_2.args).__qualname__}' == 'builtins.tuple'
    assert len(var_2.args) == 2
    assert var_2.kwds == {}
    var_2.__enter__()