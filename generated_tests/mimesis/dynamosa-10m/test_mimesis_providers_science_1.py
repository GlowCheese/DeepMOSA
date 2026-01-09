# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.providers.base as module_1
import mimesis.providers.science as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Science(random=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_3 = var_2.__str__()
    assert var_3 == 'Science'
    var_4 = var_2.measure_unit(symbol=var_1)
    var_5 = module_0.Science(random=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_6 = True
    var_7 = var_5.measure_unit(var_1, var_6)
    var_8 = var_2.measure_unit()
    var_9 = module_1.BaseProvider()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_9.random).__module__}.{type(var_9.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_9.seed).__module__}.{type(var_9.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_2.validate_enum(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.Science()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2 = var_1.metric_prefix()
    var_1.validate_enum(var_0, var_0)

def test_case_2():
    var_0 = module_0.Science()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}

def test_case_3():
    var_0 = True
    var_1 = module_0.Science(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is True
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2 = None
    var_3 = var_1.metric_prefix(symbol=var_2)
    assert var_3 == 'deca'
    var_4 = var_1.dna_sequence()
    assert var_4 == 'AACCCGATTA'
    var_5 = module_0.Science()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_6 = var_5.metric_prefix()
    var_7 = var_5.rna_sequence()

def test_case_4():
    var_0 = module_0.Science()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_1 = var_0.dna_sequence()
    var_2 = var_0.dna_sequence()
    var_3 = None
    var_4 = module_0.Science(random=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_5 = module_0.Science(seed=var_3, random=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed is None
    var_6 = var_4.metric_prefix()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 370
    var_1 = module_0.Science(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed == 370
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2 = var_1.metric_prefix()
    assert var_2 == 'tera'
    var_3 = None
    var_4 = var_1.dna_sequence()
    assert var_4 == 'GCACGGATAG'
    var_5 = -2097
    var_6 = var_1.rna_sequence()
    assert var_6 == 'GCGCGUCGAG'
    var_7 = var_1.measure_unit()
    assert var_7 == 'newton'
    var_8 = None
    var_9 = module_1.BaseProvider(seed=var_5, random=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_9.random).__module__}.{type(var_9.random).__qualname__}' == 'mimesis.random.Random'
    assert var_9.seed == -2097
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_10 = module_0.Science()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_10.random).__module__}.{type(var_10.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_10.seed).__module__}.{type(var_10.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_11 = var_1.measure_unit()
    assert var_11 == 'mole'
    var_12 = var_1.measure_unit()
    assert var_12 == 'pascal'
    var_13 = True
    var_14 = var_1.metric_prefix(symbol=var_13)
    assert var_14 == 'G'
    var_10.dna_sequence(var_3)