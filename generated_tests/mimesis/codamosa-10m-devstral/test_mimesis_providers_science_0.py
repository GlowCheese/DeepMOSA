# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.science as module_0
import mimesis.providers.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = module_0.Science(random=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_3 = var_2.measure_unit(symbol=var_0)
    var_4 = 933
    var_5 = module_0.Science(seed=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed == 933
    var_6 = var_5.rna_sequence()
    assert var_6 == 'GUGGGUCUGG'
    var_7 = True
    var_8 = 123
    var_9 = None
    var_10 = (var_7, var_8, var_9, var_7)
    var_11 = module_0.Science()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_11.random).__module__}.{type(var_11.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_11.seed).__module__}.{type(var_11.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_11.dna_sequence(var_10)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = False
    var_1 = module_0.Science(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is False
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2 = var_1.metric_prefix(symbol=var_0)
    assert var_2 == 'femto'
    var_3 = var_1.reseed()
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_4 = None
    var_5 = module_1.BaseProvider(random=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_6 = var_1.measure_unit(var_4, var_0)
    assert var_6 == 'byte'
    var_5.validate_enum(var_4, var_4)

def test_case_2():
    var_0 = None
    var_1 = module_0.Science(random=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2 = var_1.metric_prefix(symbol=var_0)
    var_3 = var_1.dna_sequence()
    var_4 = var_1.dna_sequence()

def test_case_3():
    var_0 = None
    var_1 = module_0.Science(random=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Science()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_1.rna_sequence(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = b'\x17\xa7\\@\x15_\xaa\xe7'
    var_2 = module_0.Science(seed=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert var_2.seed == b'\x17\xa7\\@\x15_\xaa\xe7'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_2.dna_sequence(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Science()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.SI_PREFIXES == {'negative': ['deci', 'centi', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto'], 'positive': ['yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo', 'hecto', 'deca']}
    assert module_0.SI_PREFIXES_SYM == {'negative': ['d', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'], 'positive': ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da']}
    var_1 = var_0.rna_sequence()
    var_2 = var_0.measure_unit()
    var_3 = True
    var_4 = var_0.rna_sequence(var_3)
    var_5 = var_0.metric_prefix(symbol=var_4)
    var_6 = var_0.measure_unit()
    var_7 = True
    var_8 = var_0.rna_sequence(var_7)
    var_9 = module_0.Science()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.providers.science.Science'
    assert f'{type(var_9.random).__module__}.{type(var_9.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_9.seed).__module__}.{type(var_9.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_10 = var_9.metric_prefix()
    var_11 = var_9.__str__()
    assert var_11 == 'Science'
    module_0.Science(random=var_10)