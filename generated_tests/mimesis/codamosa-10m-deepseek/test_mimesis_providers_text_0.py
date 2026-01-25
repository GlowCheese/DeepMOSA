# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.text as module_0
import mimesis.providers.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.word()
    var_2 = None
    var_3 = var_0.reseed()
    var_4 = var_0.alphabet(var_3)
    var_5 = var_0.level()
    var_0.validate_enum(var_2, var_2)

def test_case_1():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.sentence()
    var_3 = None
    var_4 = var_1.emoji(var_3)
    var_5 = module_1.BaseProvider(random=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_5.validate_enum(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.emoji(var_0)
    var_3 = var_1.title()
    var_4 = module_1.BaseProvider()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_5 = var_1.hex_color()
    var_4.validate_enum(var_0, var_0)

def test_case_4():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.sentence()

def test_case_5():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.title()
    var_2 = module_1.BaseProvider()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_3 = var_0.hex_color()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.word()
    var_2 = None
    var_3 = var_0.reseed()
    var_4 = var_0.quote()
    var_5 = var_0.alphabet(var_3)
    var_6 = var_0.level()
    var_0.validate_enum(var_2, var_2)

def test_case_7():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.word()
    var_2 = var_0.word()
    var_3 = module_0.Text()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    var_4 = module_0.Text()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_4.locale == 'en'
    var_5 = var_4.answer()
    var_6 = var_3.alphabet()
    var_7 = var_0.hex_color()
    var_8 = var_3.color()
    var_9 = var_0.sentence()
    var_10 = var_3.sentence()
    assert var_10 == 'Do you come here often?'
    var_11 = var_0.emoji()
    var_12 = module_0.Text()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_12.random).__module__}.{type(var_12.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_12.seed).__module__}.{type(var_12.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_12.locale == 'en'
    var_13 = var_12.level()
    var_14 = var_12.sentence()

def test_case_8():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color(var_0)

def test_case_9():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.level()
    var_2 = var_0.level()
    var_3 = module_0.Text()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    var_4 = var_0.rgb_color()
    var_5 = var_3.alphabet(var_4)
    var_6 = var_0.sentence()
    var_7 = var_3.answer()
    var_8 = var_0.level()
    assert var_8 == 'extreme'
    var_9 = var_3.sentence()
    var_10 = var_0.__str__()
    assert var_10 == 'Text <Locale.EN>'
    var_11 = var_3.rgb_color()
    var_12 = var_3.color()
    var_13 = var_0.word()
    var_14 = module_0.Text()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_14.random).__module__}.{type(var_14.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_14.seed).__module__}.{type(var_14.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_14.locale == 'en'
    var_15 = var_0.words()
    var_16 = module_0.Text()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_16.random).__module__}.{type(var_16.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_16.seed).__module__}.{type(var_16.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_16.locale == 'en'
    var_17 = var_16.words()
    var_18 = var_16.alphabet()