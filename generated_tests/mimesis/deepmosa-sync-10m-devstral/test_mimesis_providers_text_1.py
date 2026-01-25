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
    var_1 = None
    var_2 = var_0.color()
    var_3 = var_0.title()
    var_4 = var_0.alphabet(var_1)
    var_5 = var_0.word()
    var_6 = var_0.sentence()
    module_0.Text(*var_4)

def test_case_1():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()

def test_case_2():
    var_0 = True
    var_1 = []
    var_2 = module_0.Text(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_3 = var_2.rgb_color(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = None
    var_2 = var_0.color()
    var_3 = var_0.title()
    var_4 = var_0.alphabet(var_1)
    var_5 = var_0.word()
    var_6 = var_0.emoji()
    var_7 = var_0.alphabet(var_1)
    module_1.BaseDataProvider(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.quote()
    var_3 = var_1.title()
    var_4 = var_1.level()
    var_5 = var_1.hex_color()
    var_1.validate_enum(var_0, var_0)

def test_case_5():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = True
    var_3 = var_1.rgb_color(var_2)
    var_4 = var_1.title()
    var_5 = var_1.word()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.quote()
    var_2 = None
    var_3 = module_1.BaseDataProvider()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_4 = module_0.Text()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_4.locale == 'en'
    var_5 = var_4.sentence()
    var_3.validate_enum(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.word()
    var_3 = var_1.answer()
    assert var_3 == 'Maybe'
    var_4 = {}
    var_5 = var_1.sentence()
    var_6 = var_1.update_dataset(var_4)
    var_7 = False
    var_4.rgb_color(var_7)

def test_case_8():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.color()
    assert var_2 == 'Red'
    var_3 = module_0.Text()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    var_4 = 1942
    var_5 = var_3.words(var_4)
    var_6 = var_3.__str__()
    assert var_6 == 'Text <Locale.EN>'
    var_7 = None
    var_8 = var_3.emoji(var_7)
    var_9 = False
    var_10 = var_1.hex_color(var_9)
    var_11 = var_3.word()
    var_12 = var_3.answer()
    var_13 = var_3.quote()
    assert var_13 == 'Let them eat cake.'
    var_14 = var_3.title()
    var_15 = var_3.rgb_color()
    var_16 = var_3.words()
    var_17 = var_1.__str__()
    assert var_17 == 'Text <Locale.EN>'
    var_18 = var_3.sentence()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = None
    var_2 = var_0.title()
    var_3 = var_0.quote()
    var_4 = True
    var_5 = var_0.alphabet(var_4)
    var_6 = var_0.word()
    var_7 = var_0.emoji(var_1)
    var_8 = var_0.alphabet()
    var_9 = var_0.answer()
    var_10 = var_0.hex_color()
    var_11 = [var_1, var_1]
    var_12 = {}
    module_0.Text(*var_11, **var_12)