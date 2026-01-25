# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.text as module_0
import mimesis.providers.base as module_1

def test_case_0():
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
    assert var_5 == 'Yes'
    var_6 = var_3.alphabet()
    var_7 = var_0.hex_color()
    var_8 = var_3.color()
    var_9 = var_0.sentence()
    var_10 = var_3.sentence()
    var_11 = var_0.emoji()
    var_12 = module_0.Text()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_12.random).__module__}.{type(var_12.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_12.seed).__module__}.{type(var_12.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_12.locale == 'en'
    var_13 = var_12.level()
    var_14 = var_12.sentence()

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
    var_0 = False
    var_1 = {var_0}
    var_2 = None
    var_3 = module_0.Text()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_4 = var_3.emoji(var_2)
    var_5 = (var_0, var_1)
    module_0.Text(**var_5)

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
    assert var_1 == 'Initially composing light-hearted and irreverent works, he also wrote serious, sombre and religious pieces beginning in the 1930s.'
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
    module_1.BaseDataProvider(var_2)

@pytest.mark.xfail(strict=True)
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
    var_5 = var_3.alphabet()
    var_6 = var_0.hex_color()
    var_7 = var_3.color()
    var_8 = var_0.sentence()
    var_9 = var_3.sentence()
    var_10 = var_0.emoji()
    var_11 = module_0.Text()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_11.random).__module__}.{type(var_11.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_11.seed).__module__}.{type(var_11.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_11.locale == 'en'
    var_12 = var_11.level()
    var_13 = var_4.color()
    var_14 = var_3.quote()
    var_15 = var_0.text()
    var_16 = None
    var_0.words(var_16)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.word()
    var_2 = var_0.word()
    var_3 = None
    var_4 = module_0.Text()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_4.locale == 'en'
    var_5 = module_0.Text()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_5.locale == 'en'
    var_6 = var_5.answer()
    var_7 = True
    var_8 = var_0.hex_color()
    var_9 = var_4.color()
    var_10 = var_0.sentence()
    var_11 = var_4.sentence()
    var_12 = var_0.emoji()
    var_13 = {}
    var_14 = True
    var_15 = var_0.rgb_color(var_14)
    var_16 = module_0.Text()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_16.random).__module__}.{type(var_16.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_16.seed).__module__}.{type(var_16.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_16.locale == 'en'
    var_17 = var_16.level()
    var_18 = var_4.word()
    var_19 = var_4.hex_color()
    var_20 = module_0.Text(**var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_20.random).__module__}.{type(var_20.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_20.seed).__module__}.{type(var_20.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_20.locale == 'en'
    var_21 = var_20.text()
    var_22 = var_16.alphabet()
    var_23 = var_20.text(var_7)
    var_16.validate_enum(var_3, var_22)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.level()
    assert var_1 == 'moderate'
    var_2 = var_0.level()
    var_3 = module_0.Text()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_3.locale == 'en'
    var_4 = var_0.rgb_color()
    var_5 = var_3.alphabet(var_3)
    var_6 = var_0.sentence()
    var_7 = var_3.color()
    assert var_7 == 'Purple'
    var_8 = var_0.level()
    var_9 = var_3.sentence()
    var_10 = var_3.emoji()
    var_11 = var_3.rgb_color()
    var_12 = module_0.Text()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_12.random).__module__}.{type(var_12.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_12.seed).__module__}.{type(var_12.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_12.locale == 'en'
    var_13 = var_0.__str__()
    assert var_13 == 'Text <Locale.EN>'
    var_14 = var_12.__str__()
    assert var_14 == 'Text <Locale.EN>'
    var_15 = var_12.sentence()
    var_16 = module_0.Text()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_16.random).__module__}.{type(var_16.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_16.seed).__module__}.{type(var_16.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_16.locale == 'en'
    var_17 = var_0.__str__()
    assert var_17 == 'Text <Locale.EN>'
    var_18 = var_16.word()
    var_19 = var_12.text()
    var_20 = var_3.color()
    var_21 = -34
    var_22 = var_3.text(var_21)
    assert var_22 == ''
    var_23 = var_3.quote()
    var_24 = var_16.text()
    var_25 = None
    var_26 = var_16.words()
    var_27 = module_1.BaseProvider(seed=var_8)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_27.random).__module__}.{type(var_27.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_5.validate_enum(var_25, var_25)