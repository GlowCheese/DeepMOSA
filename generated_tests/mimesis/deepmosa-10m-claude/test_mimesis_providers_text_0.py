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
    var_7 = var_4.alphabet()
    var_8 = var_0.hex_color()
    var_9 = var_4.color()
    var_10 = var_0.sentence()
    var_11 = var_4.sentence()
    var_12 = var_5.__str__()
    assert var_12 == 'Text <Locale.EN>'
    var_13 = var_4.alphabet()
    var_0.words(var_3)

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

def test_case_3():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.sentence()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()
    var_2 = module_0.Text()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_2.word()
    var_4 = var_0.level()
    var_5 = None
    module_0.Text(*var_5, **var_5)

def test_case_5():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()
    var_2 = module_0.Text()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_0.title()
    var_4 = var_2.alphabet(var_3)

def test_case_6():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()
    var_2 = module_0.Text()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_2.word()
    var_4 = var_2.alphabet(var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.word()
    var_2 = var_0.title()
    var_3 = var_0.color()
    var_4 = var_0.title()
    var_5 = True
    var_6 = var_0.quote()
    var_7 = var_0.rgb_color(var_5)
    var_8 = var_0.rgb_color()
    var_9 = None
    var_10 = var_0.text()
    var_11 = var_0.answer()
    assert var_11 == 'Maybe'
    var_12 = var_0.emoji(var_9)
    module_0.Text(*var_9)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.color()
    var_2 = []
    var_3 = '\x0c*Q<\tjukmQ3'
    var_4 = None
    var_5 = {var_3: var_4}
    module_1.BaseDataProvider(*var_2, **var_5)

def test_case_9():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()
    var_2 = module_0.Text()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_2.word()
    var_4 = var_2.word()
    var_5 = var_0.answer()

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    assert var_6 == 'No'
    var_7 = var_4.alphabet()
    var_8 = var_0.hex_color()
    var_9 = var_4.color()
    var_10 = var_0.sentence()
    var_11 = var_4.sentence()
    var_12 = var_0.emoji()
    var_13 = {}
    var_14 = module_0.Text()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_14.random).__module__}.{type(var_14.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_14.seed).__module__}.{type(var_14.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_14.locale == 'en'
    var_15 = module_0.Text()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_15.random).__module__}.{type(var_15.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_15.seed).__module__}.{type(var_15.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_15.locale == 'en'
    var_16 = var_15.level()
    var_17 = var_4.word()
    var_18 = var_4.hex_color()
    var_19 = module_0.Text(**var_13)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_19.random).__module__}.{type(var_19.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_19.seed).__module__}.{type(var_19.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_19.locale == 'en'
    var_20 = var_19.text()
    var_21 = var_15.alphabet()
    var_22 = var_5.__str__()
    assert var_22 == 'Text <Locale.EN>'
    var_15.validate_enum(var_3, var_7)

def test_case_11():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color(var_0)