# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.providers.base as module_1
import mimesis.providers.text as module_0
import pytest


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Text(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_3 = var_2.words()
    var_4 = var_2.__str__()
    assert var_4 == 'Text <Locale.EN>'
    var_5 = None
    var_6 = var_2.alphabet(var_5)
    var_7 = module_0.Text()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_7.seed).__module__}.{type(var_7.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_7.locale == 'en'

def test_case_1():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = var_0.rgb_color()

def test_case_2():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.rgb_color(var_1)

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
    var_7 = module_0.Text()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_7.seed).__module__}.{type(var_7.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_7.locale == 'en'
    var_8 = var_7.emoji(var_1)
    var_9 = var_7.rgb_color()
    var_10 = var_7.alphabet()
    var_11 = var_7.answer()
    var_12 = var_0.hex_color()
    var_13 = module_0.Text()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_13.random).__module__}.{type(var_13.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_13.seed).__module__}.{type(var_13.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_13.locale == 'en'
    var_14 = [var_1, var_1]
    module_0.Text(*var_14)

def test_case_4():
    var_0 = module_0.Text()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_1 = None
    var_2 = var_0.color()
    var_3 = var_0.title()
    assert var_3 == 'The sequential subset of Erlang supports eager evaluation, single assignment, and dynamic typing.'
    var_4 = var_0.alphabet(var_1)
    var_5 = var_0.word()
    var_6 = var_0.emoji()
    var_7 = var_0.alphabet(var_1)
    var_8 = var_0.answer()
    var_9 = var_0.hex_color()
    var_10 = module_0.Text()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_10.random).__module__}.{type(var_10.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_10.seed).__module__}.{type(var_10.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_10.locale == 'en'
    var_11 = module_0.Text()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_11.random).__module__}.{type(var_11.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_11.seed).__module__}.{type(var_11.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_11.locale == 'en'
    var_12 = var_0.title()
    var_13 = var_11.sentence()
    assert var_13 == 'It is also a garbage-collected runtime system.'
    var_14 = {var_2: var_12, var_2: var_9, var_8: var_8}
    var_15 = var_11.update_dataset(var_14)
    var_16 = var_0.rgb_color()

@pytest.mark.xfail(strict=True)
def test_case_5():
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

def test_case_7():
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
    var_5 = var_1.title()
    var_6 = var_1.words()
    var_7 = var_1.word()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = []
    var_1 = module_0.Text(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.word()
    var_3 = var_1.answer()
    var_4 = {}
    var_5 = var_1.sentence()
    var_6 = var_1.update_dataset(var_4)
    var_7 = False
    var_4.rgb_color(var_7)

def test_case_9():
    var_0 = True
    var_1 = module_0.Text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    assert module_0.SAFE_COLORS == ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad', '#34495e', '#2c3e50', '#f1c40f', '#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
    var_2 = var_1.color()
    var_3 = []
    var_4 = module_0.Text(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_4.locale == 'en'
    var_5 = var_4.rgb_color(var_0)

def test_case_10():
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
    var_5 = -735
    var_6 = var_0.text(var_5)
    assert var_6 == ''
    var_7 = var_0.word()
    var_8 = True
    var_9 = var_0.alphabet(var_8)
    var_10 = var_0.answer()
    var_11 = var_0.hex_color(var_1)
    var_12 = module_0.Text()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_12.random).__module__}.{type(var_12.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_12.seed).__module__}.{type(var_12.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_12.locale == 'en'
    var_13 = var_0.title()
    var_14 = var_12.sentence()
    var_15 = var_12.rgb_color()
    var_16 = module_0.Text()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'mimesis.providers.text.Text'
    assert f'{type(var_16.random).__module__}.{type(var_16.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_16.seed).__module__}.{type(var_16.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_16.locale == 'en'
    var_17 = var_16.rgb_color()
    var_18 = var_16.text()
    var_19 = var_0.text()
    var_20 = {}
    var_21 = module_1.BaseDataProvider(**var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'mimesis.providers.base.BaseDataProvider'
    assert f'{type(var_21.random).__module__}.{type(var_21.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_21.seed).__module__}.{type(var_21.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_21.locale == 'en'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_22 = var_16.__str__()
    assert var_22 == 'Text <Locale.EN>'
    var_23 = var_12.sentence()