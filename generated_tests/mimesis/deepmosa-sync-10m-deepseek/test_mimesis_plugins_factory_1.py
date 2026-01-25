# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.plugins.factory as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'Nj"w<=/%zL4_TL'
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field == 'Nj"w<=/%zL4_TL'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = module_0.FactoryField(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {}
    assert var_3.field == 'Nj"w<=/%zL4_TL'
    var_1.evaluate(var_2, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = '~[2,[HqQc$Ldf'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.FactoryField(var_0, var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {'~[2,[HqQc$Ldf': None}
    assert var_3.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3.evaluate(var_0, var_0)

def test_case_2():
    pass

def test_case_3():
    var_0 = None
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'tst_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_2.locale is None
    assert var_2.kwargs == {}
    assert var_2.field == 'tst_field'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3 = 'builder'
    var_4 = ()
    var_5 = 'factory_meta'
    var_6 = ()
    var_7 = 'declarations'
    var_8 = {var_7: var_1}
    var_9 = type(var_7, var_6, var_8)
    var_10 = {var_5: var_9}
    var_11 = type(var_5, var_4, var_10)
    var_12 = {var_3: var_11}
    var_13 = type(var_3, var_6, var_12)
    var_14 = var_13()
    var_2.evaluate(var_11, var_14)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = '\\f\r-'
    var_2 = ()
    var_3 = 'builder'
    var_4 = ()
    var_5 = 'factory_meta'
    var_6 = ()
    var_7 = 'declarations'
    var_8 = {var_5: var_0, var_3: var_0, var_7: var_0, var_5: var_0}
    var_9 = type(var_1, var_6, var_8)
    var_10 = {var_5: var_9}
    var_11 = type(var_7, var_4, var_10)
    var_12 = module_0.FactoryField(var_11, var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_12.kwargs == {}
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_13 = {var_3: var_11}
    var_14 = type(var_1, var_2, var_13)
    var_15 = var_14()
    var_12.evaluate(var_9, var_15)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_2.locale is None
    assert var_2.kwargs == {}
    assert var_2.field == 'test_field'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3 = ()
    var_4 = 'builder'
    var_5 = 'Builder'
    var_6 = ()
    var_7 = 'factory_meta'
    var_8 = ()
    var_9 = 'declarations'
    var_10 = 'field_handlers'
    var_11 = [var_8, var_5]
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = type(var_4, var_8, var_13)
    var_15 = module_0.FactoryField(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_15.locale is None
    assert var_15.kwargs == {}
    assert var_15.field == 'field_handlers'
    var_16 = {var_7: var_14}
    var_17 = type(var_5, var_6, var_16)
    var_18 = {var_4: var_17}
    var_19 = type(var_5, var_3, var_18)
    var_20 = var_19()
    var_21 = None
    var_2.evaluate(var_21, var_20)