# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.plugins.factory as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_1.evaluate(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = "`^?9:>l'"
    var_2 = module_0.FactoryField(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_2.locale is None
    assert var_2.kwargs == {}
    assert var_2.field == "`^?9:>l'"
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3 = var_2.touch_creation_counter()
    var_4 = module_0.FactoryField(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_4.locale is None
    assert var_4.kwargs == {}
    assert var_4.field is None
    var_4.evaluate(var_0, var_0)

def test_case_2():
    pass

def test_case_3():
    var_0 = '[rcFG'
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field == '[rcFG'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = {}
    var_1 = ()
    var_2 = 'builder'
    var_3 = 'factory_meta'
    var_4 = 'declarations'
    var_5 = 'field_handlers'
    var_6 = {var_4: var_0}
    var_7 = type(var_5, var_1, var_6)
    var_8 = {var_3: var_7}
    var_9 = type(var_4, var_1, var_8)
    var_10 = {var_2: var_9}
    var_11 = type(var_2, var_1, var_10)
    var_12 = module_0.FactoryField(var_0, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_12.kwargs == {}
    assert var_12.field == {}
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_13 = var_11()
    var_14 = None
    var_15 = None
    var_12.evaluate(var_15, var_13, var_14)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_2.locale is None
    assert var_2.kwargs == {}
    assert var_2.field == 'test_field'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3 = 'Step'
    var_4 = ()
    var_5 = 'builder'
    var_6 = 'factory_meta'
    var_7 = 'Meta'
    var_8 = 'declarations'
    var_9 = 'field_handlers'
    var_10 = []
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = type(var_7, var_4, var_12)
    var_14 = {var_6: var_13}
    var_15 = type(var_3, var_4, var_14)
    var_16 = {var_5: var_15}
    var_17 = type(var_3, var_4, var_16)
    var_18 = var_17()
    var_19 = None
    var_20 = None
    var_2.evaluate(var_20, var_18, var_19)

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
    var_5 = 'factory_meta'
    var_6 = 'Meta'
    var_7 = 'declarations'
    var_8 = 'field_handlers'
    var_9 = {var_8: var_4}
    var_10 = {var_7: var_9}
    var_11 = type(var_6, var_3, var_10)
    var_12 = {var_5: var_11}
    var_13 = type(var_5, var_3, var_12)
    var_14 = {var_4: var_13}
    var_15 = type(var_6, var_3, var_14)
    var_16 = var_15()
    var_17 = None
    var_2.evaluate(var_15, var_16, var_17)