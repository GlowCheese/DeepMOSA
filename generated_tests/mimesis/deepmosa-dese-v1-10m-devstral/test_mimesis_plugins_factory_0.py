# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.plugins.factory as module_0

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