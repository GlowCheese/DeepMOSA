# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.plugins.factory as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.FactoryField(var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {'': None}
    assert var_3.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_4 = module_0.FactoryField(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_4.locale is None
    assert var_4.kwargs == {}
    assert var_4.field is None
    var_5 = module_0.FactoryField(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_5.locale is None
    assert var_5.kwargs == {}
    assert var_5.field is None
    var_4.evaluate(var_0, var_0)

def test_case_1():
    var_0 = 'V;DCEzF$Af4+'
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field == 'V;DCEzF$Af4+'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_2 = 'H*!A$2~1^MwTRo\\8K'
    var_3 = module_0.FactoryField(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {}
    assert var_3.field == 'H*!A$2~1^MwTRo\\8K'
    var_3.evaluate(var_0, var_0, var_1)