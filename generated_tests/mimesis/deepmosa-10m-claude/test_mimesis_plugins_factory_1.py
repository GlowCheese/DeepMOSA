# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.plugins.factory as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'iBbtPC(#'
    var_1 = None
    var_2 = '-v^/Y7K"O'
    var_3 = {var_0: var_1, var_0: var_1, var_2: var_1, var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_4.locale is None
    assert var_4.kwargs == {'iBbtPC(#': None, '-v^/Y7K"O': None}
    assert var_4.field == 'iBbtPC(#'
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_4.evaluate(var_1, var_1, var_1)

def test_case_1():
    pass

def test_case_2():
    var_0 = None
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "|X5J'E"
    var_1 = module_0.FactoryField(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field == "|X5J'E"
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_1.evaluate(var_1, var_1, var_1)