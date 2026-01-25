# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.plugins.factory as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.FactoryField(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_1.evaluate(var_0, var_0, var_0)

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
    var_0 = None
    var_1 = module_0.FactoryField(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_1.locale is None
    assert var_1.kwargs == {}
    assert var_1.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_2 = '"G/wRo< {\x0cZ;p"'
    var_3 = "E-\rRjszppHU0O2/?+'@"
    var_4 = '^\x0b[s%7V[."U.]z'
    var_5 = {var_2: var_0, var_4: var_0, var_4: var_2, var_4: var_0}
    var_6 = module_0.FactoryField(var_0, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_6.locale is None
    assert var_6.kwargs == {'"G/wRo< {\x0cZ;p"': None, '^\x0b[s%7V[."U.]z': None}
    assert var_6.field is None
    var_7 = {var_2: var_0, var_3: var_0, var_3: var_0, var_3: var_0}
    var_8 = module_0.FactoryField(var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_8.locale is None
    assert var_8.kwargs == {}
    assert var_8.field is None
    var_9 = module_0.FactoryField(var_0, **var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_9.locale is None
    assert var_9.kwargs == {'"G/wRo< {\x0cZ;p"': None, "E-\rRjszppHU0O2/?+'@": None}
    assert var_9.field is None
    var_9.evaluate(var_0, var_0, var_9)