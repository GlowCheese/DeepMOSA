# Check out: https://github.com/GlowCheese/deepmosa
import factory.declarations as module_1
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
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {}
    assert var_3.field is None
    var_4 = 'QCveMz6cR\no'
    var_5 = module_0.FactoryField(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_5.locale is None
    assert var_5.kwargs == {}
    assert var_5.field == 'QCveMz6cR\no'
    var_6 = module_1.Iterator(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'factory.declarations.Iterator'
    assert var_6.getter is None
    assert var_6.iterator is None
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'factory.generate'
    assert module_1.logger.level == 0
    assert f'{type(module_1.logger.parent).__module__}.{type(module_1.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.propagate is True
    assert module_1.logger.handlers == []
    assert module_1.logger.disabled is False
    assert f'{type(module_1.logger.manager).__module__}.{type(module_1.logger.manager).__qualname__}' == 'logging.Manager'
    assert f'{type(module_1.SKIP).__module__}.{type(module_1.SKIP).__qualname__}' == 'factory.declarations.Skip'
    var_5.evaluate(var_0, var_0, var_6)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = '\n`\rm>V\n6cm?9'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.FactoryField(var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.plugins.factory.FactoryField'
    assert var_3.locale is None
    assert var_3.kwargs == {'\n`\rm>V\n6cm?9': None}
    assert var_3.field is None
    assert f'{type(module_0.FactoryField.override_locale).__module__}.{type(module_0.FactoryField.override_locale).__qualname__}' == 'builtins.method'
    var_3.evaluate(var_0, var_0, var_0)

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