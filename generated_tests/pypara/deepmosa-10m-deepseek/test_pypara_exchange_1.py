# Check out: https://github.com/GlowCheese/deepmosa
import pypara.exchange as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.FXRate()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.FXRateLookupError(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.exchange.FXRateLookupError'
    assert var_1.ccy1 is None
    assert var_1.ccy2 is None
    assert var_1.asof is None
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    var_2 = [var_0, var_0, var_0]
    module_0.FXRate(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = [var_0, var_1, var_0, var_0]
    var_3 = module_0.FXRate(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.exchange.FXRate'
    assert len(var_3) == 4
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.FXRate.ccy1).__module__}.{type(module_0.FXRate.ccy1).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.FXRate.ccy2).__module__}.{type(module_0.FXRate.ccy2).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.FXRate.date).__module__}.{type(module_0.FXRate.date).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.FXRate.value).__module__}.{type(module_0.FXRate.value).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.FXRate.of).__module__}.{type(module_0.FXRate.of).__qualname__}' == 'builtins.method'
    var_3.__invert__()