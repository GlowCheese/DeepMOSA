# Check out: https://github.com/GlowCheese/deepmosa
import pypara.accounting.generic as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.Balance(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_1.date is None
    assert var_1.value is None