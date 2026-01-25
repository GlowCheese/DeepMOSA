# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.errors as module_0


def test_case_0():
    var_0 = module_0.ProgrammingError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.commons.errors.ProgrammingError'
    assert f'{type(module_0.ProgrammingError.passert).__module__}.{type(module_0.ProgrammingError.passert).__qualname__}' == 'builtins.method'