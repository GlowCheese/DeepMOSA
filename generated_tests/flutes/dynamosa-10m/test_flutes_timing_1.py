# Check out: https://github.com/GlowCheese/deepmosa
import flutes.timing as module_0


def test_case_0():
    var_0 = module_0.work_in_progress()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}