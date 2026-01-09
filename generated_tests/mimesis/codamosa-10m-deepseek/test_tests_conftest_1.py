# Check out: https://github.com/GlowCheese/deepmosa
import tests.conftest as module_0


def test_case_0():
    var_0 = module_0.get_locales()
    assert module_0.platform == ['win32', 'linux', 'darwin']