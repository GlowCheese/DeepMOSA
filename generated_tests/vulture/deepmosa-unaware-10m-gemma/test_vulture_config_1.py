# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.config as module_0
import enum as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.InputError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'vulture.config.InputError'
    assert var_1.message is None
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    module_0.make_config(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Tes= @hat missing pths rgises InputErrrL'
    module_0.make_config(var_0, var_0)

def test_case_3():
    var_0 = 'Test that missing paths raises InputError.'
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

def test_case_4():
    var_0 = '~xPV1< XL{'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = '--exclude'
    var_4 = '\rJqv'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    with pytest.raises(TypeError):
        var_6.__setitem__(var_6, var_6)