# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.config as module_0

def test_case_0():
    pass

def test_case_1():
    var_0 = None
    var_1 = module_0.InputError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'vulture.config.InputError'
    assert var_1.message is None
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

def test_case_2():
    var_0 = 'atf'
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = []
    module_0.make_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'atf'
    module_0.make_config(var_0, var_0)

def test_case_5():
    var_0 = 'key1'
    var_1 = False
    var_2 = {var_0: var_1}
    with pytest.raises(module_0.InputError):
        module_0._check_input_config(var_2)

def test_case_6():
    var_0 = 'path'
    var_1 = '--exclude'
    var_2 = '*.pyc'
    var_3 = '--ignore-decorators'
    var_4 = 'deco1,deco2'
    var_5 = '--ignore-names'
    var_6 = 'name1,name2'
    var_7 = '--make-whitelist'
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = '--sort-by-size'
    var_11 = '--verbose'
    var_12 = '--config'
    var_13 = 'myconfig.toml'
    var_14 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13]
    var_15 = module_0.make_config(var_14)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}