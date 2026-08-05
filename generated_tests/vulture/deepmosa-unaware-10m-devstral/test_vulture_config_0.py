# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.config as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.InputError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'vulture.config.InputError'
    assert var_1.message is None
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ()
    module_0.make_config(var_0)

def test_case_2():
    var_0 = 'W'
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'i\n|Yo|Tfr]_,'
    module_0.make_config(var_0, var_0)

def test_case_4():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--make-whitelist'
    var_7 = '--sort-by-size'
    var_8 = '--verbose'
    var_9 = '--exclude'
    var_10 = '--ignore-decorators'
    var_11 = 'deco1,deco2'
    var_12 = '--ignore-names'
    var_13 = 'name1,name2'
    var_14 = [var_0, var_1, var_4, var_5, var_6, var_7, var_8, var_9, var_5, var_10, var_11, var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = module_0.make_config(var_3)