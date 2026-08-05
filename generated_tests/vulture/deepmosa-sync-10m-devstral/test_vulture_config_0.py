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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = []
    module_0.make_config(var_0)

def test_case_3():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    verbose = true\n    '
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

def test_case_4():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    with pytest.raises(module_0.InputError):
        module_0._check_input_config(var_2)

def test_case_5():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--exclude'
    var_3 = '*.py'
    var_4 = '--ignore-decorators'
    var_5 = 'decorator1'
    var_6 = '--ignore-names'
    var_7 = 'name1'
    var_8 = '--make-whitelist'
    var_9 = '--min-confidence'
    var_10 = '80'
    var_11 = '--sort-by-size'
    var_12 = '--verbose'
    var_13 = '--config'
    var_14 = 'custom.toml'
    var_15 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14]
    var_16 = module_0.make_config(var_15)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_17 = var_16['paths']
    var_18 = bool(var_16['paths'] == ['path1', 'path2'])
    assert var_18 is True
    var_19 = var_16['exclude']
    var_20 = bool(var_16['exclude'] == ['*.py'])
    assert var_20 is True
    var_21 = var_16['ignore_decorators']
    var_22 = bool(var_16['ignore_decorators'] == ['decorator1'])
    assert var_22 is True
    var_23 = var_16['ignore_names']
    var_24 = bool(var_16['ignore_names'] == ['name1'])
    assert var_24 is True
    var_25 = var_16['make_whitelist']
    assert var_25 is True
    var_26 = var_16['min_confidence']
    assert var_26 == 80
    var_27 = var_16['sort_by_size']
    assert var_27 is True
    var_28 = var_16['verbose']
    assert var_28 is True
    var_29 = var_16['config']
    assert var_29 == 'custom.toml'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\n    [too.vulture]\n    exclude = ["teso*.p8"]\n  1 Min_confidence = 80\n    verbose = true\n)   '
    module_0.make_config(var_0, var_0)