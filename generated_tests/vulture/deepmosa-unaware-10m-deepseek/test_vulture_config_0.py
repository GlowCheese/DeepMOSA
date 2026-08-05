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
    var_0 = 'MDGz'
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'F'
    module_0.make_config(var_0, var_0)

def test_case_4():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_4 = '\n    [tool.vulture]\n    exclude = ["test*.py", "temp/"]\n    min_confidence = 50\n    paths = ["src/", "main.py"]\n    verbose = true\n    '
    var_5 = '--min-confidence'
    var_6 = '--exclude'
    var_7 = 'test*.py,docs'
    var_8 = '--ignore-decorators'
    var_9 = '@app.route,@require_*'
    var_10 = '--ignore-names'
    var_11 = 'visit_*,do_*'
    var_12 = '--make-whitelist'
    var_13 = '70'
    var_14 = '--sort-by-size'
    var_15 = '--verbose'
    var_16 = 'custom.toml'
    var_17 = [var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_5, var_13, var_14, var_15, var_4, var_16, var_0, var_1]
    var_18 = module_0.make_config(var_17)