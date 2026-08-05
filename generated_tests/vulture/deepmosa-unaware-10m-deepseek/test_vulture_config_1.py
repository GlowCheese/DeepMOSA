# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.config as module_0
import enum as module_1
import re as module_2

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
    var_0 = module_2.RegexFlag.LOCALE
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    module_0.make_config(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'path1'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = [var_0, var_2, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_5 = '--verbose'
    var_6 = [var_5]
    module_0.make_config(var_6)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '--exclude'
    var_1 = '*.pyc,venv'
    var_2 = '--ignore-decorators'
    var_3 = '@app.route,@login_required'
    var_4 = '--ignore-names'
    var_5 = 'helper_*'
    var_6 = '--make-whitelist'
    var_7 = '--sort-by-size'
    var_8 = '--verbose'
    var_9 = 'src/'
    var_10 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = module_0.make_config(var_10)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_12 = '--config'
    var_13 = 'path.py'
    var_14 = [var_12, var_13, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--min-confidence'
    var_17 = '10'
    var_18 = [var_16, var_17]
    module_0.make_config(var_18)