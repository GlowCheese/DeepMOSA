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
    var_0 = "D\t\n5{X8D)+'DCjH{V,<"
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'OtCd'
    module_0.make_config(var_0, var_0)

def test_case_4():
    var_0 = '--ignore-names'
    var_1 = 'visit_*,do_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_4 = bool(var_3 == {'ignore_names': ['visit_*', 'do_*'], 'config': 'pyproject.toml'})
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = ''
    module_0.make_config(var_0)

def test_case_6():
    var_0 = 'key1'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    with pytest.raises(module_0.InputError):
        module_0._check_input_config(var_2)