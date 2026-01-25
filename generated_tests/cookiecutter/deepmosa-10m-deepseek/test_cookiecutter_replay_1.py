# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.replay as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.dump(var_0, var_0, var_0)

def test_case_1():
    var_0 = '/path'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == '/path.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_2():
    var_0 = 'my.template.json'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == 'my.template.json/my.template.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_3():
    var_0 = ''
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '/tmp/test_replay2'
    module_0.load(var_0, var_0)
    var_2 = module_0.dump(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/tmp/test_replay2'
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '/tmp/test_replay2'
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '/tmp/test_replay2'
    var_1 = var_0.__iter__()
    var_2 = 'cookiecutter'
    var_3 = {var_2: var_1}
    module_0.dump(var_0, var_0, var_3)