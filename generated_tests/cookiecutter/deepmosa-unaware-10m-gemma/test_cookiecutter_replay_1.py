# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.dump(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'my_template'
    module_0.load(var_0, var_0)

def test_case_2():
    var_0 = 'zc2G Z!K3fYwvoZl'
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

def test_case_3():
    var_0 = '/tmp/replay'
    var_1 = 'cookiecutter'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = module_0.dump(var_0, var_2, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_4 = {}
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_2, var_4)

def test_case_4():
    var_0 = 'cookiecutter'
    var_1 = module_0.dump(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '`'
    module_0.load(var_0, var_0)
    var_1 = 2
    var_3 = var_2.__setitem__(var_1, var_2)
    var_4 = module_0.dump(var_0, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'my_template'
    module_0.load(var_0, var_0)