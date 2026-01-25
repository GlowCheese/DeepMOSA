# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.dump(var_0, var_0, var_0)

def test_case_1():
    var_0 = 'Test load function raisds FileNotFoundError when file does not exist.'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == 'Test load function raisds FileNotFoundError when file does not exist./Test load function raisds FileNotFoundError when file does not exist..json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    var_2 = module_0.get_file_name(var_1, var_1)
    assert var_2 == 'None/None.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    module_0.load(var_2, var_2)

def test_case_3():
    var_0 = ''
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)