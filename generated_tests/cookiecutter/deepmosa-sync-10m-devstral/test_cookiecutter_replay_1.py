# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.dump(var_0, var_0, var_0)

def test_case_1():
    var_0 = '.S6]"7q$'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == '.S6]"7q$/.S6]"7q$.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_2():
    var_0 = 'template.json'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == 'template.json/template.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_3():
    var_0 = '/tmp/test'
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

def test_case_4():
    var_0 = '/tmp/test_replay'
    var_1 = 'cookiecutter'
    var_2 = {var_1: var_1}
    var_3 = module_0.dump(var_0, var_0, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_5():
    var_0 = '/tmp/test_replay'
    var_1 = module_0.load(var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False