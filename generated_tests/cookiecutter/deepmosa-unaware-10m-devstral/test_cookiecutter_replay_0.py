# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ''
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {var_0: var_0}
    var_4 = {var_2: var_3}
    module_0.dump(var_0, var_1, var_4)

def test_case_2():
    var_0 = '/tmp/rekl'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == '/tmp/rekl.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    with pytest.raises(ValueError):
        module_0.dump(var_1, var_1, var_1)

def test_case_3():
    var_0 = '/tmp/rekl'
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    var_1 = 'cookiecutter'
    module_0.dump(var_0, var_1, var_1)