# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.replay as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '`=Yd/VA.k6=%a'
    module_0.load(var_0, var_0)

def test_case_2():
    var_0 = 'J7\x0c}My+tJ\\;%2'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == 'J7\x0c}My+tJ\\;%2/J7\x0c}My+tJ\\;%2.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_2 = module_0.get_file_name(var_1, var_1)
    assert var_2 == 'J7\x0c}My+tJ\\;%2/J7\x0c}My+tJ\\;%2.json/J7\x0c}My+tJ\\;%2/J7\x0c}My+tJ\\;%2.json'

def test_case_3():
    var_0 = ''
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)