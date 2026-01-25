# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.prompt as module_0
import pytest
import rich.prompt as module_1


def test_case_0():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_1():
    var_0 = None
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_0, var_0, var_0, var_0)

def test_case_2():
    var_0 = 'j!oc_w\\;v0S~d^R'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_3():
    var_0 = '>>5b'
    var_1 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '2\r>q#,*-u\r(=\x0c\n'
    var_2 = 'tSlhVsR'
    var_3 = {var_0: var_1, var_1: var_2, var_2: var_1, var_2: var_0}
    var_4 = True
    module_0.prompt_choice_for_template(var_1, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = 'k58$EF/*O\x0c"p'
    module_0.render_variable(var_1, var_2, var_1)