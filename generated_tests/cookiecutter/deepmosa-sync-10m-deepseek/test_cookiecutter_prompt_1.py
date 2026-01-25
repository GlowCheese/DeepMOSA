# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_2

import cookiecutter.prompt as module_0
import pytest
import rich.prompt as module_1


def test_case_0():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_1():
    var_0 = None
    var_1 = None
    var_2 = 'ET@@mz5.e\x0c{Iy8|!W'
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_1, var_0, prefix=var_2)

def test_case_2():
    var_0 = None
    var_1 = '$Om\r^a>q\\m{>e'
    var_2 = module_0.JsonPrompt(choices=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_2.console).__module__}.{type(var_2.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_2.prompt).__module__}.{type(var_2.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_2.prompt) == 0
    assert var_2.password is False
    assert var_2.case_sensitive is True
    assert var_2.show_default is True
    assert var_2.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '=9pp$L0'
    var_1 = None
    module_0.render_variable(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = 'jk|sTkEYzIa29|'
    var_2 = {}
    module_0.prompt_choice_for_template(var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = '=F:Z(<J,|?F4@{a k)\r_'
    module_0.prompt_choice_for_config(var_0, var_0, var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'jk|sTkEYzIa29|'
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = '}O-r%\rcV~\x0b@-]'
    var_2 = {var_1: var_1, var_0: var_1}
    module_0.prompt_choice_for_template(var_1, var_2, var_0)