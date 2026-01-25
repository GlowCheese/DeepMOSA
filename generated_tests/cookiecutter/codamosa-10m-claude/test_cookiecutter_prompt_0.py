# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.prompt as module_0
import pytest
import rich.prompt as module_1


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_0.prompt_and_delete(var_0, var_0)

def test_case_1():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_2():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0, var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.JsonPrompt(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert var_1.prompt is None
    assert var_1.password is False
    assert var_1.case_sensitive is True
    assert var_1.show_default is True
    assert var_1.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_2 = '\n\t\rsFJ'
    var_1.process_response(var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = {}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

def test_case_6():
    var_0 = '7'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '<^[Y$Q[o0>F2>K'
    module_0.render_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = {var_0: var_0}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'BYWMzPs"Zih"AE\x0c\x0ca@'
    var_1 = True
    var_2 = None
    var_3 = module_0.JsonPrompt(var_2, case_sensitive=var_2, show_choices=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert var_3.prompt is None
    assert var_3.password is False
    assert var_3.case_sensitive is None
    assert var_3.show_default is True
    assert var_3.show_choices == 'BYWMzPs"Zih"AE\x0c\x0ca@'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_4 = module_0.render_variable(var_1, var_1, var_1)
    assert var_4 is True
    module_0.prompt_and_delete(var_0, var_1)