# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import re as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_0.prompt_and_delete(var_0, var_0)

def test_case_1():
    var_0 = None
    var_1 = []
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_0, var_1, var_0)

def test_case_2():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_3():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.JsonPrompt(case_sensitive=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 0
    assert var_1.password is False
    assert var_1.case_sensitive is None
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
    var_2 = module_2.purge()
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_3 = 'Wru/Z>'
    var_1.process_response(var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = {}
    var_2 = False
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = {}
    var_2 = True
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_2.purge()
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_1 = module_1.InvalidResponse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'rich.prompt.InvalidResponse'
    assert var_1.message is None
    assert f'{type(module_1.PromptType).__module__}.{type(module_1.PromptType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.DefaultType).__module__}.{type(module_1.DefaultType).__qualname__}' == 'typing.TypeVar'
    module_0.render_variable(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'b\nSL;nT=Sb5$H'
    var_1 = None
    module_0.render_variable(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = {var_0: var_0}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_2.RegexFlag.IGNORECASE
    module_0.read_user_choice(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'b\nSL;nT=Sb5$H'
    var_1 = None
    module_0.prompt_choice_for_config(var_1, var_1, var_1, var_0, var_1, prefix=var_1)