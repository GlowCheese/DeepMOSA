# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import re as module_2
import cookiecutter.exceptions as module_3
import pathlib as module_4
import jinja2.environment as module_5

def test_case_0():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.read_user_variable(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'cookiecutter'
    var_1 = 'choice'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    module_0.prompt_for_config(var_4)

def test_case_4():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '{"key": "value"'
    var_1 = None
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
    var_3 = {var_1: var_0, var_0: var_2}
    module_0.prompt_choice_for_template(var_1, var_3, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'X.!\n)\n\x0b>{ov'
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'yes'
    var_1 = 'no'
    var_2 = [var_0, var_1]
    var_3 = 'choice'
    var_4 = 'Do you want to continue?'
    var_5 = {var_3: var_4}
    module_0.read_user_choice(var_3, var_2, var_5)
    assert var_6 == 'yes'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = None
    var_3 = None
    var_4 = module_0.JsonPrompt(choices=var_1, case_sensitive=var_2, show_default=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_4.console).__module__}.{type(var_4.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_4.prompt).__module__}.{type(var_4.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_4.prompt) == 0
    assert var_4.password is False
    assert var_4.choices == []
    assert var_4.case_sensitive is None
    assert var_4.show_default is None
    assert var_4.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_4.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = ';'
    var_1 = 'Klo,xd_\\6GP'
    var_2 = None
    var_3 = [var_1, var_0, var_1, var_2]
    module_0.read_user_choice(var_2, var_3, prefix=var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'o%'
    var_1 = {}
    var_2 = None
    var_3 = module_0.render_variable(var_2, var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_4 = module_0.JsonPrompt(console=var_3, choices=var_2, show_default=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_4.console).__module__}.{type(var_4.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_4.prompt).__module__}.{type(var_4.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_4.prompt) == 0
    assert var_4.password is False
    assert var_4.case_sensitive is True
    assert var_4.show_default is None
    assert var_4.show_choices is True
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_5 = True
    module_0.prompt_choice_for_template(var_0, var_1, var_5)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '/fake/path'
    var_1 = True
    module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'response'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'g%'
    var_1 = {}
    var_2 = None
    var_3 = module_0.render_variable(var_2, var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_4 = module_0.JsonPrompt(console=var_3, choices=var_2, show_default=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_4.console).__module__}.{type(var_4.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_4.prompt).__module__}.{type(var_4.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_4.prompt) == 0
    assert var_4.password is False
    assert var_4.case_sensitive is True
    assert var_4.show_default is None
    assert var_4.show_choices is True
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_5 = False
    module_0.prompt_choice_for_template(var_0, var_1, var_5)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'o%'
    var_1 = {var_0: var_0}
    var_2 = None
    var_3 = module_0.JsonPrompt(console=var_1, choices=var_2, show_default=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert var_3.console == {'o%': 'o%'}
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_3.prompt) == 0
    assert var_3.password is False
    assert var_3.case_sensitive is True
    assert var_3.show_default is None
    assert var_3.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_4 = True
    var_5 = {var_4: var_1, var_2: var_3, var_2: var_1, var_3: var_0}
    module_0.prompt_choice_for_template(var_3, var_5, var_4)

def test_case_16():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.prompt_for_config(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.OrderedDict'
    assert len(var_4) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_3.CookiecutterException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.exceptions.CookiecutterException'
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    var_1 = None
    module_0.render_variable(var_0, var_0, var_1)

def test_case_18():
    var_0 = '/'
    var_1 = None
    var_2 = module_0.YesNoPrompt(console=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
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
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    with pytest.raises(module_1.InvalidResponse):
        var_2.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = False
    module_0.prompt_for_config(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'cookiecutter'
    module_0.read_user_yes_no(var_0, var_0, var_0)

def test_case_21():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_22():
    var_0 = '[1, 2, 3]'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = 'PRE: '
    var_2 = 'uxUR?hM.-gPBY'
    module_0.read_user_variable(var_2, var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 'cookiecutter'
    var_1 = 'choice'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    module_0.prompt_for_config(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = 'web'
    var_3 = 'mobile'
    var_4 = 'deOk\x0cop'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    module_0.prompt_for_config(var_7, var_8)

def test_case_26():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = 'web'
    var_3 = 'mobile'
    var_4 = 'desktop'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.OrderedDict'
    assert len(var_9) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_10 = var_9['type']
    assert var_10 == 'web'

def test_case_27():
    var_0 = 'cookiecutter'
    var_1 = '_private_va[r'
    var_2 = {var_1: var_0, var_0: var_0, var_1: var_0}
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.prompt_for_config(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_28():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'metadata'
    var_3 = 'App'
    var_4 = 'v1ersion'
    var_5 = {var_4: var_2, var_2: var_2}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_2: var_6, var_0: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.OrderedDict'
    assert len(var_9) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(TypeError):
        var_10 = bool(var_3['metadata'] == {'version': '1.0', 'author': 'Admin'})
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'default_val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    module_0.prompt_for_config(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'cookiecutter'
    var_1 = '_private_va[r'
    var_2 = '__internal_var__'
    var_3 = {var_1: var_0, var_0: var_0, var_2: var_0}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.prompt_for_config(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.OrderedDict'
    assert len(var_6) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_7 = None
    module_0.read_user_variable(var_2, var_7)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 'B'
    var_1 = [var_0, var_0]
    var_2 = 'var'
    var_3 = '__prompt__'
    var_4 = '1'
    var_5 = ''
    var_6 = 'Label One'
    var_7 = {var_3: var_6, var_0: var_2, var_4: var_6, var_5: var_5}
    var_8 = {var_2: var_7}
    module_0.read_user_choice(var_2, var_1, var_8)
    assert var_9 == 'A'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = [var_0, var_1]
    var_3 = '__prompt__'
    var_4 = '1'
    var_5 = '2'
    var_6 = 'Label One'
    var_7 = 'Label Two'
    var_8 = {var_3: var_6, var_4: var_6, var_5: var_7}
    var_9 = {var_6: var_8}
    module_0.read_user_choice(var_6, var_2, var_9)
    assert var_10 == 'A'

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = [var_0, var_1]
    var_3 = 'N'
    var_4 = '2'
    var_5 = 'Custom Prompt'
    var_6 = 'Lael ;ne'
    var_7 = 'Label Two'
    var_8 = {var_6: var_5, var_3: var_6, var_4: var_7}
    var_9 = {var_1: var_8}
    module_0.read_user_choice(var_0, var_2, var_9)
    assert var_10 == 'A'

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = [var_0, var_1]
    var_3 = 'var'
    var_4 = 'XxSKkl+x.2o0o@Ac'
    var_5 = '1'
    var_6 = ''
    var_7 = 'Label One'
    var_8 = 'Lbel Two'
    var_9 = {var_4: var_7, var_1: var_0, var_5: var_7, var_6: var_8}
    var_10 = {var_3: var_9}
    module_0.read_user_choice(var_3, var_2, var_10)
    assert var_11 == 'A'

def test_case_35():
    var_0 = 'cookiecutter'
    var_1 = '_private_var'
    var_2 = '__internal_var__'
    var_3 = 'some_value'
    var_4 = 'template_string'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = module_0.prompt_for_config(var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.OrderedDict'
    assert len(var_8) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_9 = None
    var_10 = ' }'
    var_11 = {var_3: var_3, var_3: var_6, var_10: var_0}
    var_12 = module_0.prompt_choice_for_template(var_9, var_6, var_11)
    assert var_12 == 'cookiecutter'

def test_case_36():
    var_0 = 'cookiecutter'
    var_1 = 'my_var'
    var_2 = '{{ undefined_variable }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    with pytest.raises(module_3.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 'other'
    var_1 = 'Something else'
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = False
    var_5 = ''
    module_0.read_user_yes_no(var_3, var_4, var_2, var_5)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 'action'
    var_1 = 'Do you want to proceed?'
    var_2 = {var_0: var_1}
    var_3 = 'action'
    var_4 = True
    var_5 = ''
    module_0.read_user_yes_no(var_3, var_4, var_2, var_5)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = False
    var_5 = ''
    module_0.read_user_yes_no(var_3, var_4, var_2, var_5)

def test_case_40():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = None
    var_3 = module_0.JsonPrompt(choices=var_1, case_sensitive=var_2, show_default=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_3.prompt) == 0
    assert var_3.password is False
    assert var_3.choices == []
    assert var_3.case_sensitive is None
    assert var_3.show_default is None
    assert var_3.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_4 = module_0.YesNoPrompt(console=var_2, choices=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_4.console).__module__}.{type(var_4.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_4.prompt).__module__}.{type(var_4.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_4.prompt) == 0
    assert var_4.password is False
    assert var_4.case_sensitive is True
    assert var_4.show_default is True
    assert var_4.show_choices is True
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_5 = 'my_dict'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_10, var_7)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'key'
    var_1 = 'default'
    var_2 = ''
    var_3 = {var_0: var_2}
    module_0.read_user_variable(var_0, var_1, var_3)
    assert var_4 == 'value'

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '__prompts__'
    var_3 = {var_1: var_1}
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = False
    module_0.prompt_for_config(var_5, var_6)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '__prompts__'
    var_3 = 'key'
    var_4 = {var_3: var_1}
    var_5 = {var_1: var_4, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    module_0.prompt_for_config(var_6, var_7)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = 'subdir/template1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = '/tmp/repo'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_4.Path(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_14 = '/tmp/repo/subdir/template1'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_4.Path(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pathlib.PosixPath'
    module_0.choose_nested_template(var_8, var_9)

def test_case_45():
    var_0 = module_5.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.name }}'
    var_7 = 'static'
    var_8 = [var_6, var_7]
    var_9 = module_0.render_variable(var_0, var_8, var_5)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_10 = bool(var_9 == ['test', 'static'])

def test_case_46():
    var_0 = module_5.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = False
    var_2 = {}
    var_3 = module_0.render_variable(var_0, var_1, var_2)
    assert var_3 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.YesNoPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_0.console).__module__}.{type(var_0.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_0.prompt).__module__}.{type(var_0.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_0.prompt) == 0
    assert var_0.password is False
    assert var_0.case_sensitive is True
    assert var_0.show_default is True
    assert var_0.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_1 = 'yes'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'y'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = '1'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = None
    module_2.split(var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.YesNoPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_0.console).__module__}.{type(var_0.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_0.prompt).__module__}.{type(var_0.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_0.prompt) == 0
    assert var_0.password is False
    assert var_0.case_sensitive is True
    assert var_0.show_default is True
    assert var_0.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_1 = 'no'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'n'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = None
    module_0.prompt_for_config(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = 'm@_dict'
    var_2 = '__prompts__'
    var_3 = ''
    var_4 = {var_1: var_3}
    var_5 = {var_1: var_4, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    module_0.prompt_for_config(var_6, var_7)