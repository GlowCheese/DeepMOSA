# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import re as module_1
import rich.prompt as module_2
import enum as module_3
import cookiecutter.exceptions as module_4

def test_case_0():
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

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    module_0.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.read_user_yes_no(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '@\x0buxkYH\x0b"{N1~#9FU,'
    module_0.read_user_choice(var_0, var_0)

def test_case_4():
    var_0 = 'o`A(Jt\x0cQ5E'
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0, var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_0.prompt_and_delete(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ')"GMRz//bA{5[P'
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = False
    var_2 = module_0.JsonPrompt(console=var_0, show_choices=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_2.console).__module__}.{type(var_2.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_2.prompt).__module__}.{type(var_2.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_2.prompt) == 0
    assert var_2.password is False
    assert var_2.case_sensitive is True
    assert var_2.show_default is True
    assert var_2.show_choices is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_2.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '{}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 0
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_choice_for_template(var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '>5>~2\nialR'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = None
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'confirm'
    var_1 = 'Do you want to proceed?'
    var_2 = {var_0: var_1}
    var_3 = False
    module_0.read_user_yes_no(var_0, var_3, var_2)
    assert var_4 is False

def test_case_14():
    var_0 = False
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '\x0b_@l%P:af]9\t'
    var_1 = None
    module_0.prompt_choice_for_config(var_1, var_1, var_1, var_0, var_1, prefix=var_1)

def test_case_16():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_1.RegexFlag.VERBOSE
    module_0.render_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_0.render_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'key'
    var_1 = {var_0: var_0}
    var_2 = 'PROMPT: '
    module_0.read_user_dict(var_0, var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_1}
    var_3 = {var_0: var_2}
    module_0.prompt_for_config(var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = ')\r\\'
    module_0.read_user_variable(var_0, var_0, var_0)

def test_case_22():
    var_0 = {}
    var_1 = None
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_1, var_1, var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = '{}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 0
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_choice_for_template(var_1, var_1, var_0)

def test_case_24():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = True
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_2, var_1, var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = True
    var_1 = module_3._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = None
    var_3 = [var_1, var_0, var_1]
    var_4 = module_0.render_variable(var_2, var_3, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_4)

def test_case_26():
    var_0 = '{}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 0
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_27():
    var_0 = '"just a string"'
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = '6#wh'
    module_0.read_user_choice(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'c5t%KZuX\x0bX'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = None
    module_0.prompt_choice_for_template(var_0, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'B`'
    var_1 = 'H'
    var_2 = [var_1]
    module_0.read_user_choice(var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 'bg:NQ='
    var_1 = 'b:1NQ='
    var_2 = 'b:NQc'
    module_0.read_user_variable(var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'c5t%KZuX\x0bX'
    var_1 = {var_0: var_0}
    var_2 = None
    var_3 = 'wzxY\t,zybP\x0c>yz\rQd|X'
    var_4 = {var_2: var_1, var_0: var_3}
    module_0.prompt_choice_for_template(var_3, var_4, var_2)

def test_case_33():
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
    var_1 = 'u5+ho}'
    with pytest.raises(module_2.InvalidResponse):
        var_0.process_response(var_1)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'other_key'
    var_1 = 'Ignore me'
    var_2 = {var_0: var_1}
    var_3 = 'target_key'
    var_4 = {}
    module_0.read_user_dict(var_3, var_4, var_2)

def test_case_35():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice (relative/path)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)
    assert var_8 == '/tmp/repo/relative/path'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'user_input'

def test_case_37():
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
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = var_0.process_response(var_1)
    assert var_3 is True
    var_4 = 'y'
    var_5 = var_0.process_response(var_4)
    assert var_5 is True
    var_6 = ''
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_6)

def test_case_38():
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
    var_1 = '0'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = var_0.process_response(var_1)
    assert var_3 is False
    var_4 = 'f'
    var_5 = var_0.process_response(var_4)
    assert var_5 is False
    var_6 = 'no'
    var_7 = var_0.process_response(var_6)
    assert var_7 is False
    var_8 = 'n'
    var_9 = var_0.process_response(var_8)
    assert var_9 is False
    var_10 = 'off'
    var_11 = var_0.process_response(var_10)
    assert var_11 is False
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_8)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 'Delete file?'
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = '?'
    module_0.read_user_yes_no(var_3, var_2, var_1, var_3)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = ''
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = '?'
    module_0.read_user_yes_no(var_0, var_2, var_1, var_3)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = '0Z-CUW1~9\x0b#&cv8R'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = [var_0, var_0, var_1, var_0]
    module_0.read_user_choice(var_0, var_2, var_1, var_1)

def test_case_42():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_private_var'
    var_3 = '__rendered_var__'
    var_4 = 'my_project'
    var_5 = 'hidden'
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.OrderedDict'
    assert len(var_10) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_43():
    var_0 = 'cookiecutter'
    var_1 = 'use_git'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.prompt_for_config(var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 'cookiecutter'
    var_1 = '__promptsf_'
    var_2 = 'teplate'
    var_3 = {var_2: var_2}
    var_4 = {var_2: var_2, var_1: var_3}
    var_5 = {var_0: var_4}
    module_0.prompt_for_config(var_5)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 'cookiecutter'
    var_1 = 'us< e_git'
    var_2 = False
    var_3 = {var_1: var_2, var_1: var_2, var_0: var_2, var_0: var_2, var_1: var_2}
    var_4 = {var_0: var_3}
    module_0.prompt_for_config(var_4, var_2)

def test_case_46():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = 'web'
    var_3 = 'api'
    var_4 = 'cli'
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

def test_case_47():
    var_0 = 'cookiecutter'
    var_1 = 'settings'
    var_2 = 'debug'
    var_3 = '{{ cookiecutter.debug_mode }}'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = {var_1: var_4, var_0: var_5}
    var_7 = {var_0: var_6}
    var_8 = True
    with pytest.raises(module_4.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_7, var_8)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.prompt_for_config(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_0, var_4)

def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__prompts__'
    var_3 = {var_1: var_1}
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.prompt_for_config(var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.OrderedDict'
    assert len(var_6) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_7 = module_0.JsonPrompt(password=var_2, choices=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_7.console).__module__}.{type(var_7.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_7.prompt).__module__}.{type(var_7.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_7.prompt) == 0
    assert var_7.password == '__prompts__'
    assert var_7.choices == {'cookiecutter': {'project_name': 'cookiecutter'}}
    assert var_7.case_sensitive is True
    assert var_7.show_default is True
    assert var_7.show_choices is True
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__prompts__'
    var_3 = {var_1: var_0}
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = False
    module_0.prompt_for_config(var_5, var_6)

def test_case_51():
    var_0 = 'cookiecutter'
    var_1 = 'debug'
    var_2 = '{{ cookiecutter.debug_mode }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    with pytest.raises(module_4.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = ''
    var_1 = {var_0: var_0}
    var_2 = 'PROMPT. '
    module_0.read_user_dict(var_0, var_1, var_1, var_2)

def test_case_53():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_1}
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_2, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = 'cookiecutter'
    var_1 = '__prompts__'
    var_2 = ''
    var_3 = {var_2: var_2}
    var_4 = {var_2: var_2, var_1: var_3}
    var_5 = {var_0: var_4}
    module_0.prompt_for_config(var_5)

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = 'color'
    var_1 = '__prompt__'
    var_2 = 'other'
    var_3 = 'Pick a hue:'
    var_4 = 'something'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'red'
    var_8 = 'blue'
    var_9 = [var_7, var_8]
    var_10 = 'color'
    module_0.read_user_choice(var_10, var_9, var_6)
    assert var_11 == 'blue'

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = 'color'
    var_1 = '_rompt__'
    var_2 = '1'
    var_3 = '2'
    var_4 = 'Pick a hue:'
    var_5 = 'Crimson'
    var_6 = 'Azure'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'red'
    var_10 = 'blue'
    var_11 = [var_9, var_10]
    var_12 = 'color'
    module_0.read_user_choice(var_12, var_11, var_8)
    assert var_13 == 'red'

def test_case_57():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_8, var_9, var_10)

@pytest.mark.xfail(strict=True)
def test_case_58():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 'template1'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = '/tmp/repo'
    module_0.choose_nested_template(var_10, var_11)