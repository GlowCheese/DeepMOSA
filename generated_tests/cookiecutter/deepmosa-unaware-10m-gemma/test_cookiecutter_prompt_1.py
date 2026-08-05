# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import enum as module_2
import jinja2.exceptions as module_3
import codecs as module_4
import jinja2.environment as module_5
import cookiecutter.exceptions as module_6

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
    var_0 = None
    module_0.read_user_variable(var_0, var_0, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.read_user_yes_no(var_0, var_0)

def test_case_3():
    var_0 = None
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_0, var_0)

def test_case_4():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_5():
    var_0 = '`ezFp$t/b(2[\tC'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_7():
    var_0 = {}
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    module_0.prompt_choice_for_template(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 't3\tV(*'
    module_0.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'fp?rO7T9&'
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = {}
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = '}{D=E_VoJ,<nl>'
    module_0.prompt_choice_for_config(var_1, var_0, var_1, var_2, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = [var_0, var_0]
    module_0.read_user_choice(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.YesNoPrompt(var_0, choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
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
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_2 = module_0.render_variable(var_0, var_0, var_0)
    module_0.render_variable(var_2, var_1, var_0)

def test_case_14():
    var_0 = {}
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_1, var_0, var_1, var_1, var_1, var_1)

def test_case_15():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = '`ezFp$t/b(2[\tC'
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_0, var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.render_variable(var_1, var_0, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = [var_2, var_0, var_2, var_2, var_1]
    module_0.read_user_variable(var_2, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'C%\t8-dv=\x0b7I'
    module_0.prompt_and_delete(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '|/ZjOl'
    var_1 = False
    var_2 = {var_0: var_0, var_1: var_1, var_0: var_0}
    var_3 = None
    module_0.prompt_choice_for_template(var_0, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = '__render_me__'
    var_4 = ',^1(0'
    var_5 = 'hidden'
    var_6 = {var_1: var_2, var_2: var_4, var_3: var_5, var_3: var_3}
    var_7 = False
    var_8 = module_0.JsonPrompt(var_2, password=var_7, show_choices=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_8.console).__module__}.{type(var_8.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_8.prompt).__module__}.{type(var_8.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_8.prompt) == 7
    assert var_8.password is False
    assert var_8.case_sensitive is True
    assert var_8.show_default is True
    assert var_8.show_choices is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_9 = {var_0: var_6}
    var_10 = module_0.prompt_for_config(var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.OrderedDict'
    assert len(var_10) == 3
    var_11 = None
    var_8.process_response(var_11)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = '!"L\\o*\x0bc~STXVhUH/9'
    var_2 = 'i\x0cbh4-B&eD01'
    module_0.read_user_variable(var_2, var_0, var_1)

def test_case_21():
    var_0 = True
    var_1 = module_0.YesNoPrompt(show_choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 0
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
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_2 = ''
    with pytest.raises(module_1.InvalidResponse):
        var_1.process_response(var_2)

def test_case_22():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = '`ezFp$t/b(2[\tC'
    var_2 = '|?K#a'
    var_3 = None
    var_4 = {var_2: var_0}
    var_5 = module_0.prompt_choice_for_template(var_3, var_4, var_1)
    assert var_5 == '|?K#a'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    module_0.read_user_dict(var_0, var_0, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = 'Yo'
    var_3 = {var_2: var_0}
    module_0.prompt_choice_for_template(var_2, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'w\x0bO}W2'
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.read_user_choice(var_0, var_1, var_0)

def test_case_26():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'BezFp$t/b(r[\tC'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = {}
    var_1 = None
    var_2 = True
    var_3 = module_0.YesNoPrompt(console=var_1, case_sensitive=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_3.prompt) == 0
    assert var_3.password is False
    assert var_3.case_sensitive is True
    assert var_3.show_default is True
    assert var_3.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_4 = True
    var_5 = [var_0, var_4, var_1, var_1]
    var_6 = module_0.render_variable(var_1, var_1, var_1)
    module_0.read_user_variable(var_4, var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = {}
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = 'C%\t8-d=\x0b7I'
    var_3 = '`ezFp$t/b(2[\tC'
    var_4 = 'P&$y5[2o'
    module_0.read_user_dict(var_3, var_1, var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = {}
    var_1 = None
    var_2 = True
    var_3 = module_0.YesNoPrompt(console=var_1, case_sensitive=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_3.prompt) == 0
    assert var_3.password is False
    assert var_3.case_sensitive is True
    assert var_3.show_default is True
    assert var_3.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_4 = False
    var_5 = [var_0, var_4, var_1, var_1]
    var_6 = module_0.render_variable(var_1, var_1, var_1)
    module_0.read_user_variable(var_4, var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.TemplateRuntimeError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.exceptions.TemplateRuntimeError'
    var_2 = True
    var_3 = module_0.render_variable(var_1, var_2, var_1)
    assert var_3 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.choose_nested_template(var_1, var_1)

def test_case_31():
    var_0 = {}
    var_1 = [var_0, var_0]
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = {var_2: var_0}
    module_0.prompt_choice_for_template(var_2, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = '.'
    module_0.prompt_and_delete(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'cookiecutter'
    module_0.read_user_yes_no(var_0, var_0, var_0)

def test_case_35():
    var_0 = '3'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'cookiecutter'
    var_1 = '^u8yhq\x0b#!\x0cq!"n^'
    var_2 = module_4.make_identity_dict(var_1)
    assert module_4.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_4.BOM_LE == b'\xff\xfe'
    assert module_4.BOM_UTF16_LE == b'\xff\xfe'
    assert module_4.BOM_BE == b'\xfe\xff'
    assert module_4.BOM_UTF16_BE == b'\xfe\xff'
    assert module_4.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_4.BOM == b'\xff\xfe'
    assert module_4.BOM_UTF16 == b'\xff\xfe'
    assert module_4.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_4.BOM32_LE == b'\xff\xfe'
    assert module_4.BOM32_BE == b'\xfe\xff'
    assert module_4.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM64_BE == b'\x00\x00\xfe\xff'
    module_0.read_user_yes_no(var_1, var_0, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = var_0.__repr__()
    assert var_2 == '{}'
    var_3 = var_1.__repr__()
    assert var_3 == '{}'
    var_4 = module_0.process_json(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.OrderedDict'
    assert len(var_4) == 0
    var_4.seek(var_1)

def test_case_38():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = module_0.prompt_for_config(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.OrderedDict'
    assert len(var_3) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_39():
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
    var_1 = 'Optio A'
    var_2 = [var_1, var_1, var_1]
    var_3 = 'my_choice'
    var_4 = {var_3: var_1}
    var_5 = True
    var_6 = module_0.prompt_choice_for_config(var_3, var_0, var_3, var_2, var_5)
    assert var_6 == 'Option A'
    assert var_6 == 'Optio A'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_7 = False
    module_0.prompt_choice_for_config(var_6, var_0, var_3, var_2, var_7, var_4, var_6)
    assert var_8 == 'My Project'

def test_case_40():
    var_0 = 'cookiecutter'
    var_1 = '__render_me__'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1}
    var_3 = {var_0: var_2}
    var_4 = module_0.prompt_for_config(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.OrderedDict'
    assert len(var_4) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_41():
    var_0 = 'cookiecutter'
    var_1 = 'iF$`7&9ep\\9\\'
    var_2 = 'version'
    var_3 = '_Lrnder_me_t_'
    var_4 = ',^1(0'
    var_5 = 'hidden'
    var_6 = {var_1: var_2, var_2: var_4, var_3: var_5, var_3: var_3}
    var_7 = {var_0: var_6}
    var_8 = module_0.prompt_for_config(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.OrderedDict'
    assert len(var_8) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_9 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_9)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'cookiecutter'
    var_1 = 'projepm4t_name'
    var_2 = module_4.make_identity_dict(var_1)
    assert module_4.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_4.BOM_LE == b'\xff\xfe'
    assert module_4.BOM_UTF16_LE == b'\xff\xfe'
    assert module_4.BOM_BE == b'\xfe\xff'
    assert module_4.BOM_UTF16_BE == b'\xfe\xff'
    assert module_4.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_4.BOM == b'\xff\xfe'
    assert module_4.BOM_UTF16 == b'\xff\xfe'
    assert module_4.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_4.BOM32_LE == b'\xff\xfe'
    assert module_4.BOM32_BE == b'\xfe\xff'
    assert module_4.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = {var_0: var_2}
    var_4 = False
    module_0.prompt_for_config(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'cookiecutter'
    var_1 = module_4.make_identity_dict(var_0)
    assert module_4.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_4.BOM_LE == b'\xff\xfe'
    assert module_4.BOM_UTF16_LE == b'\xff\xfe'
    assert module_4.BOM_BE == b'\xfe\xff'
    assert module_4.BOM_UTF16_BE == b'\xfe\xff'
    assert module_4.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_4.BOM == b'\xff\xfe'
    assert module_4.BOM_UTF16 == b'\xff\xfe'
    assert module_4.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_4.BOM32_LE == b'\xff\xfe'
    assert module_4.BOM32_BE == b'\xfe\xff'
    assert module_4.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {var_0: var_2}
    module_0.prompt_for_config(var_4, var_3)

def test_case_44():
    var_0 = 'cookiecutter'
    var_1 = ',^1(0'
    var_2 = {var_1: var_0, var_0: var_1, var_0: var_0, var_0: var_0}
    var_3 = {var_0: var_2}
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_3, var_0, var_0)

def test_case_45():
    var_0 = 'cookiecutter'
    var_1 = '__render_me_'
    var_2 = 'use_git'
    var_3 = 'InteractiveProject'
    var_4 = True
    var_5 = [var_3, var_1]
    var_6 = {var_1: var_3, var_2: var_4, var_0: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.prompt_for_config(var_7, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.OrderedDict'
    assert len(var_8) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_46():
    var_0 = 'cookiecutter'
    var_1 = '}7C'
    var_2 = '_private_var'
    var_3 = 'q\t:G~"3BmR]]"&f9rY'
    var_4 = 'hidden'
    var_5 = 'Hello {{ cookiecutter.project_name }}'
    var_6 = {var_1: var_2, var_1: var_5, var_2: var_4, var_3: var_5}
    var_7 = {var_3: var_6, var_2: var_6, var_0: var_6}
    var_8 = True
    with pytest.raises(module_6.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_7, var_8)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = 'cookiecutter'
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    module_0.prompt_for_config(var_3, var_1)

def test_case_48():
    var_0 = 'cookiecutter'
    var_1 = module_4.make_identity_dict(var_0)
    assert module_4.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_4.BOM_LE == b'\xff\xfe'
    assert module_4.BOM_UTF16_LE == b'\xff\xfe'
    assert module_4.BOM_BE == b'\xfe\xff'
    assert module_4.BOM_UTF16_BE == b'\xfe\xff'
    assert module_4.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_4.BOM == b'\xff\xfe'
    assert module_4.BOM_UTF16 == b'\xff\xfe'
    assert module_4.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_4.BOM32_LE == b'\xff\xfe'
    assert module_4.BOM32_BE == b'\xfe\xff'
    assert module_4.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_4.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_2}
    var_5 = module_0.prompt_for_config(var_4, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_6 = str(var_2)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = '__render_me_'
    var_2 = 'use_git'
    var_3 = 'InteractiveProject'
    var_4 = False
    var_5 = [var_3, var_1]
    var_6 = {var_1: var_3, var_2: var_4, var_0: var_5}
    var_7 = {var_0: var_6}
    module_0.prompt_for_config(var_7, var_4)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_1}
    module_0.read_user_dict(var_0, var_2, var_2)

def test_case_51():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'path'
    var_3 = {var_0: var_0, var_2: var_0}
    var_4 = {var_1: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.choose_nested_template(var_6, var_0, var_7)
    assert var_8 == '/workspace/run/cookiecutter/cookiecutter'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_52():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 't1'
    var_3 = 'path'
    var_4 = '/absolute/path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_8, var_9, var_10)

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'path'
    var_3 = {var_0: var_0, var_2: var_0}
    var_4 = {var_1: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    module_0.choose_nested_template(var_6, var_0, var_7)

def test_case_54():
    var_0 = 'Question?'
    var_1 = module_0.YesNoPrompt(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 9
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
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_2 = '  YES  '
    var_3 = var_1.process_response(var_2)
    assert var_3 is True
    var_4 = 'No\n'
    var_5 = var_1.process_response(var_4)
    assert var_5 is False
    var_6 = 'maybe'
    with pytest.raises(module_1.InvalidResponse):
        var_1.process_response(var_6)

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = 'cookiecutter'
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    module_0.read_user_choice(var_0, var_2, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = 'cookiecutter'
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = {var_0: var_1}
    module_0.read_user_dict(var_0, var_2, var_2)

def test_case_57():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = ''
    var_3 = None
    var_4 = module_0.JsonPrompt(var_3, console=var_3, choices=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_4.console).__module__}.{type(var_4.console).__qualname__}' == 'rich.console.Console'
    assert var_4.prompt is None
    assert var_4.password is False
    assert var_4.case_sensitive is True
    assert var_4.show_default is True
    assert var_4.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_5 = 'path'
    var_6 = {var_5: var_2}
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = '/tmp/repo'
    var_11 = True
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_9, var_10, var_11)