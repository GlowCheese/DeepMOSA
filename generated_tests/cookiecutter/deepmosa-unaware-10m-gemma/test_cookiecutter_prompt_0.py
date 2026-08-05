# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import enum as module_2
import builtins as module_3
import codecs as module_4
import re as module_5
import jinja2.environment as module_6
import cookiecutter.exceptions as module_7

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
    module_0.read_user_variable(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.read_user_yes_no(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "[*N|`'mEt7"
    module_0.read_user_choice(var_0, var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_1, var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_6():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_0.prompt_and_delete(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.JsonPrompt(choices=var_0, show_choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 0
    assert var_1.password is False
    assert var_1.case_sensitive is True
    assert var_1.show_default is True
    assert var_1.show_choices is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_1.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = ':3O5>=W/FI'
    module_0.read_user_choice(var_0, var_0, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'AXhI'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = True
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

def test_case_14():
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

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = '",z!n\r$zOs'
    module_0.prompt_choice_for_config(var_1, var_1, var_0, var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__dir__()
    module_0.render_variable(var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'Gdk '
    module_0.read_user_yes_no(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = module_3.object()
    var_3 = module_0.YesNoPrompt(var_2, choices=var_0, case_sensitive=var_1, show_default=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'builtins.object'
    assert var_3.password is False
    assert var_3.case_sensitive is None
    assert f'{type(var_3.show_default).__module__}.{type(var_3.show_default).__qualname__}' == 'builtins.object'
    assert var_3.show_choices is True
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    module_0.render_variable(var_0, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = ':3O5>=W/FI'
    module_0.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = True
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert var_1 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_1, var_0)

def test_case_21():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_0, var_0, var_0, var_0, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
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
    module_0.read_user_dict(var_1, var_1, prefix=var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    module_0.prompt_choice_for_template(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = ' V};A'
    var_1 = 'V;'
    module_0.read_user_choice(var_1, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_yes_no(var_0, var_1, var_1)

def test_case_26():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__dir__()
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_0, var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.JsonPrompt(console=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
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
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_2 = module_2._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = var_2.__dir__()
    var_4 = '3HwsMk!2e7'
    module_0.prompt_choice_for_template(var_4, var_2, var_4)

def test_case_28():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'CpZ='
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_2 = var_1.__dir__()
    module_0.read_user_dict(var_2, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_variable(var_0, var_1, var_1)

def test_case_31():
    var_0 = '1'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = False
    var_1 = True
    var_2 = module_2._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = 'grPN?xBl'
    var_4 = {var_0: var_2}
    var_5 = module_0.prompt_choice_for_template(var_3, var_4, var_1)
    assert var_5 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_6 = None
    var_7 = None
    var_8 = module_4.iterencode(var_7, var_7, **var_2)
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
    var_8.__new__(var_2, var_6, var_6, incrementaldecoder=var_6)

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
    var_1 = 'maybe'
    with pytest.raises(module_1.InvalidResponse):
        var_0.process_response(var_1)

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    var_2 = None
    var_3 = module_0.render_variable(var_2, var_1, var_1)
    var_4 = 'S&rX;?yB\nb'
    var_5 = {var_4: var_3}
    module_0.prompt_choice_for_template(var_4, var_5, var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = '"\'3B,'
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
    var_2 = var_1.__dir__()
    var_3 = None
    module_0.render_variable(var_3, var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = '{"key": "value", "nubber": 123, "bool": true}'
    var_1 = module_0.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'bool'
    var_2 = module_0.process_json(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.OrderedDict'
    assert len(var_2) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_variable(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'bool'
    var_2 = module_0.process_json(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.OrderedDict'
    assert len(var_2) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_yes_no(var_1, var_2, var_2)

def test_case_39():
    var_0 = module_5.purge()
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    var_1 = None
    var_2 = []
    var_3 = module_0.render_variable(var_0, var_2, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'bool'
    var_2 = module_0.process_json(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.OrderedDict'
    assert len(var_2) == 3
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_choice(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    module_0.prompt_for_config(var_3)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = '__prompts__'
    var_3 = '__prompt__'
    var_4 = {var_3: var_0}
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_6.Environment()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'jinja2.environment.Environment'
    assert var_8.block_start_string == '{%'
    assert var_8.block_end_string == '%}'
    assert var_8.variable_start_string == '{{'
    assert var_8.variable_end_string == '}}'
    assert var_8.comment_start_string == '{#'
    assert var_8.comment_end_string == '#}'
    assert var_8.line_statement_prefix is None
    assert var_8.line_comment_prefix is None
    assert var_8.trim_blocks is False
    assert var_8.lstrip_blocks is False
    assert var_8.newline_sequence == '\n'
    assert var_8.keep_trailing_newline is False
    assert var_8.optimized is True
    assert var_8.finalize is None
    assert var_8.autoescape is False
    assert f'{type(var_8.filters).__module__}.{type(var_8.filters).__qualname__}' == 'builtins.dict'
    assert len(var_8.filters) == 54
    assert f'{type(var_8.tests).__module__}.{type(var_8.tests).__qualname__}' == 'builtins.dict'
    assert len(var_8.tests) == 39
    assert f'{type(var_8.globals).__module__}.{type(var_8.globals).__qualname__}' == 'builtins.dict'
    assert len(var_8.globals) == 6
    assert var_8.loader is None
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.extensions == {}
    assert var_8.is_async is False
    assert module_6.BLOCK_END_STRING == '%}'
    assert module_6.BLOCK_START_STRING == '{%'
    assert module_6.COMMENT_END_STRING == '#}'
    assert module_6.COMMENT_START_STRING == '{#'
    assert f'{type(module_6.DEFAULT_FILTERS).__module__}.{type(module_6.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_FILTERS) == 54
    assert f'{type(module_6.DEFAULT_NAMESPACE).__module__}.{type(module_6.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_NAMESPACE) == 6
    assert module_6.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_6.DEFAULT_TESTS).__module__}.{type(module_6.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_TESTS) == 39
    assert module_6.KEEP_TRAILING_NEWLINE is False
    assert module_6.LINE_COMMENT_PREFIX is None
    assert module_6.LINE_STATEMENT_PREFIX is None
    assert module_6.LSTRIP_BLOCKS is False
    assert module_6.NEWLINE_SEQUENCE == '\n'
    assert module_6.TRIM_BLOCKS is False
    assert module_6.VARIABLE_END_STRING == '}}'
    assert module_6.VARIABLE_START_STRING == '{{'
    assert f'{type(module_6.missing).__module__}.{type(module_6.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_6.Environment.sandboxed is False
    assert module_6.Environment.overlayed is False
    assert module_6.Environment.linked_to is None
    assert module_6.Environment.shared is False
    assert f'{type(module_6.Environment.lexer).__module__}.{type(module_6.Environment.lexer).__qualname__}' == 'builtins.property'
    var_9 = False
    module_0.prompt_for_config(var_7, var_9)

def test_case_43():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.prompt_for_config(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.OrderedDict'
    assert len(var_4) == 0
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_44():
    var_0 = 'cookiecutter'
    var_1 = '{{ cookiecutter.non_existent }}'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = True
    with pytest.raises(module_7.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 'cookiecutter'
    var_1 = 'author'
    var_2 = {var_1: var_1}
    var_3 = {var_0: var_2}
    var_4 = module_6.Environment()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'jinja2.environment.Environment'
    assert var_4.block_start_string == '{%'
    assert var_4.block_end_string == '%}'
    assert var_4.variable_start_string == '{{'
    assert var_4.variable_end_string == '}}'
    assert var_4.comment_start_string == '{#'
    assert var_4.comment_end_string == '#}'
    assert var_4.line_statement_prefix is None
    assert var_4.line_comment_prefix is None
    assert var_4.trim_blocks is False
    assert var_4.lstrip_blocks is False
    assert var_4.newline_sequence == '\n'
    assert var_4.keep_trailing_newline is False
    assert var_4.optimized is True
    assert var_4.finalize is None
    assert var_4.autoescape is False
    assert f'{type(var_4.filters).__module__}.{type(var_4.filters).__qualname__}' == 'builtins.dict'
    assert len(var_4.filters) == 54
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 6
    assert var_4.loader is None
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.extensions == {}
    assert var_4.is_async is False
    assert module_6.BLOCK_END_STRING == '%}'
    assert module_6.BLOCK_START_STRING == '{%'
    assert module_6.COMMENT_END_STRING == '#}'
    assert module_6.COMMENT_START_STRING == '{#'
    assert f'{type(module_6.DEFAULT_FILTERS).__module__}.{type(module_6.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_FILTERS) == 54
    assert f'{type(module_6.DEFAULT_NAMESPACE).__module__}.{type(module_6.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_NAMESPACE) == 6
    assert module_6.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_6.DEFAULT_TESTS).__module__}.{type(module_6.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_TESTS) == 39
    assert module_6.KEEP_TRAILING_NEWLINE is False
    assert module_6.LINE_COMMENT_PREFIX is None
    assert module_6.LINE_STATEMENT_PREFIX is None
    assert module_6.LSTRIP_BLOCKS is False
    assert module_6.NEWLINE_SEQUENCE == '\n'
    assert module_6.TRIM_BLOCKS is False
    assert module_6.VARIABLE_END_STRING == '}}'
    assert module_6.VARIABLE_START_STRING == '{{'
    assert f'{type(module_6.missing).__module__}.{type(module_6.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_6.Environment.sandboxed is False
    assert module_6.Environment.overlayed is False
    assert module_6.Environment.linked_to is None
    assert module_6.Environment.shared is False
    assert f'{type(module_6.Environment.lexer).__module__}.{type(module_6.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = False
    module_0.prompt_for_config(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = '__prompts__'
    var_3 = {}
    var_4 = {var_1: var_3}
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_6.Environment()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'jinja2.environment.Environment'
    assert var_7.block_start_string == '{%'
    assert var_7.block_end_string == '%}'
    assert var_7.variable_start_string == '{{'
    assert var_7.variable_end_string == '}}'
    assert var_7.comment_start_string == '{#'
    assert var_7.comment_end_string == '#}'
    assert var_7.line_statement_prefix is None
    assert var_7.line_comment_prefix is None
    assert var_7.trim_blocks is False
    assert var_7.lstrip_blocks is False
    assert var_7.newline_sequence == '\n'
    assert var_7.keep_trailing_newline is False
    assert var_7.optimized is True
    assert var_7.finalize is None
    assert var_7.autoescape is False
    assert f'{type(var_7.filters).__module__}.{type(var_7.filters).__qualname__}' == 'builtins.dict'
    assert len(var_7.filters) == 54
    assert f'{type(var_7.tests).__module__}.{type(var_7.tests).__qualname__}' == 'builtins.dict'
    assert len(var_7.tests) == 39
    assert f'{type(var_7.globals).__module__}.{type(var_7.globals).__qualname__}' == 'builtins.dict'
    assert len(var_7.globals) == 6
    assert var_7.loader is None
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
    assert module_6.BLOCK_END_STRING == '%}'
    assert module_6.BLOCK_START_STRING == '{%'
    assert module_6.COMMENT_END_STRING == '#}'
    assert module_6.COMMENT_START_STRING == '{#'
    assert f'{type(module_6.DEFAULT_FILTERS).__module__}.{type(module_6.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_FILTERS) == 54
    assert f'{type(module_6.DEFAULT_NAMESPACE).__module__}.{type(module_6.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_NAMESPACE) == 6
    assert module_6.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_6.DEFAULT_TESTS).__module__}.{type(module_6.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_TESTS) == 39
    assert module_6.KEEP_TRAILING_NEWLINE is False
    assert module_6.LINE_COMMENT_PREFIX is None
    assert module_6.LINE_STATEMENT_PREFIX is None
    assert module_6.LSTRIP_BLOCKS is False
    assert module_6.NEWLINE_SEQUENCE == '\n'
    assert module_6.TRIM_BLOCKS is False
    assert module_6.VARIABLE_END_STRING == '}}'
    assert module_6.VARIABLE_START_STRING == '{{'
    assert f'{type(module_6.missing).__module__}.{type(module_6.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_6.Environment.sandboxed is False
    assert module_6.Environment.overlayed is False
    assert module_6.Environment.linked_to is None
    assert module_6.Environment.shared is False
    assert f'{type(module_6.Environment.lexer).__module__}.{type(module_6.Environment.lexer).__qualname__}' == 'builtins.property'
    var_8 = False
    module_0.prompt_for_config(var_6, var_8)

def test_case_47():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = '__p@rompts__'
    var_3 = {var_2: var_2}
    var_4 = {var_1: var_3, var_2: var_1}
    var_5 = {var_0: var_4}
    var_6 = True
    var_7 = module_0.prompt_for_config(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.OrderedDict'
    assert len(var_7) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_8 = ''
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_8)

def test_case_48():
    var_0 = '1'
    var_1 = 'cookiecutter'
    var_2 = 'version'
    var_3 = 'options_list'
    var_4 = 'w~`E?7j"0yb>z?>jY|'
    var_5 = 'opt1'
    var_6 = [var_5, var_2]
    var_7 = True
    var_8 = {var_4: var_1, var_4: var_5, var_1: var_0}
    var_9 = {var_4: var_0, var_2: var_4, var_3: var_6, var_4: var_7, var_1: var_8}
    var_10 = {var_1: var_9}
    var_11 = True
    var_12 = module_0.prompt_for_config(var_10, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.OrderedDict'
    assert len(var_12) == 4
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = '__p@rompts__'
    var_2 = {var_1: var_1}
    var_3 = '__prompt__'
    var_4 = ":n[3}'\x0cd"
    var_5 = {var_3: var_4}
    var_6 = {var_1: var_5, var_1: var_2, var_3: var_2}
    var_7 = {var_0: var_2, var_1: var_6, var_0: var_1}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.OrderedDict'
    assert len(var_10) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_11 = ''
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_11)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = 'cookiecutter'
    var_1 = '__p@rompts__'
    var_2 = 'author'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = '__prompt__'
    var_6 = ":n[3}'\x0cd"
    var_7 = {var_5: var_6}
    var_8 = {var_3: var_7}
    var_9 = {var_1: var_4, var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_6.Environment()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'jinja2.environment.Environment'
    assert var_11.block_start_string == '{%'
    assert var_11.block_end_string == '%}'
    assert var_11.variable_start_string == '{{'
    assert var_11.variable_end_string == '}}'
    assert var_11.comment_start_string == '{#'
    assert var_11.comment_end_string == '#}'
    assert var_11.line_statement_prefix is None
    assert var_11.line_comment_prefix is None
    assert var_11.trim_blocks is False
    assert var_11.lstrip_blocks is False
    assert var_11.newline_sequence == '\n'
    assert var_11.keep_trailing_newline is False
    assert var_11.optimized is True
    assert var_11.finalize is None
    assert var_11.autoescape is False
    assert f'{type(var_11.filters).__module__}.{type(var_11.filters).__qualname__}' == 'builtins.dict'
    assert len(var_11.filters) == 54
    assert f'{type(var_11.tests).__module__}.{type(var_11.tests).__qualname__}' == 'builtins.dict'
    assert len(var_11.tests) == 39
    assert f'{type(var_11.globals).__module__}.{type(var_11.globals).__qualname__}' == 'builtins.dict'
    assert len(var_11.globals) == 6
    assert var_11.loader is None
    assert f'{type(var_11.cache).__module__}.{type(var_11.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_11.cache) == 0
    assert var_11.bytecode_cache is None
    assert var_11.auto_reload is True
    assert var_11.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_11.extensions == {}
    assert var_11.is_async is False
    assert module_6.BLOCK_END_STRING == '%}'
    assert module_6.BLOCK_START_STRING == '{%'
    assert module_6.COMMENT_END_STRING == '#}'
    assert module_6.COMMENT_START_STRING == '{#'
    assert f'{type(module_6.DEFAULT_FILTERS).__module__}.{type(module_6.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_FILTERS) == 54
    assert f'{type(module_6.DEFAULT_NAMESPACE).__module__}.{type(module_6.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_NAMESPACE) == 6
    assert module_6.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_6.DEFAULT_TESTS).__module__}.{type(module_6.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_6.DEFAULT_TESTS) == 39
    assert module_6.KEEP_TRAILING_NEWLINE is False
    assert module_6.LINE_COMMENT_PREFIX is None
    assert module_6.LINE_STATEMENT_PREFIX is None
    assert module_6.LSTRIP_BLOCKS is False
    assert module_6.NEWLINE_SEQUENCE == '\n'
    assert module_6.TRIM_BLOCKS is False
    assert module_6.VARIABLE_END_STRING == '}}'
    assert module_6.VARIABLE_START_STRING == '{{'
    assert f'{type(module_6.missing).__module__}.{type(module_6.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_6.Environment.sandboxed is False
    assert module_6.Environment.overlayed is False
    assert module_6.Environment.linked_to is None
    assert module_6.Environment.shared is False
    assert f'{type(module_6.Environment.lexer).__module__}.{type(module_6.Environment.lexer).__qualname__}' == 'builtins.property'
    var_12 = False
    var_13 = module_0.prompt_for_config(var_10, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.OrderedDict'
    assert len(var_13) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.choose_nested_template(var_8, var_2)

def test_case_51():
    var_0 = '\n    Tests the second pass of prompt_for_config which handles dictionary variables.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'metadata'
    var_3 = '_Lp@rompts__'
    var_4 = {var_3: var_0}
    var_5 = {var_2: var_4, var_3: var_2}
    var_6 = {var_1: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.OrderedDict'
    assert len(var_8) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_9 = ''
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_9)

def test_case_52():
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
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = 'New Project'
    var_1 = '{"new_key": "val"}'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'options_list'
    var_5 = 'default_name'
    var_6 = 'cv*A<U/6;X;.>6VQ'
    var_7 = [var_6, var_0]
    var_8 = True
    var_9 = 'kV<'
    var_10 = 'valu7'
    var_11 = {var_0: var_10}
    var_12 = {var_3: var_5, var_1: var_9, var_4: var_7, var_3: var_8, var_2: var_11}
    var_13 = {var_2: var_12}
    var_14 = True
    var_15 = module_0.prompt_for_config(var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.OrderedDict'
    assert len(var_15) == 4
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_for_config(var_13)