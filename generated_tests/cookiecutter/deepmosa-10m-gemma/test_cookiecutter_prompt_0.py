# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import enum as module_1
import rich.prompt as module_2
import re as module_3
import codecs as module_4
import builtins as module_5
import pathlib as module_6
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
    var_0 = 'sK@lx[ph?\x0cDCg}'
    module_0.read_user_choice(var_0, var_0, var_0)

def test_case_4():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_0, var_0, var_0, var_0, prefix=var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(module_2.InvalidResponse):
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

def test_case_11():
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
    var_2 = None
    with pytest.raises(module_2.InvalidResponse):
        var_1.process_response(var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_3.RegexFlag.DOTALL
    var_1 = var_0.__dir__()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    module_0.read_user_choice(var_1, var_1, prefix=var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = "/'X_T%e\x0cyYXJv}\tzQO@"
    var_1 = {var_0: var_0}
    var_2 = None
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

def test_case_14():
    var_0 = module_1._EnumDict()
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
    var_0 = 'c$\n,'
    var_1 = None
    module_0.render_variable(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '[VH868'
    module_0.read_user_yes_no(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'cookiecutter'
    var_1 = module_4.IncrementalEncoder(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'codecs.IncrementalEncoder'
    assert var_1.errors == 'cookiecutter'
    assert var_1.buffer == ''
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
    var_3 = {var_0: var_2, var_0: var_2}
    module_0.prompt_for_config(var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '{tkRO'
    module_0.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = True
    var_2 = module_0.render_variable(var_0, var_1, var_0)
    assert var_2 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = '8Ij[j0f5k69 '
    module_0.read_user_dict(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = ''
    var_1 = {}
    var_2 = False
    module_0.prompt_choice_for_template(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_3.purge()
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_1 = var_0.__dir__()
    module_0.render_variable(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'l;oPZ='
    var_2 = None
    var_3 = module_5.ValueError(*var_1)
    var_4 = module_0.render_variable(var_2, var_2, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_5 = '3HwsMk!2e7'
    var_6 = {var_4: var_2, var_4: var_0}
    var_7 = module_0.prompt_choice_for_template(var_2, var_6, var_6)
    module_0.prompt_choice_for_template(var_5, var_6, var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__dir__()
    var_2 = '6S\nT!gGmV.o\n~~Awy'
    var_3 = '[VH868'
    module_0.read_user_yes_no(var_3, var_2, var_1)

def test_case_26():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = 'T6m+CeyONN'
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_2, var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'l;oPZ='
    var_2 = module_5.ValueError(*var_1)
    var_3 = {var_2: var_0}
    var_4 = module_0.prompt_choice_for_template(var_0, var_3, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.prompt_and_delete(var_3)

def test_case_28():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = '$e~N&w7oUqV1'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__dir__()
    var_2 = module_0.render_variable(var_0, var_0, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = 'qyp 33 O'
    var_4 = '-W/@bNH"n^G'
    module_0.read_user_dict(var_3, var_2, var_4, var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__dir__()
    var_2 = module_0.render_variable(var_0, var_0, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_variable(var_1, var_1, var_1)

def test_case_31():
    var_0 = '1'
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    module_0.prompt_for_config(var_3)

def test_case_33():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'l;oPZ='
    var_2 = None
    var_3 = module_5.ValueError(*var_1)
    var_4 = '3HwsMk!2e7'
    var_5 = False
    var_6 = {var_4: var_2, var_4: var_0}
    var_7 = module_0.prompt_choice_for_template(var_2, var_6, var_6)
    assert var_7 == '3HwsMk!2e7'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_8 = '3tK3tVbD6%k9'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_8, var_5)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'test_dir_to_delete'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_6.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_6.EINVAL == 22
    assert module_6.ENOENT == 2
    assert module_6.ENOTDIR == 20
    assert module_6.EBADF == 9
    assert module_6.ELOOP == 40
    module_0.prompt_and_delete(var_0, var_0)
    var_4 = module_0.process_json(var_1)

def test_case_35():
    var_0 = ''
    var_1 = module_0.YesNoPrompt(var_0, console=var_0, password=var_0, show_default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 0
    assert var_1.password == ''
    assert var_1.case_sensitive is True
    assert var_1.show_default == ''
    assert var_1.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    with pytest.raises(module_2.InvalidResponse):
        var_1.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_36():
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
    module_0.read_user_yes_no(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'default_val'
    module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'user_input'

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 'test_var'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'def'
    module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'val'

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = []
    var_1 = module_6.Path(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pathlib.PosixPath'
    assert module_6.EINVAL == 22
    assert module_6.ENOENT == 2
    assert module_6.ENOTDIR == 20
    assert module_6.EBADF == 9
    assert module_6.ELOOP == 40
    var_2 = None
    var_3 = module_0.render_variable(var_2, var_0, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_1.mkdir(exist_ok=var_2)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 'delete'
    var_1 = {var_0: var_0}
    var_2 = False
    var_3 = 'CONFIRM: '
    module_0.read_user_yes_no(var_0, var_2, var_1, var_3)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'empty'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = False
    module_0.read_user_yes_no(var_0, var_3, var_2, var_1)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = {}
    var_1 = '8'
    var_2 = {var_1: var_1}
    module_0.read_user_dict(var_1, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = {}
    var_1 = ''
    var_2 = {var_1: var_1}
    module_0.read_user_dict(var_1, var_0, var_2)

def test_case_44():
    var_0 = 't'
    var_1 = module_0.YesNoPrompt(var_0, console=var_0, password=var_0, show_default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert var_1.console == 't'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 1
    assert var_1.password == 't'
    assert var_1.case_sensitive is True
    assert var_1.show_default == 't'
    assert var_1.show_choices is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    assert module_0.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_0.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_2 = var_1.process_response(var_0)
    assert var_2 is True
    var_3 = 'l\\'
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_3)

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    var_1 = None
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    var_3 = '0'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = module_0.YesNoPrompt(show_default=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_5.console).__module__}.{type(var_5.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_5.prompt).__module__}.{type(var_5.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_5.prompt) == 0
    assert var_5.password is False
    assert var_5.case_sensitive is True
    assert var_5.show_default is False
    assert var_5.show_choices is True
    var_6 = None
    module_0.read_user_variable(var_0, var_6)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = '1'
    var_1 = 'Vb{S,\x0c7xNHo'
    var_2 = [var_1, var_0]
    var_3 = 'fruit'
    var_4 = '__prompt__'
    var_5 = '1'
    var_6 = '\n'
    var_7 = 'Qck a flmvor:'
    var_8 = {var_4: var_7, var_5: var_6, var_6: var_6}
    var_9 = {var_3: var_8, var_0: var_8, var_3: var_8}
    module_0.read_user_choice(var_3, var_2, var_9)
    assert var_10 == 'apple'

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = 'banana'
    var_1 = [var_0, var_0]
    var_2 = {var_0: var_0}
    module_0.read_user_choice(var_0, var_1, var_2)
    assert var_3 == 'apple'

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 'fruit'
    var_4 = '1'
    var_5 = '2'
    var_6 = 'Pick a flavor:'
    var_7 = 'Sour Banana'
    var_8 = {var_6: var_6, var_4: var_0, var_5: var_7}
    var_9 = {var_3: var_8}
    module_0.read_user_choice(var_3, var_2, var_9)
    assert var_10 == 'apple'

def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = '/absolute/path/to/template'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_8, var_9, var_10)

def test_case_50():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo7'
    var_10 = True
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_8, var_9, var_10)

def test_case_51():
    var_0 = 'cookiecutter'
    var_1 = 'path'
    var_2 = {}
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = module_0.prompt_for_config(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 0
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(ValueError):
        module_0.choose_nested_template(var_3, var_1, var_4)

def test_case_52():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = '_internal'
    var_3 = 'web'
    var_4 = 'api'
    var_5 = 'cli'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'val'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.prompt_for_config(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.OrderedDict'
    assert len(var_11) == 2
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_12 = var_11['type']
    assert var_12 == 'web'

def test_case_53():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice1 (templates/choice1)'
    var_3 = [var_2, var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = '/tmp/repo'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_6.Path(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pathlib.PosixPath'
    assert module_6.EINVAL == 22
    assert module_6.ENOENT == 2
    assert module_6.ENOTDIR == 20
    assert module_6.EBADF == 9
    assert module_6.ELOOP == 40
    var_11 = '/tmp/repo/templates/choice1'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_6.Path(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pathlib.PosixPath'
    var_15 = True
    var_16 = module_0.choose_nested_template(var_5, var_6, var_15)
    assert var_16 == '/tmp/repo/templates/choice1'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_17 = [var_11]
    var_18 = {}
    var_19 = module_6.Path(*var_17, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pathlib.PosixPath'
    var_20 = str(var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    module_0.prompt_for_config(var_2)

def test_case_55():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = module_0.prompt_for_config(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.OrderedDict'
    assert len(var_3) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_56():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2, var_0: var_2, var_0: var_2}
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
    with pytest.raises(module_2.InvalidResponse):
        module_0.process_json(var_2)

@pytest.mark.xfail(strict=True)
def test_case_57():
    var_0 = 'cookiecutter'
    var_1 = '__template_var__'
    var_2 = 'test'
    var_3 = '{{ cookiecutter.project_name }}'
    var_4 = {var_2: var_2, var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = True
    module_0.prompt_for_config(var_5, var_6)

def test_case_58():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__template_var__'
    var_3 = 'test'
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
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
    var_9 = var_8['__template_var__']
    assert var_9 == 'test'

@pytest.mark.xfail(strict=True)
def test_case_59():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    module_0.prompt_for_config(var_6, var_7)

def test_case_60():
    var_0 = 'cookiecutter'
    var_1 = 'is_enabled'
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
    var_6 = var_5['is_enabled']

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = 'cookiecutter'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = False
    module_0.prompt_for_config(var_3, var_4)

def test_case_62():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '{{ non_existent_var }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    with pytest.raises(module_7.UndefinedVariableInTemplate):
        module_0.prompt_for_config(var_4, var_5)