# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import re as module_2
import cookiecutter.utils as module_3
import enum as module_4

def test_case_0():
    var_0 = module_0.JsonPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
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
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'

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
    var_0 = 'JQE+!'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_0, var_0)

def test_case_6():
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
    var_0 = None
    var_1 = {}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'fp?rO7T9&'
    module_0.read_repo_password(var_0)

def test_case_11():
    var_0 = {}
    var_1 = module_3.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_2, var_2, var_2)

def test_case_12():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = module_0.render_variable(var_1, var_2, var_1)
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_3, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = '4`>02Lfl<'
    module_0.read_user_choice(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '{`0~LY9l'
    module_0.read_user_choice(var_0, var_0, prefix=var_0)

def test_case_15():
    var_0 = {}
    var_1 = None
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_1, var_0, var_0, var_0, var_1, prefix=var_0)

def test_case_16():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_0, var_0, var_0, var_0, var_1, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'Li+2\n[9 gnd'
    module_0.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'km8B'
    module_0.prompt_and_delete(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = {var_0: var_0}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.JsonPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
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
    assert module_0.JsonPrompt.default is None
    assert module_0.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_1 = '6)\t1Gt\x0bW((Ehra\ny'
    var_0.process_response(var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'Xe[-(q6&b\nmXI;8'
    var_1 = ")-S@{}W}{ri{'"
    var_2 = None
    module_0.read_user_variable(var_1, var_2, var_0, var_2)

def test_case_22():
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
    var_1 = ''
    with pytest.raises(module_1.InvalidResponse):
        var_0.process_response(var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = {}
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = True
    module_0.prompt_choice_for_template(var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = {}
    module_0.read_user_dict(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = {}
    var_1 = module_3.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = module_0.render_variable(var_1, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = module_0.prompt_choice_for_config(var_2, var_1, var_2, var_2, var_2, var_2, var_2)
    assert var_3 == '<'
    module_0.read_user_choice(var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = False
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    module_0.read_user_choice(var_2, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.render_variable(var_2, var_1, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_choice(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = True
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.render_variable(var_0, var_0, var_2)
    assert var_3 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    module_0.read_user_choice(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = False
    var_2 = 'x#XCU/5'
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_2, var_2: var_2}
    module_0.prompt_choice_for_template(var_2, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = '6I~^WG:Qzk\nz'
    module_0.read_user_yes_no(var_0, var_0, var_0)

def test_case_31():
    var_0 = {}
    var_1 = module_2.RegexFlag.TEMPLATE
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = {}
    var_1 = module_3.create_env_with_context(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is True
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 56
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 8
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.datetime_format == '%Y-%m-%d'
    assert f'{type(var_1.extensions).__module__}.{type(var_1.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_1.extensions) == 5
    assert var_1.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = module_0.render_variable(var_1, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = var_2.__iter__()
    module_0.read_user_dict(var_3, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = {}
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    var_3 = module_0.render_variable(var_1, var_0, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_4 = module_3.create_env_with_context(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert var_4.keep_trailing_newline is True
    assert var_4.optimized is True
    assert var_4.finalize is None
    assert var_4.autoescape is False
    assert f'{type(var_4.filters).__module__}.{type(var_4.filters).__qualname__}' == 'builtins.dict'
    assert len(var_4.filters) == 56
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 8
    assert var_4.loader is None
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.datetime_format == '%Y-%m-%d'
    assert f'{type(var_4.extensions).__module__}.{type(var_4.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_4.extensions) == 5
    assert var_4.is_async is False
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    assert module_3.TYPE_CHECKING is False
    assert f'{type(module_3.logger).__module__}.{type(module_3.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_3.logger.filters == []
    assert module_3.logger.name == 'cookiecutter.utils'
    assert module_3.logger.level == 0
    assert f'{type(module_3.logger.parent).__module__}.{type(module_3.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_3.logger.propagate is True
    assert module_3.logger.handlers == []
    assert module_3.logger.disabled is False
    assert f'{type(module_3.logger.manager).__module__}.{type(module_3.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = module_0.render_variable(var_1, var_1, var_3)
    var_5.raw_decode(var_2)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'LU]uL]IQ#X=.G#J^"'
    var_1 = True
    var_2 = [var_1, var_0, var_0]
    module_0.read_user_choice(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    module_0.read_user_choice(var_0, var_1, var_1)

def test_case_36():
    var_0 = '2'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = False
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = '\t,NmBC'
    module_0.read_user_variable(var_0, var_2, var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = '!l:m\\D&\nsWNx>^'
    var_1 = var_0.__iter__()
    module_0.read_user_yes_no(var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = True
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = '\t,+9Bmy'
    module_0.read_user_variable(var_0, var_2, var_1, var_3)

def test_case_40():
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
    var_1 = '\t1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = False
    var_2 = 'W('
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_2, var_2: var_0, var_2: var_0}
    module_0.prompt_choice_for_template(var_2, var_3, var_1)