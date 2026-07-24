# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import enum as module_2
import cookiecutter.utils as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = True
    var_1 = '[F!Ao.Q!Z4Ex'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_0}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_4 = 'aT-O*`1bK@?,**Fc09'
    module_0.is_copy_only_path(var_4, var_3)

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.generate_context()

def test_case_3():
    var_0 = None
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.generate_files(var_0, var_0)

def test_case_5():
    var_0 = 'SW&"\x0cp,uc`3Mg'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.apply_overwrites_to_context(var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_6():
    var_0 = True
    var_1 = '[F!Ao.Q!Z4Ex'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_0}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'I'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = True
    var_2 = 'o\\\tw E)WZQ'
    var_3 = '[F!Ao.Q!Z4Ex'
    var_4 = {var_3: var_2, var_3: var_3, var_3: var_1}
    module_0.generate_files(var_0, var_4, accept_hooks=var_0, keep_project_on_failure=var_0)

def test_case_9():
    var_0 = 'type'
    var_1 = 'CEF"\x0b8F&Ww\th]`9o]>'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_0}
    var_3 = False
    var_4 = module_0.apply_overwrites_to_context(var_2, var_2, in_dictionary_variable=var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = [var_1, var_0]
    var_6 = {var_0: var_5}
    var_7 = {var_0: var_1, var_1: var_5}
    var_8 = module_0.apply_overwrites_to_context(var_6, var_7)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = True
    var_1 = 'o\\\tw EWZQ'
    var_2 = 'O\r *>hA'
    var_3 = {var_2: var_1}
    module_0.apply_overwrites_to_context(var_1, var_3, in_dictionary_variable=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'repo'
    var_1 = 'post_gen_project'
    var_2 = 'project'
    var_3 = {}
    var_4 = False
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = False
    module_0.generate_context(var_0)

def test_case_13():
    var_0 = False
    var_1 = 'o\\\tw EWZQ'
    var_2 = module_2._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = {var_1: var_0, var_1: var_1, var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_3, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_14():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = False
    var_1 = 'o\\\tw EWZQ'
    var_2 = '[F!Ao.Q!Z4Ex'
    var_3 = {var_2: var_1, var_1: var_2, var_2: var_2, var_2: var_0, var_2: var_0}
    var_4 = {var_2: var_0, var_2: var_1, var_1: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = '%O'
    var_7 = None
    module_0.render_and_create_dir(var_6, var_3, var_5, var_7)

def test_case_16():
    var_0 = 'CEF"\x0b8F&Ww\th]`9o]>'
    var_1 = [var_0, var_0]
    var_2 = {var_0: var_0, var_0: var_1}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_17():
    var_0 = 'small'
    var_1 = [var_0, var_0]
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    var_4 = module_0.apply_overwrites_to_context(var_2, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_18():
    var_0 = 'type'
    var_1 = 'small'
    var_2 = [var_1, var_1]
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_4)

def test_case_19():
    var_0 = 'CEF"\x0b8F&Ww\th]$9]>'
    var_1 = '&'
    var_2 = '('
    var_3 = {var_2: var_0, var_1: var_1, var_2: var_0, var_0: var_1, var_0: var_0, var_2: var_0}
    var_4 = module_0.is_copy_only_path(var_1, var_3)
    assert var_4 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = [var_0, var_1]
    var_6 = {var_0: var_5, var_1: var_0}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_6, var_6, in_dictionary_variable=var_7)

def test_case_20():
    var_0 = 'Y'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = False
    var_3 = [var_0, var_0, var_0]
    var_4 = {var_2: var_3, var_0: var_2}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_4)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = module_3.create_env_with_context(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_6.block_start_string == '{%'
    assert var_6.block_end_string == '%}'
    assert var_6.variable_start_string == '{{'
    assert var_6.variable_end_string == '}}'
    assert var_6.comment_start_string == '{#'
    assert var_6.comment_end_string == '#}'
    assert var_6.line_statement_prefix is None
    assert var_6.line_comment_prefix is None
    assert var_6.trim_blocks is False
    assert var_6.lstrip_blocks is False
    assert var_6.newline_sequence == '\n'
    assert var_6.keep_trailing_newline is True
    assert var_6.optimized is True
    assert var_6.finalize is None
    assert var_6.autoescape is False
    assert f'{type(var_6.filters).__module__}.{type(var_6.filters).__qualname__}' == 'builtins.dict'
    assert len(var_6.filters) == 56
    assert f'{type(var_6.tests).__module__}.{type(var_6.tests).__qualname__}' == 'builtins.dict'
    assert len(var_6.tests) == 39
    assert f'{type(var_6.globals).__module__}.{type(var_6.globals).__qualname__}' == 'builtins.dict'
    assert len(var_6.globals) == 8
    assert var_6.loader is None
    assert f'{type(var_6.cache).__module__}.{type(var_6.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_6.cache) == 0
    assert var_6.bytecode_cache is None
    assert var_6.auto_reload is True
    assert var_6.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_6.datetime_format == '%Y-%m-%d'
    assert f'{type(var_6.extensions).__module__}.{type(var_6.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_6.extensions) == 5
    assert var_6.is_async is False
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
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_6)

def test_case_21():
    var_0 = 'Y'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_3.create_env_with_context(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_2.block_start_string == '{%'
    assert var_2.block_end_string == '%}'
    assert var_2.variable_start_string == '{{'
    assert var_2.variable_end_string == '}}'
    assert var_2.comment_start_string == '{#'
    assert var_2.comment_end_string == '#}'
    assert var_2.line_statement_prefix is None
    assert var_2.line_comment_prefix is None
    assert var_2.trim_blocks is False
    assert var_2.lstrip_blocks is False
    assert var_2.newline_sequence == '\n'
    assert var_2.keep_trailing_newline is True
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 56
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 8
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.datetime_format == '%Y-%m-%d'
    assert f'{type(var_2.extensions).__module__}.{type(var_2.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_2.extensions) == 5
    assert var_2.is_async is False
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
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'P$lh%s\\&'
    var_1 = {var_0: var_0}
    var_2 = module_3.create_env_with_context(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_2.block_start_string == '{%'
    assert var_2.block_end_string == '%}'
    assert var_2.variable_start_string == '{{'
    assert var_2.variable_end_string == '}}'
    assert var_2.comment_start_string == '{#'
    assert var_2.comment_end_string == '#}'
    assert var_2.line_statement_prefix is None
    assert var_2.line_comment_prefix is None
    assert var_2.trim_blocks is False
    assert var_2.lstrip_blocks is False
    assert var_2.newline_sequence == '\n'
    assert var_2.keep_trailing_newline is True
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 56
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 8
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.datetime_format == '%Y-%m-%d'
    assert f'{type(var_2.extensions).__module__}.{type(var_2.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_2.extensions) == 5
    assert var_2.is_async is False
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
    module_0.generate_file(var_0, var_0, var_1, var_2)
    var_3 = {}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    var_5 = 'Y'
    var_6 = {var_0: var_5, var_5: var_5, var_5: var_0, var_5: var_5}
    var_8 = module_0.is_copy_only_path(var_7, var_1)
    var_9 = module_3.create_env_with_context(var_6)
    var_10 = module_0.render_and_create_dir(var_5, var_6, var_5, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 'P$lh%s\\&'
    var_1 = {var_0: var_0}
    var_2 = module_3.create_env_with_context(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_2.block_start_string == '{%'
    assert var_2.block_end_string == '%}'
    assert var_2.variable_start_string == '{{'
    assert var_2.variable_end_string == '}}'
    assert var_2.comment_start_string == '{#'
    assert var_2.comment_end_string == '#}'
    assert var_2.line_statement_prefix is None
    assert var_2.line_comment_prefix is None
    assert var_2.trim_blocks is False
    assert var_2.lstrip_blocks is False
    assert var_2.newline_sequence == '\n'
    assert var_2.keep_trailing_newline is True
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 56
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 8
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.datetime_format == '%Y-%m-%d'
    assert f'{type(var_2.extensions).__module__}.{type(var_2.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_2.extensions) == 5
    assert var_2.is_async is False
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
    module_0.generate_file(var_0, var_0, var_1, var_2)