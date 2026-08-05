# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.utils as module_0
import builtins as module_1
import pathlib as module_2
import cookiecutter.environment as module_3

def test_case_0():
    var_0 = module_0.work_in()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.force_delete(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.work_in()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_1 = var_0.__enter__()
    module_0.rmtree(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.make_executable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.simple_filter(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'UV}WQrw'
    module_0.create_tmp_repo_dir(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_1.type
    module_0.create_env_with_context(var_0)

def test_case_8():
    var_0 = 'conflict_file'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = module_3.StrictEnvironment()
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
    assert var_4.keep_trailing_newline is False
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
    var_5 = 'conflict_file/sub_dir'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_2.Path(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pathlib.PosixPath'
    var_9 = module_0.make_sure_path_exists(var_8)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_9():
    var_0 = module_0.work_in()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_1 = var_0.__enter__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'conflic%_file'
    var_1 = module_0.work_in()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == ()
    assert var_1.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = module_0.work_in(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_2.gen).__module__}.{type(var_2.gen).__qualname__}' == 'builtins.generator'
    assert var_2.args == ('conflic%_file',)
    assert var_2.kwds == {}
    var_2.__enter__()
    var_3 = module_0.work_in()
    var_4 = None
    var_5 = module_0.make_sure_path_exists(var_4)