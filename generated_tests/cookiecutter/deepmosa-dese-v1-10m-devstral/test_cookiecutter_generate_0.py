# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import jinja2.environment as module_2
import pathlib as module_3
import cookiecutter.utils as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = 'm4_:!3np8NAqtB[&8'
    var_2 = {var_1: var_1}
    var_3 = module_0.is_copy_only_path(var_0, var_2)
    assert var_3 is False
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
    module_0.generate_context(default_context=var_0, extra_context=var_3)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.apply_overwrites_to_context(var_2, var_2)
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
    module_0.is_copy_only_path(var_3, var_4)

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
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'xnG6Lu\x0c^Vc`1'
    var_1 = {var_0: var_0, var_0: var_0}
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
    module_0.apply_overwrites_to_context(var_1, var_2)

def test_case_7():
    var_0 = True
    var_1 = '<#1KUm'
    var_2 = {var_1: var_1, var_1: var_0}
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
def test_case_8():
    var_0 = 'u"TGb\t"'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'repo'
    var_1 = 'hook'
    var_2 = 'project'
    var_3 = {}
    var_4 = False
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'S&w80OBs]9Hpb)mOED'
    var_1 = "ZbT#']Y\x0c\rX\x0c'8p:x,"
    var_2 = {var_1: var_1}
    var_3 = True
    module_0.generate_files(var_0, var_2, overwrite_if_exists=var_3, accept_hooks=var_3)

def test_case_11():
    var_0 = 'ne8cw'
    var_1 = {var_0: var_0}
    var_2 = {var_0: var_1}
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

def test_case_12():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_1: var_1}
    var_3 = {var_0: var_2}
    var_4 = 'ne8cw'
    var_5 = {var_4: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_3, var_6)
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
def test_case_13():
    var_0 = 'b'
    var_1 = 'c'
    var_2 = [var_0, var_0, var_1]
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_2}
    var_5 = module_0.apply_overwrites_to_context(var_3, var_4)
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
    var_6 = {var_0: var_2, var_0: var_4, var_0: var_1, var_0: var_4}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
    module_0.generate_context(default_context=var_6, extra_context=var_4)

def test_case_14():
    var_0 = 'b'
    var_1 = 'c'
    var_2 = [var_1, var_0]
    var_3 = {var_1: var_2}
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

def test_case_15():
    var_0 = 'var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

def test_case_16():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_3, var_2, var_2]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_7)

def test_case_17():
    var_0 = True
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '/tmp'
    var_4 = module_2.Environment()
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
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = module_0.render_and_create_dir(var_1, var_2, var_3, var_4, var_0)
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
def test_case_18():
    var_0 = '<'
    var_1 = {}
    var_2 = module_3.Path()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pathlib.PosixPath'
    assert module_3.EINVAL == 22
    assert module_3.ENOENT == 2
    assert module_3.ENOTDIR == 20
    assert module_3.EBADF == 9
    assert module_3.ELOOP == 40
    var_3 = module_2.Environment()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.environment.Environment'
    assert var_3.block_start_string == '{%'
    assert var_3.block_end_string == '%}'
    assert var_3.variable_start_string == '{{'
    assert var_3.variable_end_string == '}}'
    assert var_3.comment_start_string == '{#'
    assert var_3.comment_end_string == '#}'
    assert var_3.line_statement_prefix is None
    assert var_3.line_comment_prefix is None
    assert var_3.trim_blocks is False
    assert var_3.lstrip_blocks is False
    assert var_3.newline_sequence == '\n'
    assert var_3.keep_trailing_newline is False
    assert var_3.optimized is True
    assert var_3.finalize is None
    assert var_3.autoescape is False
    assert f'{type(var_3.filters).__module__}.{type(var_3.filters).__qualname__}' == 'builtins.dict'
    assert len(var_3.filters) == 54
    assert f'{type(var_3.tests).__module__}.{type(var_3.tests).__qualname__}' == 'builtins.dict'
    assert len(var_3.tests) == 39
    assert f'{type(var_3.globals).__module__}.{type(var_3.globals).__qualname__}' == 'builtins.dict'
    assert len(var_3.globals) == 6
    assert var_3.loader is None
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.extensions == {}
    assert var_3.is_async is False
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.render_and_create_dir(var_0, var_1, var_2, var_3)

def test_case_19():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_2.Environment()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.environment.Environment'
    assert var_3.block_start_string == '{%'
    assert var_3.block_end_string == '%}'
    assert var_3.variable_start_string == '{{'
    assert var_3.variable_end_string == '}}'
    assert var_3.comment_start_string == '{#'
    assert var_3.comment_end_string == '#}'
    assert var_3.line_statement_prefix is None
    assert var_3.line_comment_prefix is None
    assert var_3.trim_blocks is False
    assert var_3.lstrip_blocks is False
    assert var_3.newline_sequence == '\n'
    assert var_3.keep_trailing_newline is False
    assert var_3.optimized is True
    assert var_3.finalize is None
    assert var_3.autoescape is False
    assert f'{type(var_3.filters).__module__}.{type(var_3.filters).__qualname__}' == 'builtins.dict'
    assert len(var_3.filters) == 54
    assert f'{type(var_3.tests).__module__}.{type(var_3.tests).__qualname__}' == 'builtins.dict'
    assert len(var_3.tests) == 39
    assert f'{type(var_3.globals).__module__}.{type(var_3.globals).__qualname__}' == 'builtins.dict'
    assert len(var_3.globals) == 6
    assert var_3.loader is None
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.extensions == {}
    assert var_3.is_async is False
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_2, var_3)

def test_case_20():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_1}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
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

def test_case_21():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_7)

def test_case_22():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2, var_1]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_3}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
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
    var_7 = {var_0: var_3, var_1: var_5, var_0: var_0, var_0: var_5}
    var_8 = module_0.apply_overwrites_to_context(var_7, var_7)

def test_case_23():
    var_0 = 'var'
    var_1 = 'a'
    var_2 = 'c'
    var_3 = [var_1, var_0, var_2]
    var_4 = {var_0: var_3}
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
    var_6 = {var_0: var_3, var_1: var_4, var_0: var_2, var_0: var_4}
    var_7 = module_0.is_copy_only_path(var_5, var_6)
    assert var_7 is False
    var_8 = False
    var_9 = module_0.apply_overwrites_to_context(var_6, var_4, in_dictionary_variable=var_8)

def test_case_24():
    var_0 = 'dir/subdir/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'dir/**'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True
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

def test_case_25():
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False
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
def test_case_26():
    var_0 = {}
    var_1 = module_4.create_env_with_context(var_0)
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
    assert f'{type(module_4.annotations).__module__}.{type(module_4.annotations).__qualname__}' == '__future__._Feature'
    assert module_4.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_4.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_4.annotations.compiler_flag == 16777216
    assert module_4.TYPE_CHECKING is False
    assert f'{type(module_4.logger).__module__}.{type(module_4.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_4.logger.filters == []
    assert module_4.logger.name == 'cookiecutter.utils'
    assert module_4.logger.level == 0
    assert f'{type(module_4.logger.parent).__module__}.{type(module_4.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_4.logger.propagate is True
    assert module_4.logger.handlers == []
    assert module_4.logger.disabled is False
    assert f'{type(module_4.logger.manager).__module__}.{type(module_4.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = 'w'
    module_0.generate_file(var_2, var_2, var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = {}
    var_1 = module_4.create_env_with_context(var_0)
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
    assert f'{type(module_4.annotations).__module__}.{type(module_4.annotations).__qualname__}' == '__future__._Feature'
    assert module_4.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_4.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_4.annotations.compiler_flag == 16777216
    assert module_4.TYPE_CHECKING is False
    assert f'{type(module_4.logger).__module__}.{type(module_4.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_4.logger.filters == []
    assert module_4.logger.name == 'cookiecutter.utils'
    assert module_4.logger.level == 0
    assert f'{type(module_4.logger.parent).__module__}.{type(module_4.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_4.logger.propagate is True
    assert module_4.logger.handlers == []
    assert module_4.logger.disabled is False
    assert f'{type(module_4.logger.manager).__module__}.{type(module_4.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = ''
    module_0.generate_file(var_2, var_2, var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = '/tmp/project'
    var_1 = {var_0: var_0}
    var_2 = module_0.apply_overwrites_to_context(var_1, var_1, in_dictionary_variable=var_1)
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
    var_3 = {}
    var_4 = {var_0: var_3}
    var_5 = module_2.Environment()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'jinja2.environment.Environment'
    assert var_5.block_start_string == '{%'
    assert var_5.block_end_string == '%}'
    assert var_5.variable_start_string == '{{'
    assert var_5.variable_end_string == '}}'
    assert var_5.comment_start_string == '{#'
    assert var_5.comment_end_string == '#}'
    assert var_5.line_statement_prefix is None
    assert var_5.line_comment_prefix is None
    assert var_5.trim_blocks is False
    assert var_5.lstrip_blocks is False
    assert var_5.newline_sequence == '\n'
    assert var_5.keep_trailing_newline is False
    assert var_5.optimized is True
    assert var_5.finalize is None
    assert var_5.autoescape is False
    assert f'{type(var_5.filters).__module__}.{type(var_5.filters).__qualname__}' == 'builtins.dict'
    assert len(var_5.filters) == 54
    assert f'{type(var_5.tests).__module__}.{type(var_5.tests).__qualname__}' == 'builtins.dict'
    assert len(var_5.tests) == 39
    assert f'{type(var_5.globals).__module__}.{type(var_5.globals).__qualname__}' == 'builtins.dict'
    assert len(var_5.globals) == 6
    assert var_5.loader is None
    assert f'{type(var_5.cache).__module__}.{type(var_5.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_5.cache) == 0
    assert var_5.bytecode_cache is None
    assert var_5.auto_reload is True
    assert var_5.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_5.extensions == {}
    assert var_5.is_async is False
    assert module_2.BLOCK_END_STRING == '%}'
    assert module_2.BLOCK_START_STRING == '{%'
    assert module_2.COMMENT_END_STRING == '#}'
    assert module_2.COMMENT_START_STRING == '{#'
    assert f'{type(module_2.DEFAULT_FILTERS).__module__}.{type(module_2.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_FILTERS) == 54
    assert f'{type(module_2.DEFAULT_NAMESPACE).__module__}.{type(module_2.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_NAMESPACE) == 6
    assert module_2.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_2.DEFAULT_TESTS).__module__}.{type(module_2.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_2.DEFAULT_TESTS) == 39
    assert module_2.KEEP_TRAILING_NEWLINE is False
    assert module_2.LINE_COMMENT_PREFIX is None
    assert module_2.LINE_STATEMENT_PREFIX is None
    assert module_2.LSTRIP_BLOCKS is False
    assert module_2.NEWLINE_SEQUENCE == '\n'
    assert module_2.TRIM_BLOCKS is False
    assert module_2.VARIABLE_END_STRING == '}}'
    assert module_2.VARIABLE_START_STRING == '{{'
    assert f'{type(module_2.missing).__module__}.{type(module_2.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_2.Environment.sandboxed is False
    assert module_2.Environment.overlayed is False
    assert module_2.Environment.linked_to is None
    assert module_2.Environment.shared is False
    assert f'{type(module_2.Environment.lexer).__module__}.{type(module_2.Environment.lexer).__qualname__}' == 'builtins.property'
    var_6 = False
    var_7 = module_0.is_copy_only_path(var_4, var_3)
    assert var_7 is False
    module_0.generate_file(var_0, var_0, var_4, var_5, var_6)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'O+6'
    var_1 = '_Uk<uAiGDH\tASKX_e'
    var_2 = {var_1: var_0}
    var_3 = module_0.is_copy_only_path(var_0, var_2)
    assert var_3 is False
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
    module_0.generate_context(var_3)