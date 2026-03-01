# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import jinja2.environment as module_2
import pathlib as module_3

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
    var_0 = None
    module_0.is_copy_only_path(var_0, var_0)

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
    var_0 = 'b'
    var_1 = [var_0, var_0, var_0]
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

def test_case_6():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

def test_case_7():
    var_0 = 'key'
    var_1 = 'old'
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
    var_4 = bool(var_2 == {'key': 'new'})

def test_case_8():
    var_0 = True
    var_1 = 'o\\\tw EWZQ'
    var_2 = '[F!Ao.Q!Z4Ex'
    var_3 = {var_2: var_1, var_2: var_2, var_2: var_0}
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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = "w}v'EJ&eF.2'/RBcr%"
    var_1 = None
    var_2 = True
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1, var_2)

def test_case_10():
    var_0 = 'original'
    var_1 = {var_0: var_0}
    var_2 = 'overwrite'
    var_3 = {var_2: var_0}
    var_4 = module_0.apply_overwrites_to_context(var_1, var_3)
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
    var_5 = bool(var_3 == {'var': 'overwrite'})

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'dict_var'
    var_1 = 'new'
    var_2 = {var_1: var_0}
    var_3 = {var_0: var_2}
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
    module_0.generate_context(var_4, var_4, var_4)

def test_case_12():
    var_0 = 'test_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_5, var_4, in_dictionary_variable=var_6)
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
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_4)

def test_case_13():
    var_0 = 'choice_var'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_2, var_1, var_2]
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

def test_case_14():
    var_0 = 'list_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_9)

def test_case_15():
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_7)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    module_0.generate_files(var_5, var_4, var_6)

def test_case_18():
    var_0 = ','
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

def test_case_19():
    var_0 = 'b'
    var_1 = [var_0, var_0, var_0]
    var_2 = {var_0: var_1}
    var_3 = True
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

def test_case_20():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_2.Environment()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_2.keep_trailing_newline is False
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 54
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 6
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
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
        module_0.render_and_create_dir(var_1, var_0, var_1, var_2)

def test_case_21():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_3.Path(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pathlib.PosixPath'
    assert module_3.EINVAL == 22
    assert module_3.ENOENT == 2
    assert module_3.ENOTDIR == 20
    assert module_3.EBADF == 9
    assert module_3.ELOOP == 40
    var_6 = module_2.Environment()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_6.keep_trailing_newline is False
    assert var_6.optimized is True
    assert var_6.finalize is None
    assert var_6.autoescape is False
    assert f'{type(var_6.filters).__module__}.{type(var_6.filters).__qualname__}' == 'builtins.dict'
    assert len(var_6.filters) == 54
    assert f'{type(var_6.tests).__module__}.{type(var_6.tests).__qualname__}' == 'builtins.dict'
    assert len(var_6.tests) == 39
    assert f'{type(var_6.globals).__module__}.{type(var_6.globals).__qualname__}' == 'builtins.dict'
    assert len(var_6.globals) == 6
    assert var_6.loader is None
    assert f'{type(var_6.cache).__module__}.{type(var_6.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_6.cache) == 0
    assert var_6.bytecode_cache is None
    assert var_6.auto_reload is True
    assert var_6.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_6.extensions == {}
    assert var_6.is_async is False
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
    var_7 = True
    var_8 = [var_5, var_0]
    var_9 = {}
    var_10 = module_3.Path(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pathlib.PosixPath'
    var_11 = var_10.mkdir(parents=var_7, exist_ok=var_7)
    var_12 = module_0.render_and_create_dir(var_0, var_1, var_5, var_6, var_7)
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
    var_13 = var_12[1]
    assert var_13 is False

def test_case_22():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_1: var_1}
    var_3 = {var_0: var_2}
    var_4 = 'not_a_dict'
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
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
    var_7 = bool(var_2 == {'key': {'subkey': 'value'}})

def test_case_23():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False
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

def test_case_24():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False
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
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'temp*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'temp_file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
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
    var_0 = 'text.txt'
    var_1 = {}
    var_2 = module_2.Environment()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_2.keep_trailing_newline is False
    assert var_2.optimized is True
    assert var_2.finalize is None
    assert var_2.autoescape is False
    assert f'{type(var_2.filters).__module__}.{type(var_2.filters).__qualname__}' == 'builtins.dict'
    assert len(var_2.filters) == 54
    assert f'{type(var_2.tests).__module__}.{type(var_2.tests).__qualname__}' == 'builtins.dict'
    assert len(var_2.tests) == 39
    assert f'{type(var_2.globals).__module__}.{type(var_2.globals).__qualname__}' == 'builtins.dict'
    assert len(var_2.globals) == 6
    assert var_2.loader is None
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
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
    var_3 = False
    module_0.generate_file(var_0, var_0, var_1, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'q2)6}lT@_#Z5A24)&'
    var_1 = 'cookiecutter'
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = {var_1: var_2, var_0: var_2, var_0: var_2, var_0: var_2}
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
    module_0.generate_file(var_0, var_0, var_3, var_4, var_4)