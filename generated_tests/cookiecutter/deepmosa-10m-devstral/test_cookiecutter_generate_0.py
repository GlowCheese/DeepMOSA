# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import cookiecutter.utils as module_2
import codecs as module_3
import jinja2.environment as module_4

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
    var_0 = 'c'
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
    var_4 = ',(B9Ep'
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
    var_0 = 'B^\x0bw2@\n{69~/'
    var_1 = {var_0: var_0}
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
    var_1 = 'lhZ#5qCr\x0cpq'
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

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = ','
    module_0.render_and_create_dir(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = {}
    var_4 = False
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'fl}=ag'
    var_1 = True
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
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
    var_4 = None
    module_0.generate_files(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = True
    var_2 = 'o\\\tw EWZQ'
    var_3 = {var_2: var_0, var_2: var_2, var_2: var_2, var_2: var_1, var_2: var_0}
    var_4 = module_2.create_env_with_context(var_3)
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
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'cookiecutter.utils'
    assert module_2.logger.level == 0
    assert f'{type(module_2.logger.parent).__module__}.{type(module_2.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.propagate is True
    assert module_2.logger.handlers == []
    assert module_2.logger.disabled is False
    assert f'{type(module_2.logger.manager).__module__}.{type(module_2.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = 'C'
    module_0.render_and_create_dir(var_5, var_3, var_2, var_4)

def test_case_11():
    var_0 = 'choices'
    var_1 = 'c'
    var_2 = [var_0, var_1, var_1]
    var_3 = {var_0: var_2}
    var_4 = module_3.make_identity_dict(var_2)
    assert module_3.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_3.BOM_LE == b'\xff\xfe'
    assert module_3.BOM_UTF16_LE == b'\xff\xfe'
    assert module_3.BOM_BE == b'\xfe\xff'
    assert module_3.BOM_UTF16_BE == b'\xfe\xff'
    assert module_3.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_3.BOM == b'\xff\xfe'
    assert module_3.BOM_UTF16 == b'\xff\xfe'
    assert module_3.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_3.BOM32_LE == b'\xff\xfe'
    assert module_3.BOM32_BE == b'\xfe\xff'
    assert module_3.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM64_BE == b'\x00\x00\xfe\xff'
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

def test_case_12():
    var_0 = None
    var_1 = '%FI5D'
    var_2 = True
    var_3 = '[F!Ao.Q!Z4Ex'
    var_4 = 'g{'
    var_5 = {var_4: var_0}
    var_6 = {var_3: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6, in_dictionary_variable=var_2)
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
    var_8 = {var_1: var_0, var_3: var_1, var_3: var_3, var_3: var_2, var_1: var_0}
    var_9 = module_0.apply_overwrites_to_context(var_8, var_8)
    var_10 = module_2.create_env_with_context(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_10.block_start_string == '{%'
    assert var_10.block_end_string == '%}'
    assert var_10.variable_start_string == '{{'
    assert var_10.variable_end_string == '}}'
    assert var_10.comment_start_string == '{#'
    assert var_10.comment_end_string == '#}'
    assert var_10.line_statement_prefix is None
    assert var_10.line_comment_prefix is None
    assert var_10.trim_blocks is False
    assert var_10.lstrip_blocks is False
    assert var_10.newline_sequence == '\n'
    assert var_10.keep_trailing_newline is True
    assert var_10.optimized is True
    assert var_10.finalize is None
    assert var_10.autoescape is False
    assert f'{type(var_10.filters).__module__}.{type(var_10.filters).__qualname__}' == 'builtins.dict'
    assert len(var_10.filters) == 56
    assert f'{type(var_10.tests).__module__}.{type(var_10.tests).__qualname__}' == 'builtins.dict'
    assert len(var_10.tests) == 39
    assert f'{type(var_10.globals).__module__}.{type(var_10.globals).__qualname__}' == 'builtins.dict'
    assert len(var_10.globals) == 8
    assert var_10.loader is None
    assert f'{type(var_10.cache).__module__}.{type(var_10.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_10.cache) == 0
    assert var_10.bytecode_cache is None
    assert var_10.auto_reload is True
    assert var_10.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_10.datetime_format == '%Y-%m-%d'
    assert f'{type(var_10.extensions).__module__}.{type(var_10.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_10.extensions) == 5
    assert var_10.is_async is False
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'cookiecutter.utils'
    assert module_2.logger.level == 0
    assert f'{type(module_2.logger.parent).__module__}.{type(module_2.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.propagate is True
    assert module_2.logger.handlers == []
    assert module_2.logger.disabled is False
    assert f'{type(module_2.logger.manager).__module__}.{type(module_2.logger.manager).__qualname__}' == 'logging.Manager'

def test_case_13():
    var_0 = 'choices'
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = {var_0: var_1, var_0: var_1}
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

def test_case_14():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_3.make_identity_dict(var_1)
    assert module_3.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_3.BOM_LE == b'\xff\xfe'
    assert module_3.BOM_UTF16_LE == b'\xff\xfe'
    assert module_3.BOM_BE == b'\xfe\xff'
    assert module_3.BOM_UTF16_BE == b'\xfe\xff'
    assert module_3.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_3.BOM == b'\xff\xfe'
    assert module_3.BOM_UTF16 == b'\xff\xfe'
    assert module_3.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_3.BOM32_LE == b'\xff\xfe'
    assert module_3.BOM32_BE == b'\xfe\xff'
    assert module_3.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = {var_0: var_2, var_0: var_2}
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
    var_0 = 'vFriable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

def test_case_16():
    var_0 = 'choices'
    var_1 = 'c'
    var_2 = [var_0, var_1, var_1]
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_3, var_1: var_3}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_4)

def test_case_17():
    var_0 = 'choices'
    var_1 = 'c'
    var_2 = [var_0, var_1, var_1]
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = [var_0]
    var_5 = {var_0: var_4}
    var_6 = None
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_3, in_dictionary_variable=var_6)

def test_case_18():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = {var_0: var_2}
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_3, var_7, in_dictionary_variable=var_8)
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
    with pytest.raises(KeyError):
        var_10 = var_3['key']

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_3.make_identity_dict(var_1)
    assert module_3.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_3.BOM_LE == b'\xff\xfe'
    assert module_3.BOM_UTF16_LE == b'\xff\xfe'
    assert module_3.BOM_BE == b'\xfe\xff'
    assert module_3.BOM_UTF16_BE == b'\xfe\xff'
    assert module_3.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_3.BOM == b'\xff\xfe'
    assert module_3.BOM_UTF16 == b'\xff\xfe'
    assert module_3.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_3.BOM32_LE == b'\xff\xfe'
    assert module_3.BOM32_BE == b'\xfe\xff'
    assert module_3.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_3.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = {var_0: var_2, var_0: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_3, var_2)
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
    var_5 = module_0.apply_overwrites_to_context(var_3, var_3)
    var_2.write(var_5)

def test_case_20():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_4.Environment()
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
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_1, var_0, var_1, var_2)

def test_case_21():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = '*.md'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'README.md'
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

def test_case_22():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'README.md'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
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
    var_7 = 'main.py'
    var_8 = module_0.is_copy_only_path(var_7, var_4)
    assert var_8 is False

def test_case_23():
    var_0 = True
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = '/tmp'
    var_4 = module_4.Environment()
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
    assert module_4.BLOCK_END_STRING == '%}'
    assert module_4.BLOCK_START_STRING == '{%'
    assert module_4.COMMENT_END_STRING == '#}'
    assert module_4.COMMENT_START_STRING == '{#'
    assert f'{type(module_4.DEFAULT_FILTERS).__module__}.{type(module_4.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_FILTERS) == 54
    assert f'{type(module_4.DEFAULT_NAMESPACE).__module__}.{type(module_4.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_NAMESPACE) == 6
    assert module_4.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_4.DEFAULT_TESTS).__module__}.{type(module_4.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_4.DEFAULT_TESTS) == 39
    assert module_4.KEEP_TRAILING_NEWLINE is False
    assert module_4.LINE_COMMENT_PREFIX is None
    assert module_4.LINE_STATEMENT_PREFIX is None
    assert module_4.LSTRIP_BLOCKS is False
    assert module_4.NEWLINE_SEQUENCE == '\n'
    assert module_4.TRIM_BLOCKS is False
    assert module_4.VARIABLE_END_STRING == '}}'
    assert module_4.VARIABLE_START_STRING == '{{'
    assert f'{type(module_4.missing).__module__}.{type(module_4.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_4.Environment.sandboxed is False
    assert module_4.Environment.overlayed is False
    assert module_4.Environment.linked_to is None
    assert module_4.Environment.shared is False
    assert f'{type(module_4.Environment.lexer).__module__}.{type(module_4.Environment.lexer).__qualname__}' == 'builtins.property'
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
def test_case_24():
    var_0 = False
    var_1 = None
    module_0.generate_context(var_0, extra_context=var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = -1313
    var_1 = None
    module_0.generate_context(var_0, var_1)