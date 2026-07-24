# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.utils as module_1
import cookiecutter.exceptions as module_2
import codecs as module_3

def test_case_0():
    var_0 = 'ydwmJdR*HLkp\x0bg<z'
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
    var_3 = module_1.create_env_with_context(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert var_3.keep_trailing_newline is True
    assert var_3.optimized is True
    assert var_3.finalize is None
    assert var_3.autoescape is False
    assert f'{type(var_3.filters).__module__}.{type(var_3.filters).__qualname__}' == 'builtins.dict'
    assert len(var_3.filters) == 56
    assert f'{type(var_3.tests).__module__}.{type(var_3.tests).__qualname__}' == 'builtins.dict'
    assert len(var_3.tests) == 39
    assert f'{type(var_3.globals).__module__}.{type(var_3.globals).__qualname__}' == 'builtins.dict'
    assert len(var_3.globals) == 8
    assert var_3.loader is None
    assert f'{type(var_3.cache).__module__}.{type(var_3.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_3.cache) == 0
    assert var_3.bytecode_cache is None
    assert var_3.auto_reload is True
    assert var_3.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_3.datetime_format == '%Y-%m-%d'
    assert f'{type(var_3.extensions).__module__}.{type(var_3.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_3.extensions) == 5
    assert var_3.is_async is False
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'cookiecutter.utils'
    assert module_1.logger.level == 0
    assert f'{type(module_1.logger.parent).__module__}.{type(module_1.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.propagate is True
    assert module_1.logger.handlers == []
    assert module_1.logger.disabled is False
    assert f'{type(module_1.logger.manager).__module__}.{type(module_1.logger.manager).__qualname__}' == 'logging.Manager'
    with pytest.raises(module_2.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.is_copy_only_path(var_0, var_0)

def test_case_2():
    var_0 = 'oNbBk K}NFXF'
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

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.generate_context()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'P_(E\rY*ePh'
    module_0.render_and_create_dir(var_0, var_0, var_0, var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(module_2.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '2L'
    module_0.generate_files(var_0, accept_hooks=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'oNbBk K}NFXF'
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
    module_0.generate_files(var_0, var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'oNbBk K}NFXF'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = {}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_1)
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
    module_0.apply_overwrites_to_context(var_3, var_3, in_dictionary_variable=var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'oNbBk K}NFXF'
    var_1 = None
    var_2 = 'cr;\nVY3dA('
    var_3 = {var_2: var_1}
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
    var_5 = {var_0: var_0, var_0: var_0}
    var_6 = '`'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_5, var_3, in_dictionary_variable=var_8)
    module_0.apply_overwrites_to_context(var_9, var_5)

def test_case_10():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

def test_case_11():
    var_0 = 'enable%d'
    var_1 = False
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = {var_0: var_0, var_0: var_0}
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
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = 'cr;\nVY3dA('
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
    var_4 = {var_1: var_2}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_2)
    var_6 = module_0.apply_overwrites_to_context(var_4, var_2)
    module_0.generate_files(var_5, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = 'cr;\nVY3dA('
    var_2 = {var_1: var_0}
    var_3 = {var_1: var_0, var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
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
    var_5 = {var_1: var_2}
    var_6 = module_0.apply_overwrites_to_context(var_5, var_3)
    var_7 = module_3.iterencode(var_6, var_0)
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
    module_0.apply_overwrites_to_context(var_7, var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'cr;\nVY3dA('
    var_2 = {var_1: var_0}
    var_3 = {var_1: var_1, var_1: var_1}
    var_4 = module_0.is_copy_only_path(var_1, var_2)
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
    var_5 = module_0.is_copy_only_path(var_0, var_3)
    assert var_5 is False
    var_6 = {var_1: var_5, var_1: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_2)
    var_8 = module_0.apply_overwrites_to_context(var_2, var_6, in_dictionary_variable=var_5)
    var_9 = module_3.make_encoding_map(var_6)
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
    module_0.generate_context(var_4, var_9)

def test_case_15():
    var_0 = 'ydwmJdR*HLkp\x0bg<z'
    var_1 = {}
    var_2 = module_1.create_env_with_context(var_1)
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
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'cookiecutter.utils'
    assert module_1.logger.level == 0
    assert f'{type(module_1.logger.parent).__module__}.{type(module_1.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.propagate is True
    assert module_1.logger.handlers == []
    assert module_1.logger.disabled is False
    assert f'{type(module_1.logger.manager).__module__}.{type(module_1.logger.manager).__qualname__}' == 'logging.Manager'
    with pytest.raises(module_2.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_2)
    var_3 = module_0.generate_context()

def test_case_16():
    var_0 = 'release'
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

def test_case_17():
    var_0 = 'settings'
    var_1 = 'modes'
    var_2 = 'a'
    var_3 = 'd'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_6, var_6, in_dictionary_variable=var_7)
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
    var_9 = var_6['settings']['modes']
    var_10 = bool(var_6['settings']['modes'] == ['c', 'd'])

def test_case_18():
    var_0 = 'mode'
    var_1 = 'release'
    var_2 = [var_0, var_1]
    var_3 = {var_0: var_2}
    var_4 = 'production'
    var_5 = {var_0: var_4}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_5)

def test_case_19():
    var_0 = 'mode'
    var_1 = '!~~<5LMZZR<\\G1&N'
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = None
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
    var_5 = [var_0, var_0]
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_2)

def test_case_20():
    var_0 = 'ydwmJdR*HLkp\x0bg<z'
    var_1 = {}
    var_2 = module_1.create_env_with_context(var_1)
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
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'cookiecutter.utils'
    assert module_1.logger.level == 0
    assert f'{type(module_1.logger.parent).__module__}.{type(module_1.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.propagate is True
    assert module_1.logger.handlers == []
    assert module_1.logger.disabled is False
    assert f'{type(module_1.logger.manager).__module__}.{type(module_1.logger.manager).__qualname__}' == 'logging.Manager'
    with pytest.raises(module_2.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_0, var_2)

def test_case_21():
    var_0 = 'features'
    var_1 = 'auth'
    var_2 = 'log\\ging'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'database'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_7)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'template_dir'
    var_1 = 'pre_gen_project'
    var_2 = 'output_dir'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = False
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)

def test_case_23():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '\rm7bvk?\rc=E'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
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

def test_case_24():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.txt'
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

def test_case_25():
    var_0 = 'src/templates/static/*'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/templates/static/*'
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