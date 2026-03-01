# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import codecs as module_2
import cookiecutter.utils as module_3
import binaryornot.check as module_4
import re as module_5

def test_case_0():
    var_0 = '7+m0m%`{8tgr@]be\x0bE'
    var_1 = 'L|~8ynQO\ruipIvM\x0b[O]'
    var_2 = "CF>e3SOJU\\^g;2o Y'"
    var_3 = '@hi%}q{Fa'
    var_4 = {var_1: var_1, var_2: var_1, var_3: var_2, var_3: var_0}
    var_5 = module_0.is_copy_only_path(var_0, var_4)
    assert var_5 is False
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
    var_0 = '@V^dSU\r'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1)

def test_case_5():
    var_0 = None
    var_1 = "oY>^A!x7w'"
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_1, var_0)

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

def test_case_9():
    var_0 = 'oNbBk K}NFXF'
    var_1 = None
    var_2 = 'cr;q\nVY3dA('
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
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5)
    var_7 = True
    var_8 = module_0.apply_overwrites_to_context(var_3, var_5, in_dictionary_variable=var_7)

def test_case_10():
    var_0 = '\tRh)&cFMN**)'
    var_1 = 'cr;q\nVY3dA('
    var_2 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = module_0.is_copy_only_path(var_2, var_2)
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
    var_4 = {var_0: var_0, var_1: var_2, var_0: var_0}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_4)

def test_case_11():
    var_0 = '\tRh)&cFMN**)'
    var_1 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = module_0.is_copy_only_path(var_1, var_1)
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
    var_3 = '%,'
    var_4 = {var_3: var_2, var_0: var_1, var_0: var_1, var_0: var_1}
    var_5 = {var_0: var_0, var_3: var_1, var_3: var_0, var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'oNbBk K}NFXF'
    var_1 = 'cr;q\nVY3dA('
    var_2 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_3 = module_0.is_copy_only_path(var_1, var_2)
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
    var_4 = {var_0: var_0, var_1: var_2, var_0: var_0}
    var_5 = '\nqDbS?\rVC'
    var_6 = 'Q'
    var_7 = {var_5: var_3, var_6: var_5, var_1: var_2}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_2)
    var_2.iterencode(var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = -3663
    module_0.generate_context(var_0)

def test_case_14():
    var_0 = ''
    var_1 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = var_1.__dir__()
    var_3 = '%,'
    var_4 = "4Oomvs/$[qk'"
    var_5 = 'EU'
    var_6 = {var_3: var_2, var_5: var_1, var_4: var_1, var_5: var_0}
    var_7 = {var_4: var_4, var_3: var_1, var_4: var_4, var_5: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_6, var_7)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'T4jAd3^HzY>*!X'
    var_1 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
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
    module_0.render_and_create_dir(var_0, var_1, var_0, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'K'
    var_1 = {}
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

def test_case_17():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = 'images/**'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'readme.txt'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
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
    var_10 = 'docs/index.md'
    var_11 = module_0.is_copy_only_path(var_10, var_7)
    assert var_11 is True
    var_12 = 'images/photo.jpg'
    var_13 = module_0.is_copy_only_path(var_12, var_7)
    assert var_13 is True
    var_14 = 'images/subfolder/photo.png'
    var_15 = module_0.is_copy_only_path(var_14, var_7)
    assert var_15 is True
    var_16 = 'src/main.py'
    var_17 = module_0.is_copy_only_path(var_16, var_7)
    assert var_17 is False
    var_18 = 'config.yaml'
    var_19 = module_0.is_copy_only_path(var_18, var_7)
    assert var_19 is False
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = 'any/path'
    var_23 = module_0.is_copy_only_path(var_22, var_21)
    assert var_23 is False
    var_24 = []
    var_25 = {var_1: var_24}
    var_26 = {var_0: var_25}
    var_27 = module_0.is_copy_only_path(var_22, var_26)
    assert var_27 is False
    var_28 = 'exact_file.md'
    var_29 = {var_0: var_6}
    var_30 = module_0.is_copy_only_path(var_28, var_29)
    assert var_30 is False
    var_31 = 'other_file.md'
    var_32 = module_0.is_copy_only_path(var_31, var_29)
    assert var_32 is False
    var_33 = 'data/2023.csv'
    var_34 = module_0.is_copy_only_path(var_33, var_26)
    assert var_34 is False
    var_35 = 'logs/app.log'
    var_36 = module_0.is_copy_only_path(var_35, var_29)
    assert var_36 is False
    var_37 = 'data/backup/old.csv'
    var_38 = module_0.is_copy_only_path(var_37, var_26)
    assert var_38 is False
    var_39 = 'assets/**/*.png'
    var_40 = 'templates/*.html'
    var_41 = [var_39, var_40]
    var_42 = {var_1: var_41}
    var_43 = {var_0: var_42}
    var_44 = 'assets/images/icon.png'
    var_45 = module_0.is_copy_only_path(var_44, var_43)
    assert var_45 is True
    var_46 = 'templates/index.html'
    var_47 = module_0.is_copy_only_path(var_46, var_43)
    assert var_47 is True
    var_48 = 'assets/readme.txt'
    var_49 = module_0.is_copy_only_path(var_48, var_43)
    assert var_49 is False

def test_case_18():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 'QP'
    var_4 = {var_0: var_2, var_1: var_3}
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
    var_6 = 'existing'
    var_7 = {var_6: var_3}
    var_8 = 'new_var'
    var_9 = 'new_value'
    var_10 = {var_8: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = 'choices'
    var_13 = ''
    var_14 = 'b'
    var_15 = module_4.is_binary(var_0)
    assert var_15 is False
    assert f'{type(module_4.logger).__module__}.{type(module_4.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_4.logger.filters == []
    assert module_4.logger.name == 'binaryornot.check'
    assert module_4.logger.level == 0
    assert f'{type(module_4.logger.parent).__module__}.{type(module_4.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_4.logger.propagate is True
    assert module_4.logger.handlers == []
    assert module_4.logger.disabled is False
    assert f'{type(module_4.logger.manager).__module__}.{type(module_4.logger.manager).__qualname__}' == 'logging.Manager'
    var_16 = {var_12: var_15}
    var_17 = [var_14, var_6]
    var_18 = {var_12: var_17}
    var_19 = module_0.apply_overwrites_to_context(var_16, var_18)
    var_20 = [var_13, var_14, var_13]
    var_21 = {var_12: var_20}
    var_22 = module_0.apply_overwrites_to_context(var_21, var_21)

def test_case_19():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
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
    var_8 = '5=^]i)v<!\x0budq~M'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'new_var'
    var_12 = 'new_value'
    var_13 = {var_11: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_10, var_13)
    var_15 = 'choices'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = 'd'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = {var_15: var_20}
    var_22 = [var_17, var_18]
    var_23 = {var_15: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = [var_16, var_17, var_18]
    var_26 = {var_15: var_25}
    var_27 = [var_17, var_19]
    var_28 = {var_15: var_27}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_26, var_28)

def test_case_20():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'origiHnal'
    var_3 = {var_0: var_2, var_1: var_1}
    var_4 = 'new_name'
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
    var_7 = 'value'
    var_8 = module_5.purge()
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
    var_9 = 'choices'
    var_10 = ''
    var_11 = module_4.is_binary(var_0)
    assert var_11 is False
    assert f'{type(module_4.logger).__module__}.{type(module_4.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_4.logger.filters == []
    assert module_4.logger.name == 'binaryornot.check'
    assert module_4.logger.level == 0
    assert f'{type(module_4.logger.parent).__module__}.{type(module_4.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_4.logger.propagate is True
    assert module_4.logger.handlers == []
    assert module_4.logger.disabled is False
    assert f'{type(module_4.logger.manager).__module__}.{type(module_4.logger.manager).__qualname__}' == 'logging.Manager'
    var_12 = {var_9: var_11}
    var_13 = [var_7, var_7]
    var_14 = {var_9: var_13}
    var_15 = module_0.apply_overwrites_to_context(var_12, var_14)
    var_16 = [var_0, var_10, var_2]
    var_17 = {var_9: var_16}
    var_18 = {var_9: var_2}
    var_19 = module_0.apply_overwrites_to_context(var_17, var_18)

def test_case_21():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = 'QP'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = module_0.is_copy_only_path(var_2, var_4)
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
    var_7 = {var_0: var_5}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = 'existing'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'new_var'
    var_13 = 'new_value'
    var_14 = {var_12: var_13}
    var_15 = module_0.apply_overwrites_to_context(var_11, var_14)
    var_16 = 'choices'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = module_4.is_binary(var_0)
    assert var_19 is False
    assert f'{type(module_4.logger).__module__}.{type(module_4.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_4.logger.filters == []
    assert module_4.logger.name == 'binaryornot.check'
    assert module_4.logger.level == 0
    assert f'{type(module_4.logger.parent).__module__}.{type(module_4.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_4.logger.propagate is True
    assert module_4.logger.handlers == []
    assert module_4.logger.disabled is False
    assert f'{type(module_4.logger.manager).__module__}.{type(module_4.logger.manager).__qualname__}' == 'logging.Manager'
    var_20 = {var_16: var_19}
    var_21 = [var_17, var_18]
    var_22 = {var_16: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = {var_16: var_18}
    var_25 = module_0.apply_overwrites_to_context(var_20, var_24, in_dictionary_variable=var_17)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'R7)&FyN**)'
    var_1 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
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
    module_0.generate_file(var_0, var_0, var_1, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 'Rh)&FMN**)'
    var_1 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = module_0.is_copy_only_path(var_1, var_1)
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
    var_3 = 'Z'
    var_4 = "4Oovs/IJ$[qk'"
    var_5 = '/'
    var_6 = {var_3: var_2, var_0: var_1, var_4: var_1, var_5: var_4}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
    var_8 = module_3.create_env_with_context(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
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
    assert var_8.keep_trailing_newline is True
    assert var_8.optimized is True
    assert var_8.finalize is None
    assert var_8.autoescape is False
    assert f'{type(var_8.filters).__module__}.{type(var_8.filters).__qualname__}' == 'builtins.dict'
    assert len(var_8.filters) == 56
    assert f'{type(var_8.tests).__module__}.{type(var_8.tests).__qualname__}' == 'builtins.dict'
    assert len(var_8.tests) == 39
    assert f'{type(var_8.globals).__module__}.{type(var_8.globals).__qualname__}' == 'builtins.dict'
    assert len(var_8.globals) == 8
    assert var_8.loader is None
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.datetime_format == '%Y-%m-%d'
    assert f'{type(var_8.extensions).__module__}.{type(var_8.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_8.extensions) == 5
    assert var_8.is_async is False
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
    var_9 = module_0.generate_file(var_3, var_5, var_1, var_8, var_1)
    var_10 = 'tFJ-A9\n[uV'
    module_0.generate_file(var_5, var_10, var_1, var_8)