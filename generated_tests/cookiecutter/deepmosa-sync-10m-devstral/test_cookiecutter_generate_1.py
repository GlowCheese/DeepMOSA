# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import codecs as module_2
import re as module_3
import json.encoder as module_4
import jinja2.environment as module_5
import jinja2.loaders as module_6
import binaryornot.check as module_7

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
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'readme.txt'
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
    module_0.is_copy_only_path(var_7, var_8)

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
    var_0 = 'AA;t6xmx'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_1, var_1, var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '2L'
    module_0.generate_files(var_0, accept_hooks=var_0)

def test_case_7():
    pass

@pytest.mark.xfail(strict=True)
def test_case_8():
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
def test_case_9():
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

def test_case_10():
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

def test_case_11():
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

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'multi_var'
    var_1 = [var_0, var_0]
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
    module_0.render_and_create_dir(var_0, var_3, var_3, var_0, var_3)

def test_case_13():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '/path/to/repo'
    var_1 = 'post_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)

def test_case_15():
    var_0 = 'variable'
    var_1 = 'a'
    var_2 = 'c'
    var_3 = [var_1, var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'd'
    var_6 = [var_5, var_5]
    var_7 = {var_0: var_6}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_7)

def test_case_16():
    var_0 = 'bool_var'
    var_1 = True
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
    var_0 = 'w)'
    var_1 = 'c'
    var_2 = [var_0, var_0, var_1]
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_0}
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
    var_6 = module_3.purge()
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
    var_7 = module_2.iterencode(var_5, var_6)
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
    var_8 = bool(var_3 == {'multi_var': ['a', 'c']})

def test_case_18():
    var_0 = 'w)'
    var_1 = 'c'
    var_2 = [var_0, var_0, var_1]
    var_3 = {var_1: var_2}
    var_4 = module_4.py_encode_basestring_ascii(var_0)
    assert var_4 == '"w)"'
    assert f'{type(module_4.ESCAPE).__module__}.{type(module_4.ESCAPE).__qualname__}' == 're.Pattern'
    assert f'{type(module_4.ESCAPE_ASCII).__module__}.{type(module_4.ESCAPE_ASCII).__qualname__}' == 're.Pattern'
    assert f'{type(module_4.HAS_UTF8).__module__}.{type(module_4.HAS_UTF8).__qualname__}' == 're.Pattern'
    assert module_4.ESCAPE_DCT == {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007', '\x0b': '\\u000b', '\x0e': '\\u000e', '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017', '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f'}
    assert module_4.i == 31
    assert module_4.INFINITY == pytest.approx(1e309, abs=0.01, rel=0.01)
    var_5 = {var_1: var_4}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'multi_var'
    var_1 = 'w)'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5, in_dictionary_variable=var_1)
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
    var_7 = [var_1, var_3]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = module_3.purge()
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
    module_0.render_and_create_dir(var_0, var_9, var_9, var_10, var_9)

def test_case_20():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_5.Environment()
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
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_0, var_1, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'existing_dir'
    var_1 = {}
    var_2 = 't]'
    var_3 = module_5.Environment()
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
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.render_and_create_dir(var_0, var_1, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = True
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_5.Environment()
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
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_4 = module_0.render_and_create_dir(var_2, var_1, var_2, var_3, var_0)
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
    var_5 = {}
    module_0.generate_context(extra_context=var_5)

def test_case_23():
    var_0 = '=+6dGY5#'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    var_4 = {var_0: var_2, var_0: var_2}
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

def test_case_24():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'i{5y58*GSP'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
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
    var_9 = 'src/main.py'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is False

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'binary.jpg'
    var_1 = {}
    var_2 = module_5.Environment(loader=var_0)
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
    assert var_2.loader == 'binary.jpg'
    assert f'{type(var_2.cache).__module__}.{type(var_2.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_2.cache) == 0
    assert var_2.bytecode_cache is None
    assert var_2.auto_reload is True
    assert var_2.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_2.extensions == {}
    assert var_2.is_async is False
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    module_0.generate_file(var_0, var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = '/tmp/projet'
    var_1 = False
    var_2 = {var_1: var_1}
    var_3 = {var_0: var_2}
    var_4 = module_5.Environment()
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
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_5 = True
    module_0.generate_file(var_0, var_0, var_3, var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'cookQecuter'
    var_1 = {}
    var_2 = '.'
    var_3 = module_6.FileSystemLoader(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_3.searchpath == ['.']
    assert var_3.encoding == 'utf-8'
    assert var_3.followlinks is False
    var_4 = False
    var_5 = {var_2: var_2, var_2: var_1, var_0: var_4, var_2: var_1}
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5)
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
    var_7 = module_5.Environment(loader=var_3)
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
    assert f'{type(var_7.loader).__module__}.{type(var_7.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_7.cache).__module__}.{type(var_7.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_7.cache) == 0
    assert var_7.bytecode_cache is None
    assert var_7.auto_reload is True
    assert var_7.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_7.extensions == {}
    assert var_7.is_async is False
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_8 = module_0.generate_file(var_2, var_2, var_1, var_7, var_4)
    module_0.generate_context()

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = '|m!X-(ZS[Ww4O/{O'
    var_1 = '_new_ines'
    var_2 = {}
    var_3 = '.'
    var_4 = module_6.FileSystemLoader(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert var_4.searchpath == ['.']
    assert var_4.encoding == 'utf-8'
    assert var_4.followlinks is False
    var_5 = False
    var_6 = {var_1: var_1, var_3: var_0, var_0: var_5, var_1: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
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
    var_8 = module_5.Environment(loader=var_4)
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
    assert f'{type(var_8.loader).__module__}.{type(var_8.loader).__qualname__}' == 'jinja2.loaders.FileSystemLoader'
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.extensions == {}
    assert var_8.is_async is False
    assert module_5.BLOCK_END_STRING == '%}'
    assert module_5.BLOCK_START_STRING == '{%'
    assert module_5.COMMENT_END_STRING == '#}'
    assert module_5.COMMENT_START_STRING == '{#'
    assert f'{type(module_5.DEFAULT_FILTERS).__module__}.{type(module_5.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_FILTERS) == 54
    assert f'{type(module_5.DEFAULT_NAMESPACE).__module__}.{type(module_5.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_NAMESPACE) == 6
    assert module_5.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_5.DEFAULT_TESTS).__module__}.{type(module_5.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_5.DEFAULT_TESTS) == 39
    assert module_5.KEEP_TRAILING_NEWLINE is False
    assert module_5.LINE_COMMENT_PREFIX is None
    assert module_5.LINE_STATEMENT_PREFIX is None
    assert module_5.LSTRIP_BLOCKS is False
    assert module_5.NEWLINE_SEQUENCE == '\n'
    assert module_5.TRIM_BLOCKS is False
    assert module_5.VARIABLE_END_STRING == '}}'
    assert module_5.VARIABLE_START_STRING == '{{'
    assert f'{type(module_5.missing).__module__}.{type(module_5.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_5.Environment.sandboxed is False
    assert module_5.Environment.overlayed is False
    assert module_5.Environment.linked_to is None
    assert module_5.Environment.shared is False
    assert f'{type(module_5.Environment.lexer).__module__}.{type(module_5.Environment.lexer).__qualname__}' == 'builtins.property'
    var_9 = module_7.is_binary(var_0)
    assert f'{type(module_7.logger).__module__}.{type(module_7.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_7.logger.filters == []
    assert module_7.logger.name == 'binaryornot.check'
    assert module_7.logger.level == 0
    assert f'{type(module_7.logger.parent).__module__}.{type(module_7.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_7.logger.propagate is True
    assert module_7.logger.handlers == []
    assert module_7.logger.disabled is False
    assert f'{type(module_7.logger.manager).__module__}.{type(module_7.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context(var_9, var_9)