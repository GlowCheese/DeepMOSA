# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import codecs as module_2
import re as module_3
import enum as module_4
import jinja2.environment as module_5

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
    var_0 = '.xt'
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

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'oNbBk K}NFXF'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_1}
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
    var_4 = '1zf*T Ao1\r\x0c'
    module_0.render_and_create_dir(var_4, var_2, var_4, var_4, var_3)

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
    var_6 = {var_5: var_3, var_1: var_5, var_1: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6, in_dictionary_variable=var_2)
    module_0.generate_context(var_1, extra_context=var_2)

def test_case_13():
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
    var_4 = 's<ZI\r(^\tI'
    var_5 = {var_3: var_2, var_0: var_1, var_4: var_1, var_0: var_0}
    var_6 = {var_4: var_4, var_3: var_1, var_3: var_4, var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_5, var_6)

def test_case_14():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
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
    var_9 = module_0.apply_overwrites_to_context(var_5, var_5)
    var_10 = 'docs/index.md'
    var_11 = module_0.is_copy_only_path(var_10, var_6)
    assert var_11 is True
    var_12 = 'src/main.py'
    var_13 = module_0.is_copy_only_path(var_12, var_6)
    assert var_13 is False
    var_14 = {}
    var_15 = {var_0: var_14}
    var_16 = module_0.is_copy_only_path(var_7, var_15)
    assert var_16 is False
    var_17 = module_0.is_copy_only_path(var_7, var_14)
    assert var_17 is False
    var_18 = module_0.apply_overwrites_to_context(var_6, var_6)

def test_case_15():
    var_0 = '.xt'
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

def test_case_16():
    var_0 = 'var1'
    var_1 = 'value2'
    var_2 = module_3.purge()
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
    var_3 = [var_1, var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'va1'
    var_1 = '6D[q$ZIKJ,U3d3%p>5q'
    var_2 = 'value2'
    var_3 = module_3.purge()
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
    var_4 = 'choice1'
    var_5 = 'choice3'
    var_6 = [var_4, var_2, var_0, var_5]
    var_7 = {var_0: var_6}
    var_8 = {var_0: var_0}
    var_9 = module_0.apply_overwrites_to_context(var_7, var_8)
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
    var_10 = '%-oQ'
    module_0.generate_file(var_10, var_1, var_3, var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'A~HJcA\n'
    var_1 = module_4._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_5.Environment(loader=var_1)
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
    assert f'{type(var_2.loader).__module__}.{type(var_2.loader).__qualname__}' == 'enum._EnumDict'
    assert len(var_2.loader) == 0
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
def test_case_19():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_5.Environment(loader=var_1)
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
    assert var_2.loader == {}
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
    module_0.generate_file(var_0, var_0, var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = {}
    var_1 = module_5.Environment(loader=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
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
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert var_1.loader == {}
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
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
    var_2 = 'N96:L\x0cF9XX'
    var_3 = None
    module_0.render_and_create_dir(var_2, var_0, var_2, var_1, var_3)

def test_case_21():
    var_0 = "{\na^f6b}S#'?9PZ'\rc"
    var_1 = 't~est'
    var_2 = {var_0: var_1}
    var_3 = module_5.Environment(loader=var_1)
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
    assert var_3.loader == 't~est'
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
    var_4 = '.'
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_4, var_2, var_4, var_3)