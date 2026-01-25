# Check out: https://github.com/GlowCheese/deepmosa
import codecs as module_3
import json.encoder as module_5
import re as module_2

import cookiecutter.exceptions as module_1
import cookiecutter.generate as module_0
import jinja2.environment as module_4
import pytest


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
    module_0.generate_files(var_0)

def test_case_5():
    var_0 = True
    var_1 = ' \x0b\r'
    var_2 = {var_1: var_0}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2, in_dictionary_variable=var_0)
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
    var_0 = 'new_var'
    var_1 = []
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

def test_case_7():
    var_0 = {}
    var_1 = module_0.apply_overwrites_to_context(var_0, var_0)
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

def test_case_8():
    var_0 = 'I'
    var_1 = '%pN[B]\r@}\rM8'
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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = True
    var_1 = None
    module_0.generate_files(var_1, var_0, accept_hooks=var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'zay_<#~hQ.%7,KXd'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'name'
    var_1 = 'original'
    var_2 = {var_0: var_1, var_1: var_0, var_1: var_0}
    var_3 = module_2.purge()
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
    var_4 = {var_0: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_2)
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
    var_6 = 'green'
    var_7 = [var_1, var_6, var_1]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_8, var_4)
    module_0.generate_files(var_3, output_dir=var_5, keep_project_on_failure=var_9)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'Uggl34['
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_2.escape(var_0)
    assert var_2 == 'Uggl34\\['
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
    module_0.apply_overwrites_to_context(var_2, var_1, in_dictionary_variable=var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'iL&]_wW).H"%mTT]`)'
    var_1 = 'kcf'
    var_2 = 'neseed'
    var_3 = {var_1: var_0}
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_4, var_5, in_dictionary_variable=var_6)
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
    var_8 = 'option1'
    module_3.getencoder(var_8)

def test_case_14():
    var_0 = '&/xem>@&/u3v'
    var_1 = {var_0: var_0}
    var_2 = []
    var_3 = {var_0: var_2}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'nsme'
    var_1 = 'original'
    var_2 = module_2.purge()
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
    var_3 = {var_0: var_1}
    var_4 = 'green'
    var_5 = [var_1, var_4, var_1]
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_3)
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
    var_2.__getitem__(var_2, var_3)

def test_case_16():
    var_0 = 'name'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
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
    var_7 = {var_6: var_1}
    var_8 = 'new_var'
    var_9 = {var_8: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_7, var_9)
    var_11 = 'nested'
    var_12 = {var_6: var_9}
    var_13 = {var_11: var_12}
    var_14 = {var_11: var_0}
    var_15 = True
    var_16 = module_0.apply_overwrites_to_context(var_13, var_14, in_dictionary_variable=var_15)
    var_17 = 'choice'
    var_18 = 'option1'
    var_19 = 'option2'
    var_20 = 'option3'
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_17: var_21}
    var_23 = {var_17: var_19}
    var_24 = module_0.apply_overwrites_to_context(var_22, var_23)
    var_25 = [var_18, var_19]
    var_26 = {var_17: var_25}
    var_27 = 'invalid_option'
    var_28 = {var_17: var_27}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_26, var_28)

def test_case_17():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = 'binary/*'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'file.txt'
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
    var_10 = 'docs/readme.md'
    var_11 = module_0.is_copy_only_path(var_10, var_7)
    assert var_11 is True
    var_12 = 'binary/image.png'
    var_13 = module_0.is_copy_only_path(var_12, var_7)
    assert var_13 is True
    var_14 = 'file.py'
    var_15 = module_0.is_copy_only_path(var_14, var_7)
    assert var_15 is False
    var_16 = 'script.sh'
    var_17 = module_0.is_copy_only_path(var_16, var_7)
    assert var_17 is False
    var_18 = 'source/code.txt'
    var_19 = module_0.is_copy_only_path(var_18, var_7)
    assert var_19 is True
    var_20 = []
    var_21 = {var_1: var_20}
    var_22 = {var_0: var_21}
    var_23 = module_0.is_copy_only_path(var_8, var_22)
    assert var_23 is False
    var_24 = module_0.is_copy_only_path(var_10, var_22)
    assert var_24 is False
    var_25 = {}
    var_26 = {var_0: var_25}
    var_27 = module_0.is_copy_only_path(var_8, var_26)
    assert var_27 is False
    var_28 = module_0.is_copy_only_path(var_10, var_26)
    assert var_28 is False
    var_29 = {}
    var_30 = module_0.is_copy_only_path(var_8, var_29)
    assert var_30 is False
    var_31 = '*.min.js'
    var_32 = 'node_modules/**'
    var_33 = '.*'
    var_34 = [var_31, var_32, var_33]
    var_35 = {var_1: var_34}
    var_36 = {var_0: var_35}
    var_37 = 'script.min.js'
    var_38 = module_0.is_copy_only_path(var_37, var_36)
    assert var_38 is True
    var_39 = 'node_modules/package'
    var_40 = module_0.is_copy_only_path(var_39, var_36)
    assert var_40 is True
    var_41 = '.gitignore'
    var_42 = module_0.is_copy_only_path(var_41, var_36)
    assert var_42 is True
    var_43 = 'script.js'
    var_44 = module_0.is_copy_only_path(var_43, var_36)
    assert var_44 is False
    var_45 = 'file?.txt'
    var_46 = [var_45]
    var_47 = {var_1: var_46}
    var_48 = {var_0: var_47}
    var_49 = 'file1.txt'
    var_50 = module_0.is_copy_only_path(var_49, var_48)
    assert var_50 is True
    var_51 = 'fileA.txt'
    var_52 = module_0.is_copy_only_path(var_51, var_48)
    assert var_52 is True
    var_53 = module_0.is_copy_only_path(var_8, var_48)
    assert var_53 is False
    var_54 = 'file12.txt'
    var_55 = module_0.is_copy_only_path(var_54, var_48)
    assert var_55 is False
    var_56 = 'specific/path/file.txt'
    var_57 = [var_56]
    var_58 = {var_1: var_57}
    var_59 = {var_0: var_58}
    var_60 = module_0.is_copy_only_path(var_56, var_59)
    assert var_60 is True
    var_61 = 'specific/path/other.txt'
    var_62 = module_0.is_copy_only_path(var_61, var_59)
    assert var_62 is False
    var_63 = 'different/path/file.txt'
    var_64 = module_0.is_copy_only_path(var_63, var_59)
    assert var_64 is False

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_4.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
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
    var_1 = 'my_poject'
    var_2 = {var_1: var_1}
    module_0.render_and_create_dir(var_1, var_2, var_1, var_0)

def test_case_19():
    var_0 = 'key1'
    var_1 = 'original_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
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
    var_6 = 'value1'
    var_7 = {var_0: var_6}
    var_8 = 'key2'
    var_9 = 'value2'
    var_10 = {var_8: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = {var_0: var_6}
    var_13 = {var_8: var_9}
    var_14 = True
    var_15 = module_0.apply_overwrites_to_context(var_12, var_13, in_dictionary_variable=var_14)
    var_16 = 'choices'
    var_17 = 'option1'
    var_18 = 'option2'
    var_19 = 'option3'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_16: var_20}
    var_22 = [var_18, var_19]
    var_23 = {var_16: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = [var_17, var_18]
    var_26 = {var_16: var_25}
    var_27 = 'invalid'
    var_28 = [var_27]
    var_29 = {var_16: var_28}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_26, var_29)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'name'
    var_1 = 'new_value'
    var_2 = {var_0: var_1}
    var_3 = '`4NAl\r\t'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
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
    var_5 = 'kc'
    var_6 = {var_5: var_1}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
    var_8 = 'neseed'
    var_9 = module_2.findall(var_3, var_1)
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
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}
    var_12 = True
    var_13 = module_0.apply_overwrites_to_context(var_10, var_11, in_dictionary_variable=var_12)
    var_14 = 'choice'
    var_15 = 'option1'
    var_16 = 'option2'
    var_17 = 'option3'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = module_0.apply_overwrites_to_context(var_19, var_19)
    module_0.generate_context(default_context=var_1, extra_context=var_2)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = '7JHtp(K'
    var_1 = {var_0: var_0}
    var_2 = module_5.py_encode_basestring_ascii(var_0)
    assert var_2 == '"7JHtp(K"'
    assert f'{type(module_5.ESCAPE).__module__}.{type(module_5.ESCAPE).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.ESCAPE_ASCII).__module__}.{type(module_5.ESCAPE_ASCII).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.HAS_UTF8).__module__}.{type(module_5.HAS_UTF8).__qualname__}' == 're.Pattern'
    assert module_5.ESCAPE_DCT == {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007', '\x0b': '\\u000b', '\x0e': '\\u000e', '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017', '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f'}
    assert module_5.i == 31
    assert module_5.INFINITY == pytest.approx(1e309, abs=0.01, rel=0.01)
    var_3 = module_4.Environment()
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
    module_0.generate_file(var_2, var_0, var_1, var_3, var_2)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = '7JHtp(K'
    var_1 = {var_0: var_0}
    var_2 = module_5.py_encode_basestring_ascii(var_0)
    assert var_2 == '"7JHtp(K"'
    assert f'{type(module_5.ESCAPE).__module__}.{type(module_5.ESCAPE).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.ESCAPE_ASCII).__module__}.{type(module_5.ESCAPE_ASCII).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.HAS_UTF8).__module__}.{type(module_5.HAS_UTF8).__qualname__}' == 're.Pattern'
    assert module_5.ESCAPE_DCT == {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007', '\x0b': '\\u000b', '\x0e': '\\u000e', '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017', '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f'}
    assert module_5.i == 31
    assert module_5.INFINITY == pytest.approx(1e309, abs=0.01, rel=0.01)
    var_3 = module_4.Environment()
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
    var_4 = False
    module_0.generate_file(var_2, var_0, var_1, var_3, var_4)