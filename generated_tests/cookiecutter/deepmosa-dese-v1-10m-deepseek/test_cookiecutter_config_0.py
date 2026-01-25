# Check out: https://github.com/GlowCheese/deepmosa
import re as module_1

import cookiecutter.config as module_0
import cookiecutter.exceptions as module_2
import pytest


def test_case_0():
    var_0 = module_0.get_user_config()
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.get_user_config()
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4
    var_1 = None
    module_0.merge_configs(var_1, var_0)

def test_case_2():
    var_0 = 'kmQ3m G'
    var_1 = 'yk`dM4Y'
    var_2 = {var_0: var_0, var_1: var_1, var_0: var_0, var_1: var_1}
    var_3 = module_0.merge_configs(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4

def test_case_3():
    var_0 = "IV!#&fY{']'"
    var_1 = 'oW'
    var_2 = {var_0: var_0, var_1: var_0}
    var_3 = module_0.get_user_config(default_config=var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.get_user_config(default_config=var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4
    var_2 = module_0.get_user_config(var_0, var_1)
    module_0.get_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/path/to/custom/config.yaml'
    module_0.get_user_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = '#m24fa,.#UFY'
    var_2 = '~/o'
    var_3 = {var_1: var_1, var_2: var_2}
    var_4 = module_0.get_user_config(var_0, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4
    var_5 = None
    var_6 = True
    var_7 = module_0.get_user_config(var_5)
    var_8 = module_0.get_user_config(default_config=var_6)
    var_9 = True
    var_10 = module_0.get_user_config(var_5)
    var_11 = module_0.get_user_config(var_5, var_9)
    var_12 = module_0.get_user_config(default_config=var_5)
    module_0.get_config(var_5)

def test_case_7():
    var_0 = module_1.RegexFlag.MULTILINE
    var_1 = module_0.get_user_config()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.config'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.USER_CONFIG_PATH == '/.cookiecutterrc'
    assert module_0.BUILTIN_ABBREVIATIONS == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}'}
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'builtins.dict'
    assert len(module_0.DEFAULT_CONFIG) == 4
    var_2 = module_0.get_user_config(default_config=var_1)
    with pytest.raises(module_2.ConfigDoesNotExistException):
        module_0.get_config(var_0)
    var_3 = var_0.expect_first_block_sequence_item()