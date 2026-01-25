# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import cookiecutter.config as module_0
import pytest
import yaml as module_1


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
    var_0 = None
    var_1 = module_0.get_user_config(var_0)
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
    var_2 = module_0.get_user_config(default_config=var_0)
    var_3 = module_0.get_user_config(default_config=var_2)
    module_1.full_load(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
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

def test_case_3():
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = "IV!#&fY{']'"
    var_2 = {var_1: var_1, var_1: var_1}
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
    module_0.get_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '*)b54@-K,Q7Wg{SA'
    module_0.get_user_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'CsK*u2-"Fsp_M'
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
    var_2 = None
    var_3 = module_0.get_user_config()
    var_4 = module_2.ChainMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.ChainMap'
    assert len(var_4) == 0
    assert f'{type(module_2.ChainMap.fromkeys).__module__}.{type(module_2.ChainMap.fromkeys).__qualname__}' == 'builtins.method'
    assert f'{type(module_2.ChainMap.parents).__module__}.{type(module_2.ChainMap.parents).__qualname__}' == 'builtins.property'
    var_5 = module_0.get_user_config()
    module_0.get_config(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '/'
    module_0.get_user_config(var_0)