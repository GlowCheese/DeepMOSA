# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.hooks as module_0
import pytest


def test_case_0():
    var_0 = '>/\t=(pi\n,iy7a'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0

def test_case_1():
    var_0 = None
    var_1 = module_0.run_hook(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'qY8BQ`&WO'
    module_0.run_script(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.run_pre_prompt_hook(var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0

def test_case_4():
    var_0 = '<\tjukmQ3m GQ\\'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/nonexistent/path/script.py'
    module_0.run_script(var_0)

def test_case_6():
    var_0 = 'pre_prompt.py'
    var_1 = 'pre_prompt'
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0
    var_3 = 'pre_prompt.sh'
    var_4 = module_0.valid_hook(var_3, var_1)
    assert var_4 is True
    var_5 = 'pre_gen_project.py'
    var_6 = 'pre_gen_project'
    var_7 = module_0.valid_hook(var_5, var_6)
    assert var_7 is True
    var_8 = 'post_gen_project.sh'
    var_9 = 'post_gen_project'
    var_10 = module_0.valid_hook(var_8, var_9)
    assert var_10 is True
    var_11 = 'pre_prompt.py~'
    var_12 = module_0.valid_hook(var_11, var_1)
    assert var_12 is False
    var_13 = 'pre_gen_project.sh~'
    var_14 = module_0.valid_hook(var_13, var_6)
    assert var_14 is False
    var_15 = module_0.valid_hook(var_0, var_9)
    assert var_15 is False
    var_16 = 'pre_gen_project.sh'
    var_17 = module_0.valid_hook(var_16, var_1)
    assert var_17 is False
    var_18 = 'unsupported_hook.py'
    var_19 = 'unsupported_hook'
    var_20 = module_0.valid_hook(var_18, var_19)
    assert var_20 is False
    var_21 = 'invalid.sh'
    var_22 = 'invalid'
    var_23 = module_0.valid_hook(var_21, var_22)
    assert var_23 is False
    var_24 = 'pre_prompt.txt'
    var_25 = module_0.valid_hook(var_24, var_1)
    assert var_25 is True
    var_26 = module_0.valid_hook(var_1, var_1)
    assert var_26 is True

def test_case_7():
    var_0 = None
    var_1 = '.'
    var_2 = module_0.find_hook(var_0, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.hooks'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.EXIT_SUCCESS == 0