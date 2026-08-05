# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.hooks as module_0
import cookiecutter.exceptions as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = 'l6x0dJt>uK_V"[hC'
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
    with pytest.raises(module_1.FailedHookException):
        module_0.run_script(var_0)

def test_case_1():
    var_0 = ''
    var_1 = None
    var_2 = module_0.valid_hook(var_0, var_1)
    assert var_2 is False
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

def test_case_2():
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

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'TF!WSO]L1'
    module_0.run_script(var_0, var_0)

def test_case_4():
    var_0 = 'CF/"6\rLK{>\t+'
    var_1 = None
    var_2 = module_0.run_hook(var_0, var_0, var_1)
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

def test_case_5():
    pass

def test_case_6():
    var_0 = '.'
    with pytest.raises(module_1.FailedHookException):
        module_0.run_script(var_0, var_0)

def test_case_7():
    var_0 = 'test.py'
    var_1 = module_2.make_sure_path_exists(var_0)
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
    var_2 = str(var_0)
    var_3 = module_0.find_hook(var_2, var_2)
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
    var_4 = bool('Hook script failed (exit status: 1)' in var_2)

def test_case_8():
    var_0 = 'test.py'
    var_1 = '/tmp'
    with pytest.raises(module_1.FailedHookException):
        module_0.run_script(var_0, var_1)

def test_case_9():
    var_0 = 'test.py'
    var_1 = module_2.make_sure_path_exists(var_0)
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
    with pytest.raises(module_1.FailedHookException):
        module_0.run_script(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.find_hook(var_0)
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
    var_2 = 'test.py'
    var_3 = '/tmp'
    var_4 = module_0.find_hook(var_0, var_3)
    var_5 = module_0.run_hook(var_2, var_0, var_1)
    module_2.create_env_with_context(var_1)

def test_case_11():
    var_0 = 'pre_prompt'
    var_1 = module_0.valid_hook(var_0, var_0)
    assert var_1 is True
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
    var_2 = module_0.find_hook(var_0)