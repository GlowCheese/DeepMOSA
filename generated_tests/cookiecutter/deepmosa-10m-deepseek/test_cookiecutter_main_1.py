# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.main as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.cookiecutter(var_0, output_dir=var_0, config_file=var_0, skip_if_file_exists=var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0._patch_import_path_for_repo(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.main'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0._patch_import_path_for_repo(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.main'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = var_1.__enter__()
    var_3 = None
    var_4 = var_1.__exit__(var_0, var_0, var_2)
    var_5 = True
    module_0.cookiecutter(var_4, extra_context=var_0, replay=var_5, config_file=var_0, directory=var_3, accept_hooks=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '2+sG\t:QwR\t6|wir'
    var_1 = None
    var_2 = True
    module_0.cookiecutter(var_0, extra_context=var_1, replay=var_2, config_file=var_1, directory=var_1, accept_hooks=var_1)

def test_case_4():
    var_0 = {}
    var_1 = module_0._patch_import_path_for_repo(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.main'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = "mf86w#'\n"
    var_3 = None
    with pytest.raises(module_1.InvalidModeException):
        module_0.cookiecutter(var_2, var_3, var_3, replay=var_2, config_file=var_2)

def test_case_5():
    var_0 = None
    var_1 = module_0._patch_import_path_for_repo(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.main'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = var_1.__enter__()
    var_3 = '2+sG\t:QwR\t6|wir'
    var_4 = module_0._patch_import_path_for_repo(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    var_5 = None
    var_6 = var_1.__exit__(var_0, var_0, var_2)
    var_7 = var_4.__enter__()
    var_8 = module_0._patch_import_path_for_repo(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'cookiecutter.main._patch_import_path_for_repo'
    var_9 = True
    var_10 = 'KF Kdw(X`"9'
    with pytest.raises(module_1.InvalidModeException):
        module_0.cookiecutter(var_3, extra_context=var_4, replay=var_9, output_dir=var_10, keep_project_on_failure=var_5)