# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.utils as module_0
import pytest


def test_case_0():
    pass

def test_case_1():
    var_0 = 'K*OU}LZ|T'
    with pytest.raises(OSError):
        module_0.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.make_sure_path_exists(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.force_delete(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.rmtree(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.make_executable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.simple_filter(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.create_tmp_repo_dir(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.create_env_with_context(var_0)

def test_case_9():
    var_0 = None
    var_1 = module_0.work_in(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == (None,)
    assert var_1.kwds == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.utils'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_2 = var_1.__enter__()