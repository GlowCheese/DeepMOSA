# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.vcs as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'T!<l6\x0b.~cTiFM3o:\x0b'
    module_0.clone(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'EoU+;'
    module_0.clone(var_0)

def test_case_2():
    var_0 = 'T!<l6\x0b.~cTiFM3o:\x0b'
    var_1 = module_0.is_vcs_installed(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.vcs'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    assert module_0.BRANCH_ERRORS == ['error: pathspec', 'unknown revision']