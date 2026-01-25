# Check out: https://github.com/GlowCheese/deepmosa
import click.exceptions as module_1
import click.globals as module_2
import cookiecutter.cli as module_0
import pytest


def test_case_0():
    var_0 = module_0.version_msg()
    assert var_0 == 'Cookiecutter 2.6.0 from /workspace/project (Python 3.10.19 (main, Dec 30 2025, 00:42:16) [GCC 14.2.0])'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_1():
    var_0 = module_0.version_msg()
    assert var_0 == 'Cookiecutter 2.6.0 from /workspace/project (Python 3.10.19 (main, Dec 30 2025, 00:42:16) [GCC 14.2.0])'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    with pytest.raises(module_1.BadParameter):
        module_0.validate_extra_context(var_0, var_0, var_0)

def test_case_2():
    var_0 = []
    var_1 = module_0.validate_extra_context(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'zM^e9g8a;v:IL='
    var_1 = [var_0, var_0]
    var_2 = None
    var_3 = module_0.validate_extra_context(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.OrderedDict'
    assert len(var_3) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_3.visit_AnnAssign(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    var_1 = module_2.push_context(var_0)
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    module_0.list_installed_templates(var_0, var_1)