# Check out: https://github.com/GlowCheese/deepmosa
import click.exceptions as module_1
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
    var_0 = ''
    var_1 = module_0.validate_extra_context(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_3():
    var_0 = '='
    var_1 = module_0.validate_extra_context(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 1
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False