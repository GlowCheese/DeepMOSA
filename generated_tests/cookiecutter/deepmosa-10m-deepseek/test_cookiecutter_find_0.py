# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.find as module_0
import pytest


def test_case_0():
    var_0 = None
    with pytest.raises(module_1.NonTemplatedInputDirException):
        module_0.find_template(var_0, var_0)