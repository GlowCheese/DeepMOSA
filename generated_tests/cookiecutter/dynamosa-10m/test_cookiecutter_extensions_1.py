# Check out: https://github.com/GlowCheese/deepmosa
import platform as module_1

import cookiecutter.extensions as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.JsonifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.RandomStringExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.system()
    assert var_0 == 'Linux'
    module_0.SlugifyExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.UUIDExtension(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '\tp[r'
    module_0.TimeExtension(var_0)