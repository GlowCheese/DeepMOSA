# Check out: https://github.com/GlowCheese/deepmosa
import re as module_1

import cookiecutter.exceptions as module_2
import cookiecutter.repository as module_0
import pytest


def test_case_0():
    var_0 = 'FuGz6#s)?MPuaWU'
    var_1 = module_0.repository_has_cookiecutter_json(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.RegexFlag.ASCII
    module_0.expand_abbreviations(var_0, var_0)

def test_case_2():
    var_0 = '9U'
    var_1 = {}
    var_2 = module_0.expand_abbreviations(var_0, var_1)
    assert var_2 == '9U'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.is_repo_url(var_0)

def test_case_4():
    var_0 = '1WNXS<<cH@]3W%G.odd'
    var_1 = module_0.is_zip_file(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

def test_case_5():
    var_0 = '//'
    var_1 = module_0.repository_has_cookiecutter_json(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    var_1 = '#sX\x0b=[qz6oej^Y"mk'
    module_0.determine_repo_dir(var_1, var_0, var_0, var_1, var_0)

def test_case_7():
    var_0 = '(1'
    var_1 = '|P1{\x0c.3*'
    with pytest.raises(module_2.RepositoryNotFound):
        module_0.determine_repo_dir(var_0, var_1, var_0, var_0, var_0)

def test_case_8():
    var_0 = "X5KKm*{'8*"
    var_1 = 'V1jO.3*'
    with pytest.raises(module_2.RepositoryNotFound):
        module_0.determine_repo_dir(var_0, var_1, var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ':`eh\t'
    var_1 = 'V1jO.3'
    module_0.determine_repo_dir(var_0, var_1, var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'Oy@U:4Y.zq]>YG#K'
    var_1 = 'n0D[+jj`'
    module_0.determine_repo_dir(var_0, var_1, var_1, var_1, var_1, var_1, var_1)