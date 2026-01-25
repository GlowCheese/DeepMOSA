# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.repository as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.determine_repo_dir(var_0, var_0, var_0, var_0, var_0, directory=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'g6(bNgm2K!QO'
    module_0.expand_abbreviations(var_0, var_0)

def test_case_2():
    var_0 = {}
    var_1 = '"'
    var_2 = module_0.expand_abbreviations(var_1, var_0)
    assert var_2 == '"'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

def test_case_3():
    var_0 = "'\x0b%0:%f3\x0c$- D"
    var_1 = module_0.repository_has_cookiecutter_json(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.is_repo_url(var_0)

def test_case_5():
    var_0 = "'\x0b%0:%f3\x0c$- D"
    var_1 = module_0.is_zip_file(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

def test_case_6():
    var_0 = 'Ed7S^q=DHD(,m'
    var_1 = '}'
    with pytest.raises(module_1.RepositoryNotFound):
        module_0.determine_repo_dir(var_0, var_1, var_0, var_0, var_1)

def test_case_7():
    var_0 = 'P'
    var_1 = 'T/5OA%x;bU;o'
    with pytest.raises(module_1.RepositoryNotFound):
        module_0.determine_repo_dir(var_0, var_1, var_0, var_0, var_1, directory=var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ':m'
    var_1 = 'r'
    module_0.expand_abbreviations(var_0, var_1)

def test_case_9():
    var_0 = '.'
    var_1 = module_0.repository_has_cookiecutter_json(var_0)
    assert var_1 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.REPO_REGEX).__module__}.{type(module_0.REPO_REGEX).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'C@o(aeDl\n_3 )Pe'
    var_1 = 'boEX1)O7<s;'
    module_0.determine_repo_dir(var_0, var_1, var_0, var_0, var_1)