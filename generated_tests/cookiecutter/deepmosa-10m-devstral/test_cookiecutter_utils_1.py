# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.utils as module_0

def test_case_0():
    pass

def test_case_1():
    var_0 = '#d\tq|/ZjOlX#'
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