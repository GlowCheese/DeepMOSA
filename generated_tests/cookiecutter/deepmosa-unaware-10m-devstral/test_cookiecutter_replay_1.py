# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = module_0.get_file_name(var_0, var_1)
    assert var_3 == 'test_replay_dir/test_template.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    module_0.load(var_2, var_3)