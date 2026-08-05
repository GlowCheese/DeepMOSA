# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0
import re as module_1

def test_case_0():
    pass

def test_case_1():
    var_0 = '/m/repldays'
    var_1 = module_0.get_file_name(var_0, var_0)
    assert var_1 == '/m/repldays.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_2():
    var_0 = '/tmp/replays'
    var_1 = 'data.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replays/data.json'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_3():
    var_0 = '/tmp/repYays'
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

def test_case_4():
    var_0 = '/tmp/replay'
    var_1 = module_0.load(var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_2 = module_0.dump(var_0, var_0, var_1)

def test_case_5():
    var_0 = 'my_template'
    var_1 = module_0.load(var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'cookiecutter'
    var_1 = module_1.template(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 're.Pattern'
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_1.Pattern.pattern).__module__}.{type(module_1.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Pattern.flags).__module__}.{type(module_1.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Pattern.groups).__module__}.{type(module_1.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Pattern.groupindex).__module__}.{type(module_1.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = {var_0: var_1, var_0: var_1}
    module_0.dump(var_0, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '$l1"j58P'
    module_0.load(var_0, var_0)