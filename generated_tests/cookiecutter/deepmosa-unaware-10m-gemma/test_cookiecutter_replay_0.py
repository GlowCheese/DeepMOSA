# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0
import codecs as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'vM&G*}S!c> '
    module_0.load(var_0, var_0)

def test_case_2():
    var_0 = 'vM&G*}S!c> '
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)

def test_case_3():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'test_project'
    var_4 = {var_1: var_3}
    var_5 = {var_2: var_4}
    var_6 = module_0.dump(var_0, var_1, var_5)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_7 = 'my_template.json'
    var_8 = module_0.dump(var_0, var_7, var_5)
    var_9 = 'not_cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_1, var_11)

def test_case_4():
    var_0 = 'cookiecutter'
    var_1 = module_0.dump(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_5():
    var_0 = 'my_tem"$ate'
    var_1 = module_0.load(var_0, var_0)
    assert var_1 == 'cookiecutter'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'cookiecutter'
    var_1 = module_1.IncrementalEncoder()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'codecs.IncrementalEncoder'
    assert var_1.errors == 'strict'
    assert var_1.buffer == ''
    assert module_1.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_1.BOM_LE == b'\xff\xfe'
    assert module_1.BOM_UTF16_LE == b'\xff\xfe'
    assert module_1.BOM_BE == b'\xfe\xff'
    assert module_1.BOM_UTF16_BE == b'\xfe\xff'
    assert module_1.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_1.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_1.BOM == b'\xff\xfe'
    assert module_1.BOM_UTF16 == b'\xff\xfe'
    assert module_1.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_1.BOM32_LE == b'\xff\xfe'
    assert module_1.BOM32_BE == b'\xfe\xff'
    assert module_1.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_1.BOM64_BE == b'\x00\x00\xfe\xff'
    var_2 = {var_0: var_1, var_1: var_1}
    module_0.dump(var_0, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'cookiecutter'
    module_0.load(var_0, var_0)