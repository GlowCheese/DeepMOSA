# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.encoding as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = '!\re+]![)^(,Y'
    with pytest.raises(module_1.BadData):
        module_0.base64_decode(var_0)

def test_case_1():
    var_0 = b'\xd1\xd1'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_2():
    var_0 = True
    var_1 = module_0.int_to_bytes(var_0)
    assert var_1 == b'\x01'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.base64_encode(var_0)

def test_case_4():
    var_0 = b'w\xb0M'
    var_1 = module_0.bytes_to_int(var_0)
    assert var_1 == 7843917
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadData):
        module_0.base64_decode(var_0)