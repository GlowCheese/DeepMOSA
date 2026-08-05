# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.encoding as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = 'C'
    with pytest.raises(module_1.BadData):
        module_0.base64_decode(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = True
    module_0.base64_encode(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = -443
    module_0.int_to_bytes(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'H\xe4\xc9\x9cS\xb3^ \xebO#\x1c\x85\xbbz\xc56o\xf8'
    module_0.bytes_to_int(var_0)