# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = -2758.383612
    var_1 = module_0.BadSignature(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadSignature'
    assert var_1.message == pytest.approx(-2758.383612, abs=0.01, rel=0.01)
    assert var_1.payload is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_1():
    var_0 = ';'
    var_1 = module_0.BadHeader(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadHeader'
    assert var_1.message == ';'
    assert var_1.payload is None
    assert var_1.header is None
    assert var_1.original_error is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.__str__()
    assert var_2 == ';'

def test_case_2():
    var_0 = 'L0@h#C4\x0c`mL5Wvd'
    var_1 = None
    var_2 = module_0.BadTimeSignature(var_0, date_signed=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.exc.BadTimeSignature'
    assert var_2.message == 'L0@h#C4\x0c`mL5Wvd'
    assert var_2.payload is None
    assert var_2.date_signed is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_3():
    var_0 = False
    var_1 = module_0.BadHeader(var_0, header=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadHeader'
    assert var_1.message is False
    assert var_1.payload is None
    assert var_1.header is False
    assert var_1.original_error is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_4():
    var_0 = 'g<~8Sv@~|2'
    var_1 = module_0.BadPayload(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadPayload'
    assert var_1.message == 'g<~8Sv@~|2'
    assert var_1.original_error is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216