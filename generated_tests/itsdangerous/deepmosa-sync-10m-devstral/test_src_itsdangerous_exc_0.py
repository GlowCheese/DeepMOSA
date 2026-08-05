# Check out: https://github.com/GlowCheese/deepmosa
import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = '[aA$['
    var_1 = module_0.BadTimeSignature(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadTimeSignature'
    assert var_1.message == '[aA$['
    assert var_1.payload is None
    assert var_1.date_signed is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_1():
    var_0 = None
    var_1 = module_0.BadSignature(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadSignature'
    assert var_1.message is None
    assert var_1.payload is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.__str__()

def test_case_2():
    var_0 = "\r1&r|oAl\n'0}cr#{YK"
    var_1 = module_0.BadHeader(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadHeader'
    assert var_1.message == "\r1&r|oAl\n'0}cr#{YK"
    assert var_1.payload is None
    assert var_1.header is None
    assert var_1.original_error is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_3():
    var_0 = 'mu.E '
    var_1 = module_0.BadPayload(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.exc.BadPayload'
    assert var_1.message == 'mu.E '
    assert var_1.original_error is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = module_0.BadTimeSignature(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.exc.BadTimeSignature'
    assert f'{type(var_3.message).__module__}.{type(var_3.message).__qualname__}' == 'src.itsdangerous.exc.BadPayload'
    assert f'{type(var_3.payload).__module__}.{type(var_3.payload).__qualname__}' == 'src.itsdangerous.exc.BadPayload'
    assert var_3.date_signed is None