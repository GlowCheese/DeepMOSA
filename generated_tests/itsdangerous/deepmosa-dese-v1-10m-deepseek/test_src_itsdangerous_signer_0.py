# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.Signer(var_0, var_0, digest_method=var_0, algorithm=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.SigningAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = None
    var_3 = b'\x9c\xe4V\x8aq\x0c!\x07x'
    var_0.verify_signature(var_3, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = b'r\x9b\xfex\xde{'
    var_2 = module_0.HMACAlgorithm()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.verify_signature(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.NoneAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = module_0.NoneAlgorithm()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_2 = None
    var_3 = b'\xce'
    var_0.verify_signature(var_2, var_3, var_2)

def test_case_5():
    var_0 = None
    var_1 = b'\x15\x7f\xd8o\xa5'
    var_2 = module_0.Signer(var_1, digest_method=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key(var_0)
    assert var_3 == b'\xcc9Q-\xc0\xdf\xd7_6\x99\xe1?\x0f\xf2\xa2\xa3\xbfJV"'

def test_case_6():
    var_0 = b'\x7f\xd8o'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x7f\xd8o']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'\x7f\xd8o'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'

def test_case_7():
    var_0 = b'\x15\x7f\xd8y\xa5'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x15\x7f\xd8y\xa5']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'\x15\x7f\xd8y\xa5'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = b'\x15\x7f\xd8o\xa5'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'\x15\x7f\xd8o\xa5'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False
    var_1.sign(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.NoneAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = b'p\xdd'
    var_3 = b'\xce'
    var_4 = module_0.Signer(var_2, var_1, key_derivation=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'p\xdd']
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == b'\xce'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = b':u\\U2\xad\xf8\xb4BA\x9a\xbf\xee\x05'
    var_6 = module_0.Signer(var_5, var_1, algorithm=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b':u\\U2\xad\xf8\xb4BA\x9a\xbf\xee\x05']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert var_6.algorithm == b'\xce'
    var_4.get_signature(var_2)

def test_case_10():
    var_0 = 'seret-y'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'seret-y']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'

def test_case_11():
    var_0 = 'none'
    var_1 = module_0.Signer(var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'none']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'none'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.derive_key()
    assert var_2 == b'none'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'p\xdd'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'p\xdd']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.validate(var_0)
    assert var_2 is False
    var_3 = b'\xcc'
    var_4 = b'\xaf\xb0u\xadYL\xa0\x11\x89\xb6\xb9%='
    var_2.get_signature(var_3, var_4)

def test_case_13():
    var_0 = 'k\x0c)\t=3h.P a'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'k\x0c)\t=3h.P a']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.validate(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = b'\x15\x7f\xd8o\xa5'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'\x15\x7f\xd8o\xa5'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False
    var_1.get_signature(var_0)

def test_case_15():
    var_0 = ']Y5_'
    var_1 = module_0.Signer(var_0, algorithm=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b']Y5_']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.algorithm == ']Y5_'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

def test_case_16():
    var_0 = None
    var_1 = ''
    with pytest.raises(ValueError):
        module_0.Signer(var_1, var_0, var_1, digest_method=var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.SigningAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = None
    var_3 = b'\x15\x7f\xd8o\xa5'
    var_4 = module_0.Signer(var_3, key_derivation=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4.validate(var_2)

def test_case_18():
    var_0 = b'\x15\x7f\xd8o\xa5'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'\x15\x7f\xd8o\xa5'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False
    with pytest.raises(TypeError):
        var_1.derive_key(var_2)

def test_case_19():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'value.RI9crmxKpuX1wTw2mg_FHFGXRic'
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'value'

def test_case_20():
    var_0 = b'\x15\xca\xb0\xd8o\xa5'
    var_1 = module_0.Signer(var_0, sep=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_1.sep == b'\x15\xca\xb0\xd8o\xa5'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.validate(var_0)
    assert var_2 is False

def test_case_21():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'old-secret', b'new-secret']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = 'value'
    var_5 = var_3.sign(var_4)
    assert var_5 == b'value.D7ERydrfBsY0huw8T88xE-SxQuQ'
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'value'

def test_case_22():
    var_0 = 'L^secret-key'
    var_1 = 'salt'
    var_2 = 'concat'
    var_3 = 'sha25'
    var_4 = module_0.Signer(var_0, var_1, key_derivation=var_2, digest_method=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'L^secret-key']
    assert var_4.sep == b'.'
    assert var_4.salt == b'salt'
    assert var_4.key_derivation == 'concat'
    assert var_4.digest_method == 'sha25'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_4.derive_key()

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = module_0.Signer(var_0, var_1, key_derivation=var_2, digest_method=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'secret-key']
    assert var_4.sep == b'.'
    assert var_4.salt == b'salt'
    assert var_4.key_derivation == 'hmac'
    assert var_4.digest_method == 'sha256'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.derive_key()
    assert var_5 == b'e!P\xdd||\x19j{y\xa6A\x8b\xbe\xdd\x11\x82E\xfd\x11i\xd1\xe6h\xce\xfa\x90EC\xc3\xf1\xec'
    var_6 = b'secret-key'
    var_7 = module_1.new(var_6, digestmod=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'hmac.HMAC'
    assert module_1.trans_5C == b'\\]^_XYZ[TUVWPQRSLMNOHIJKDEFG@ABC|}~\x7fxyz{tuvwpqrslmnohijkdefg`abc\x1c\x1d\x1e\x1f\x18\x19\x1a\x1b\x14\x15\x16\x17\x10\x11\x12\x13\x0c\r\x0e\x0f\x08\t\n\x0b\x04\x05\x06\x07\x00\x01\x02\x03<=>?89:;45670123,-./()*+$%&\' !"#\xdc\xdd\xde\xdf\xd8\xd9\xda\xdb\xd4\xd5\xd6\xd7\xd0\xd1\xd2\xd3\xcc\xcd\xce\xcf\xc8\xc9\xca\xcb\xc4\xc5\xc6\xc7\xc0\xc1\xc2\xc3\xfc\xfd\xfe\xff\xf8\xf9\xfa\xfb\xf4\xf5\xf6\xf7\xf0\xf1\xf2\xf3\xec\xed\xee\xef\xe8\xe9\xea\xeb\xe4\xe5\xe6\xe7\xe0\xe1\xe2\xe3\x9c\x9d\x9e\x9f\x98\x99\x9a\x9b\x94\x95\x96\x97\x90\x91\x92\x93\x8c\x8d\x8e\x8f\x88\x89\x8a\x8b\x84\x85\x86\x87\x80\x81\x82\x83\xbc\xbd\xbe\xbf\xb8\xb9\xba\xbb\xb4\xb5\xb6\xb7\xb0\xb1\xb2\xb3\xac\xad\xae\xaf\xa8\xa9\xaa\xab\xa4\xa5\xa6\xa7\xa0\xa1\xa2\xa3'
    assert module_1.trans_36 == b'67452301>?<=:;89&\'$%"# !./,-*+()\x16\x17\x14\x15\x12\x13\x10\x11\x1e\x1f\x1c\x1d\x1a\x1b\x18\x19\x06\x07\x04\x05\x02\x03\x00\x01\x0e\x0f\x0c\r\n\x0b\x08\tvwturspq~\x7f|}z{xyfgdebc`anolmjkhiVWTURSPQ^_\\]Z[XYFGDEBC@ANOLMJKHI\xb6\xb7\xb4\xb5\xb2\xb3\xb0\xb1\xbe\xbf\xbc\xbd\xba\xbb\xb8\xb9\xa6\xa7\xa4\xa5\xa2\xa3\xa0\xa1\xae\xaf\xac\xad\xaa\xab\xa8\xa9\x96\x97\x94\x95\x92\x93\x90\x91\x9e\x9f\x9c\x9d\x9a\x9b\x98\x99\x86\x87\x84\x85\x82\x83\x80\x81\x8e\x8f\x8c\x8d\x8a\x8b\x88\x89\xf6\xf7\xf4\xf5\xf2\xf3\xf0\xf1\xfe\xff\xfc\xfd\xfa\xfb\xf8\xf9\xe6\xe7\xe4\xe5\xe2\xe3\xe0\xe1\xee\xef\xec\xed\xea\xeb\xe8\xe9\xd6\xd7\xd4\xd5\xd2\xd3\xd0\xd1\xde\xdf\xdc\xdd\xda\xdb\xd8\xd9\xc6\xc7\xc4\xc5\xc2\xc3\xc0\xc1\xce\xcf\xcc\xcd\xca\xcb\xc8\xc9'
    assert module_1.digest_size is None
    assert module_1.HMAC.blocksize == 64
    assert f'{type(module_1.HMAC.name).__module__}.{type(module_1.HMAC.name).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.HMAC.block_size).__module__}.{type(module_1.HMAC.block_size).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.HMAC.digest_size).__module__}.{type(module_1.HMAC.digest_size).__qualname__}' == 'builtins.member_descriptor'
    var_7.derive_key()