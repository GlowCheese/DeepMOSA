# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.signer as module_0
import src.itsdangerous.exc as module_1

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

def test_case_2():
    var_0 = module_0.SigningAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_3():
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
def test_case_4():
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
def test_case_5():
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

def test_case_6():
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

def test_case_7():
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

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_1.sign(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = b'\x15\xca\xd8o\xa5'
    var_2 = module_0.Signer(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x15\xca\xd8o\xa5']
    assert var_2.sep == b'\x15\xca\xd8o\xa5'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate(var_1)
    assert var_3 is False
    var_3.derive_key(var_0)

def test_case_11():
    var_0 = b'!I\xad'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'!I\xad']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'!I\xad'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False
    with pytest.raises(module_1.BadSignature):
        var_1.unsign(var_0)

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

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = b'\x15\xca\xb0\xd8o\xa5'
    var_1 = 'un'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'un'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2.get_signature(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
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

def test_case_15():
    var_0 = None
    var_1 = ''
    with pytest.raises(ValueError):
        module_0.Signer(var_1, var_0, var_1, digest_method=var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
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

def test_case_17():
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

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.HMACAlgorithm(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'\x15\x7f\xd8o\xa5'
    var_3 = module_0.Signer(var_2, digest_method=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'\x15\x7f\xd8o\xa5']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = module_0.HMACAlgorithm()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_5 = var_3.verify_signature(var_2, var_2)
    assert var_5 is False
    var_6 = var_3.get_signature(var_2)
    assert var_6 == b'orPq3-6HfB1doX0Pc8U40i4w_SQ'
    var_7 = var_3.verify_signature(var_2, var_6)
    assert var_7 is True
    var_8 = b'H\xb2?\x9f\x12\x07\xbe7/\xf3=\xbc\xb3\x1f\xd3\x87n\xd6\xa9\xbf'
    var_1.get_signature(var_0, var_8)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = None
    var_2 = module_0.HMACAlgorithm(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = {var_0, var_2}
    var_4 = module_0.Signer(var_3, algorithm=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert f'{type(var_4.secret_keys).__module__}.{type(var_4.secret_keys).__qualname__}' == 'builtins.list'
    assert len(var_4.secret_keys) == 2
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = '2,nB"x5d27znl'
    var_6 = var_4.verify_signature(var_5, var_0)
    assert var_6 is False
    var_7 = b'\xbf\xe1E\xe1\xd7\xcd"'
    var_8 = b'<\x9fp\xbbk\xf6K$#\xab8'
    var_9 = var_2.get_signature(var_7, var_8)
    assert var_9 == b'\xa3\xe7\x9f\xab%q\xb6\xddG\xe5\xcf\n \x0ci\x83\xa4\xcc\xc6@'
    var_10 = b'4.\x10p\xa3\x00\x96\x1d\x1b~\xc8'
    var_11 = b'24\xb5\x98>\xa7\x10\xd2A'
    var_12 = var_2.get_signature(var_10, var_11)
    assert var_12 == b'@\xfd\x19\xae\xd4\x1f3(@X\xf7\xb1}>*s\xf0P\x97\xfa'
    var_4.get_signature(var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = b'\x15\xca\xb0\xd8o\xa5'
    var_2 = 'h'
    var_3 = module_0.Signer(var_1, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_3.sep == b'\x15\xca\xb0\xd8o\xa5'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.validate(var_1)
    assert var_4 is False
    var_5 = var_3.verify_signature(var_2, var_1)
    assert var_5 is False
    var_6 = var_3.sign(var_1)
    assert var_6 == b'\x15\xca\xb0\xd8o\xa5\x15\xca\xb0\xd8o\xa5yCbMjrjzGdt80DyzbWyL41WtQMA'
    var_7 = module_0.Signer(var_6, key_derivation=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_7.secret_keys == [b'\x15\xca\xb0\xd8o\xa5\x15\xca\xb0\xd8o\xa5yCbMjrjzGdt80DyzbWyL41WtQMA']
    assert var_7.sep == b'.'
    assert var_7.salt == b'itsdangerous.Signer'
    assert var_7.key_derivation == 'h'
    assert f'{type(var_7.algorithm).__module__}.{type(var_7.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_8 = var_3.unsign(var_6)
    assert var_8 == b'\x15\xca\xb0\xd8o\xa5'
    var_7.get_signature(var_6)

def test_case_21():
    var_0 = 'secret-%key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-%key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'test-salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = 'custom-secret'
    var_4 = var_2.derive_key(var_3)
    assert var_4 == b'\xa5\x8e\xce\xeft\x0c\x83;y\xcebG\xa0\xebV(\xe1\xc9*O'
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b'secret-%key']
    assert var_6.sep == b'.'
    assert var_6.salt == b'test-salt'
    assert var_6.key_derivation == 'concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.derive_key()
    assert var_7 == b'\xd5\xd2R\xac\x9aP)\xa1\x95\x08\xa71\xf7\x13\x8f\x85\xc6\x142\x11'
    var_8 = len(var_7)
    var_9 = 'django-concat'
    var_10 = var_2.derive_key()
    assert var_10 == b'1kSIT\x83\xbe-\xa9\x8fP2L\xc79bKB\xa4}'
    var_11 = len(var_10)
    var_12 = 'hmac'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_13.secret_keys == [b'secret-%key']
    assert var_13.sep == b'.'
    assert var_13.salt == b'test-salt'
    assert var_13.key_derivation == 'hmac'
    assert f'{type(var_13.algorithm).__module__}.{type(var_13.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_14 = var_13.derive_key()
    assert var_14 == b'Ro\xb3\xcf\xc6\x1b\x13\xce\x0f-l\xa5/J%ImB\x0ff'
    var_15 = len(var_14)
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_16.secret_keys == [b'secret-%key']
    assert var_16.sep == b'.'
    assert var_16.salt == b'test-salt'
    assert var_16.key_derivation == 'django-concat'
    assert f'{type(var_16.algorithm).__module__}.{type(var_16.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_17 = var_16.derive_key()
    assert var_17 == b'1kSIT\x83\xbe-\xa9\x8fP2L\xc79bKB\xa4}'
    var_18 = 'salt1'
    var_19 = module_0.Signer(var_0, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_19.secret_keys == [b'secret-%key']
    assert var_19.sep == b'.'
    assert var_19.salt == b'salt1'
    assert var_19.key_derivation == 'django-concat'
    assert f'{type(var_19.algorithm).__module__}.{type(var_19.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_20 = 'salt2'
    var_21 = module_0.Signer(var_0, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_21.secret_keys == [b'secret-%key']
    assert var_21.sep == b'.'
    assert var_21.salt == b'salt2'
    assert var_21.key_derivation == 'django-concat'
    assert f'{type(var_21.algorithm).__module__}.{type(var_21.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_22 = var_19.derive_key()
    assert var_22 == b'\x963\xd7J\x0f%\x88\xb5\x87\x08r\x7f\xc9\xd4#\x08\xd9\xed\x99j'
    var_23 = var_21.derive_key()
    assert var_23 == b'\xf8\xd99\x13\xa1wE\xa1\x95\x19L\xc8\xa9g=\xbf\xa9r`\xe2'
    var_24 = 'secret1'
    var_25 = module_0.Signer(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_25.secret_keys == [b'secret1']
    assert var_25.sep == b'.'
    assert var_25.salt == b'itsdangerous.Signer'
    assert var_25.key_derivation == 'django-concat'
    assert f'{type(var_25.algorithm).__module__}.{type(var_25.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_26 = 'secret2'
    var_27 = module_0.Signer(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_27.secret_keys == [b'secret2']
    assert var_27.sep == b'.'
    assert var_27.salt == b'itsdangerous.Signer'
    assert var_27.key_derivation == 'django-concat'
    assert f'{type(var_27.algorithm).__module__}.{type(var_27.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_28 = var_25.derive_key()
    assert var_28 == b'z\x88CWu(B\t?\x9f|>\xcb\xa7\xc3u\x96\xa6\xc5\x87'
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = module_0.Signer(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_32.secret_keys == [b'old-key', b'new-key']
    assert var_32.sep == b'.'
    assert var_32.salt == b'itsdangerous.Signer'
    assert var_32.key_derivation == 'django-concat'
    assert f'{type(var_32.algorithm).__module__}.{type(var_32.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_33 = var_32.derive_key()
    assert var_33 == b'@\x12\xbb\xe6gf\xd6_o\x7f\xc0\x802\x08=\xe6!\x1b6\x14'
    var_34 = var_32.derive_key(var_29)
    assert var_34 == b'\xfc\x05\xee4\xa1\x0c/\xc5\x8d\x1d{\xbe\x0c\xb7\xaa\xae\x1c\xfao\x0e'
    var_35 = var_32.derive_key(var_30)
    assert var_35 == b'@\x12\xbb\xe6gf\xd6_o\x7f\xc0\x802\x08=\xe6!\x1b6\x14'
    var_36 = module_0.Signer(var_0)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_36.secret_keys == [b'secret-%key']
    assert var_36.sep == b'.'
    assert var_36.salt == b'itsdangerous.Signer'
    assert var_36.key_derivation == 'django-concat'
    assert f'{type(var_36.algorithm).__module__}.{type(var_36.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_37 = 'another-secret'
    var_38 = var_36.derive_key(var_37)
    assert var_38 == b'\x02\xe9\xaf\x1c\x91\xc0\xc5\xfb,NA\x8fo\xed[\xb1\xa6\\)\xa9'
    var_39 = b'byte-secret'
    var_40 = var_36.derive_key(var_39)
    assert var_40 == b'\x10\xe7Y.F8W1\xa2\xdbf\x939\xc4\x17\xda\xffs\x01}'
    var_41 = 'invalid'
    var_42 = module_0.Signer(var_0, key_derivation=var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_42.secret_keys == [b'secret-%key']
    assert var_42.sep == b'.'
    assert var_42.salt == b'itsdangerous.Signer'
    assert var_42.key_derivation == 'invalid'
    assert f'{type(var_42.algorithm).__module__}.{type(var_42.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    with pytest.raises(TypeError):
        var_42.derive_key()

def test_case_22():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'test-salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b"n\xa3\x9a:\xd4\xe0)2\xd8.}x9\xdb\x93\xc4\xf2'\xd0\\"
    var_4 = len(var_3)
    var_5 = 'custom-secret'
    var_6 = var_2.derive_key(var_5)
    assert var_6 == b'\xa5\x8e\xce\xeft\x0c\x83;y\xcebG\xa0\xebV(\xe1\xc9*O'
    var_7 = 'concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_8.secret_keys == [b'secret-key']
    assert var_8.sep == b'.'
    assert var_8.salt == b'test-salt'
    assert var_8.key_derivation == 'concat'
    assert f'{type(var_8.algorithm).__module__}.{type(var_8.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_9 = var_8.derive_key()
    assert var_9 == b"\xa4s~\xac\xc4\x81\xf2\x16\x9d'5\x86\xbb\x9b\xe3t\x9d'oY"
    var_10 = len(var_9)
    var_11 = 'django-concat'
    var_12 = module_0.Signer(var_0, var_1, key_derivation=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_12.secret_keys == [b'secret-key']
    assert var_12.sep == b'.'
    assert var_12.salt == b'test-salt'
    assert var_12.key_derivation == 'django-concat'
    assert f'{type(var_12.algorithm).__module__}.{type(var_12.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_13 = var_12.derive_key()
    assert var_13 == b"n\xa3\x9a:\xd4\xe0)2\xd8.}x9\xdb\x93\xc4\xf2'\xd0\\"
    var_14 = len(var_13)
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_16.secret_keys == [b'secret-key']
    assert var_16.sep == b'.'
    assert var_16.salt == b'test-salt'
    assert var_16.key_derivation == 'hmac'
    assert f'{type(var_16.algorithm).__module__}.{type(var_16.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_17 = var_16.derive_key()
    assert var_17 == b'\xf57\xa0\xc3\xbc\xb0\xd2\x8b\x01\x0b\xdd\x11\x9c\xcfj\xd5\xe9\xa5\x8c\r'
    var_18 = len(var_17)
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_20.secret_keys == [b'secret-key']
    assert var_20.sep == b'.'
    assert var_20.salt == b'test-salt'
    assert var_20.key_derivation == 'none'
    assert f'{type(var_20.algorithm).__module__}.{type(var_20.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = 'salt1'
    var_23 = module_0.Signer(var_0, var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_23.secret_keys == [b'secret-key']
    assert var_23.sep == b'.'
    assert var_23.salt == b'salt1'
    assert var_23.key_derivation == 'django-concat'
    assert f'{type(var_23.algorithm).__module__}.{type(var_23.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_24 = 'salt2'
    var_25 = module_0.Signer(var_0, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_25.secret_keys == [b'secret-key']
    assert var_25.sep == b'.'
    assert var_25.salt == b'salt2'
    assert var_25.key_derivation == 'django-concat'
    assert f'{type(var_25.algorithm).__module__}.{type(var_25.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_26 = var_23.derive_key()
    assert var_26 == b'\x857\xc6\xc2q\xcf\xb0\xb7\xd9\xba\xc8\x0f\xe3M\x047t(Z+'
    var_27 = var_25.derive_key()
    assert var_27 == b'\xfe\xc9\xe7\x8b#;:W\x96\x89C\x86\r\xbc\x1a\x028\xab\x90>'
    var_28 = 'secret1'
    var_29 = module_0.Signer(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_29.secret_keys == [b'secret1']
    assert var_29.sep == b'.'
    assert var_29.salt == b'itsdangerous.Signer'
    assert var_29.key_derivation == 'django-concat'
    assert f'{type(var_29.algorithm).__module__}.{type(var_29.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_30 = 'secret2'
    var_31 = module_0.Signer(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_31.secret_keys == [b'secret2']
    assert var_31.sep == b'.'
    assert var_31.salt == b'itsdangerous.Signer'
    assert var_31.key_derivation == 'django-concat'
    assert f'{type(var_31.algorithm).__module__}.{type(var_31.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_32 = var_29.derive_key()
    assert var_32 == b'z\x88CWu(B\t?\x9f|>\xcb\xa7\xc3u\x96\xa6\xc5\x87'
    var_33 = var_31.derive_key()
    assert var_33 == b'\xd3\xeb\x1eZ8\xed\x87\xb6\n\xafn\xb8J\xe1\x9ah"_0\xbd'
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Signer(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_37.secret_keys == [b'old-key', b'new-key']
    assert var_37.sep == b'.'
    assert var_37.salt == b'itsdangerous.Signer'
    assert var_37.key_derivation == 'django-concat'
    assert f'{type(var_37.algorithm).__module__}.{type(var_37.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_38 = var_37.derive_key()
    assert var_38 == b'@\x12\xbb\xe6gf\xd6_o\x7f\xc0\x802\x08=\xe6!\x1b6\x14'
    var_39 = var_37.derive_key(var_34)
    assert var_39 == b'\xfc\x05\xee4\xa1\x0c/\xc5\x8d\x1d{\xbe\x0c\xb7\xaa\xae\x1c\xfao\x0e'
    var_40 = var_37.derive_key(var_35)
    assert var_40 == b'@\x12\xbb\xe6gf\xd6_o\x7f\xc0\x802\x08=\xe6!\x1b6\x14'
    var_41 = module_0.Signer(var_0)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_41.secret_keys == [b'secret-key']
    assert var_41.sep == b'.'
    assert var_41.salt == b'itsdangerous.Signer'
    assert var_41.key_derivation == 'django-concat'
    assert f'{type(var_41.algorithm).__module__}.{type(var_41.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_42 = 'another-secret'
    var_43 = var_41.derive_key(var_42)
    assert var_43 == b'\x02\xe9\xaf\x1c\x91\xc0\xc5\xfb,NA\x8fo\xed[\xb1\xa6\\)\xa9'
    var_44 = b'byte-secret'
    var_45 = var_41.derive_key(var_44)
    assert var_45 == b'\x10\xe7Y.F8W1\xa2\xdbf\x939\xc4\x17\xda\xffs\x01}'
    var_46 = 'invalid'
    var_47 = module_0.Signer(var_0, key_derivation=var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_47.secret_keys == [b'secret-key']
    assert var_47.sep == b'.'
    assert var_47.salt == b'itsdangerous.Signer'
    assert var_47.key_derivation == 'invalid'
    assert f'{type(var_47.algorithm).__module__}.{type(var_47.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    with pytest.raises(TypeError):
        var_47.derive_key()