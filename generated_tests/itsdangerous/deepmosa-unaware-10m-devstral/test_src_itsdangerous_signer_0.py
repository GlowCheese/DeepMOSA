# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.signer as module_0

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

@pytest.mark.xfail(strict=True)
def test_case_11():
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

def test_case_12():
    var_0 = 'un'
    var_1 = b'\xbf\xe1E\xe1\xd7\xcd"'
    var_2 = module_0.Signer(var_1, key_derivation=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\xbf\xe1E\xe1\xd7\xcd"']
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
    with pytest.raises(TypeError):
        var_2.derive_key()

@pytest.mark.xfail(strict=True)
def test_case_13():
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

def test_case_14():
    var_0 = None
    var_1 = ''
    with pytest.raises(ValueError):
        module_0.Signer(var_1, var_0, var_1, digest_method=var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
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

def test_case_16():
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
def test_case_17():
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
def test_case_18():
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
def test_case_19():
    var_0 = b'\x15\xca\xb0\xd8o\xa5'
    var_1 = 'h'
    var_2 = module_0.Signer(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_2.sep == b'\x15\xca\xb0\xd8o\xa5'
    assert var_2.salt == b'h'
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
    var_4 = var_2.verify_signature(var_1, var_0)
    assert var_4 is False
    var_5 = var_2.sign(var_0)
    assert var_5 == b'\x15\xca\xb0\xd8o\xa5\x15\xca\xb0\xd8o\xa5zuP7CYyFJ-WcP7Y-bjsvdx_uza4'
    var_6 = module_0.Signer(var_0, key_derivation=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'h'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_2.unsign(var_5)
    assert var_7 == b'\x15\xca\xb0\xd8o\xa5'
    var_6.get_signature(var_5)

def test_case_20():
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
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b'secret-key']
    assert var_6.sep == b'.'
    assert var_6.salt == b'test-salt'
    assert var_6.key_derivation == 'concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.derive_key()
    assert var_7 == b"\xa4s~\xac\xc4\x81\xf2\x16\x9d'5\x86\xbb\x9b\xe3t\x9d'oY"
    var_8 = len(var_7)
    var_9 = 'hmac'
    var_10 = module_0.Signer(var_0, var_1, key_derivation=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_10.secret_keys == [b'secret-key']
    assert var_10.sep == b'.'
    assert var_10.salt == b'test-salt'
    assert var_10.key_derivation == 'hmac'
    assert f'{type(var_10.algorithm).__module__}.{type(var_10.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_11 = var_10.derive_key()
    assert var_11 == b'\xf57\xa0\xc3\xbc\xb0\xd2\x8b\x01\x0b\xdd\x11\x9c\xcfj\xd5\xe9\xa5\x8c\r'
    var_12 = 'none'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_13.secret_keys == [b'secret-key']
    assert var_13.sep == b'.'
    assert var_13.salt == b'test-salt'
    assert var_13.key_derivation == 'none'
    assert f'{type(var_13.algorithm).__module__}.{type(var_13.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_14 = var_13.derive_key()
    assert var_14 == b'secret-key'
    var_15 = module_0.Signer(var_0, var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_15.secret_keys == [b'secret-key']
    assert var_15.sep == b'.'
    assert var_15.salt == b'test-salt'
    assert var_15.key_derivation == 'django-concat'
    assert f'{type(var_15.algorithm).__module__}.{type(var_15.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_16 = 'another-secret'
    var_17 = var_15.derive_key(var_16)
    assert var_17 == b'\x9c\xb4\xcb\xa3F\xa35\xfb\xf9\x01\xb4\x0bwQ\xf4A\xbe\xe5G\x15'
    var_18 = len(var_17)
    var_19 = 'invalid'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_20.secret_keys == [b'secret-key']
    assert var_20.sep == b'.'
    assert var_20.salt == b'test-salt'
    assert var_20.key_derivation == 'invalid'
    assert f'{type(var_20.algorithm).__module__}.{type(var_20.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    with pytest.raises(TypeError):
        var_20.derive_key()

def test_case_21():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.sep == b'.'
    assert var_1.salt == b'secret-key'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    assert var_3 == b'HLwgIBThGe5ewoviOVFvQvIMJOU'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = var_1.verify_signature(var_2, var_3)
    assert var_7 is True
    var_8 = b'invalid-base64!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'hmac'
    var_11 = module_0.Signer(var_0, var_0, key_derivation=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_11.secret_keys == [b'secret-key']
    assert var_11.sep == b'.'
    assert var_11.salt == b'secret-key'
    assert var_11.key_derivation == 'hmac'
    assert f'{type(var_11.algorithm).__module__}.{type(var_11.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_12 = var_11.get_signature(var_2)
    assert var_12 == b'bDiWLvwj_yfL16OgW7o3g4t3YU8'
    var_13 = var_11.verify_signature(var_2, var_12)
    assert var_13 is True
    var_14 = module_0.NoneAlgorithm()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_15 = None
    var_16 = module_0.Signer(var_8, algorithm=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_16.secret_keys == [b'invalid-base64!']
    assert var_16.sep == b'.'
    assert var_16.salt == b'itsdangerous.Signer'
    assert var_16.key_derivation == 'django-concat'
    assert f'{type(var_16.algorithm).__module__}.{type(var_16.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_17 = var_16.verify_signature(var_15, var_8)
    assert var_17 is False
    var_18 = var_1.verify_signature(var_15, var_0)
    assert var_18 is False