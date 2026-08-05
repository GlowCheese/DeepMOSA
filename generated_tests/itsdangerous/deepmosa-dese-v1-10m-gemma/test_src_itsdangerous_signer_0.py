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
    var_2 = var_1.get_signature(var_0)
    assert var_2 == b'FK-5z7iKpiDzBoYGclhyYn8fWIs'

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
def test_case_8():
    var_0 = 'on%'
    var_1 = module_0.Signer(var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'on%']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'on%'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_1.sign(var_0)

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
    var_0 = b'/\xfc.\xbc\xf9\x9a\xc2\x1cy\x8e\xfb'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'/\xfc.\xbc\xf9\x9a\xc2\x1cy\x8e\xfb']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == b'/\xfc.\xbc\xf9\x9a\xc2\x1cy\x8e\xfb'
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
def test_case_13():
    var_0 = 'on%'
    var_1 = module_0.Signer(var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'on%']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'on%'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_1.get_signature(var_0)

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

def test_case_17():
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
    var_2 = var_1.derive_key(var_0)
    assert var_2 == b'none'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'on%'
    var_1 = module_0.Signer(var_0, sep=var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'on%']
    assert var_1.sep == b'on%'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'on%'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_1.unsign(var_0)

def test_case_19():
    var_0 = b'\x15\xca\xb0\xd8o\xa5'
    var_1 = 'jYt'
    var_2 = module_0.Signer(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x15\xca\xb0\xd8o\xa5']
    assert var_2.sep == b'\x15\xca\xb0\xd8o\xa5'
    assert var_2.salt == b'jYt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate(var_0)
    assert var_3 is False

def test_case_20():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Signer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.sep == b'.'
    assert var_2.salt == b'salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = b'hello world'
    var_4 = var_2.sign(var_3)
    assert var_4 == b'hello world.OIwDcQces0GGRWGJsCBNLHF6R6c'
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'hello world'

def test_case_21():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'old_key', b'new_key']
    assert var_4.sep == b'.'
    assert var_4.salt == b'salt'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = module_0.Signer(var_0, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b'old_key']
    assert var_5.sep == b'.'
    assert var_5.salt == b'salt'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = 'old_data'
    var_7 = var_5.sign(var_6)
    assert var_7 == b'old_data.Z-XVGHx2lvrlogVDgqOQHk_K39o'
    var_8 = var_4.unsign(var_7)
    assert var_8 == b'old_data'
    var_9 = module_0.Signer(var_1, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_9.secret_keys == [b'new_key']
    assert var_9.sep == b'.'
    assert var_9.salt == b'salt'
    assert var_9.key_derivation == 'django-concat'
    assert f'{type(var_9.algorithm).__module__}.{type(var_9.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_10 = 'new_data'
    var_11 = var_9.sign(var_10)
    assert var_11 == b'new_data.kLpbwck04FIkOCu5N9h5nbjbaMU'
    var_12 = var_4.unsign(var_11)
    assert var_12 == b'new_data'

def test_case_22():
    var_0 = b'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'hmac'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b'(\xceIgt\xb6\xffz\xf9\xd1\xd4Mt\x02Ll\x1b\t\x82\xcf'

def test_case_23():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'secret']
    assert var_3.sep == b'.'
    assert var_3.salt == b'salt'
    assert var_3.key_derivation == 'concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.derive_key()
    assert var_4 == b'\xda\x00\xec.o\xf9\xedM4+$\xa1n&,\x82\xf3\xc8\xb1\x0b'