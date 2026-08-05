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

def test_case_1():
    var_0 = 'rYO-RZiw~zz\rZ7N'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'rYO-RZiw~zz\rZ7N']
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
    var_2 = module_0.NoneAlgorithm()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.Signer(var_0, key_derivation=var_0, algorithm=var_0)

def test_case_3():
    var_0 = module_0.NoneAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\x8dv%\xff\xd6m\xe5'
    var_1 = None
    var_2 = module_0.SigningAlgorithm()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.verify_signature(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.HMACAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = b''
    var_2 = None
    var_0.verify_signature(var_1, var_2, var_2)

def test_case_6():
    var_0 = 'secret-key'
    var_1 = module_0.NoneAlgorithm()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = module_0.Signer(var_0, algorithm=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.sign(var_0)
    assert var_3 == b'secret-key.'
    var_4 = None
    var_5 = var_2.verify_signature(var_4, var_4)
    assert var_5 is False

def test_case_7():
    var_0 = '5X(%G4JJ'
    var_1 = module_0.Signer(var_0, algorithm=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'5X(%G4JJ']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.algorithm == '5X(%G4JJ'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_1, var_0)
    assert var_2 is False

def test_case_8():
    var_0 = b'\xfc\x96p<\x9f\x89\xda\xff\xf5t\x0c\xd2\x9e'
    var_1 = 'rYO-RZiw~zz\rZ7N'
    var_2 = module_0.Signer(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'rYO-RZiw~zz\rZ7N']
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
    var_3 = var_2.verify_signature(var_0, var_0)
    assert var_3 is False
    var_4 = module_0.NoneAlgorithm()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'

def test_case_9():
    var_0 = b'\x0f\xd7\x06\xb6g\x00N\xea\x83\x9b\x02$}+\x9c\xf2\xedZ\xab\xff'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1, key_derivation=var_1, algorithm=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x0f\xd7\x06\xb6g\x00N\xea\x83\x9b\x02$}+\x9c\xf2\xedZ\xab\xff']
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

def test_case_10():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'secret-key'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_1, var_1)
    assert var_2 is False

def test_case_11():
    var_0 = 'secretkey'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'secretkey']
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
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'D\x87llTb\xb5AT\x9a\xbc'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'D\x87llTb\xb5AT\x9a\xbc']
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
    var_2 = None
    var_1.validate(var_2)

def test_case_13():
    var_0 = b'\xfc\x96p<\x9f\x89\xda\xff\xf5t\x0c\xd2\x9e'
    var_1 = 'rYO-RZiw~zz\rZ7N'
    var_2 = module_0.Signer(var_0, sep=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\xfc\x96p<\x9f\x89\xda\xff\xf5t\x0c\xd2\x9e']
    assert var_2.sep == b'\xfc\x96p<\x9f\x89\xda\xff\xf5t\x0c\xd2\x9e'
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
    var_4 = module_0.NoneAlgorithm()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'

def test_case_14():
    var_0 = '.\x0c\n~tzj 5}wIiSCR'
    var_1 = module_0.Signer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'.\x0c\n~tzj 5}wIiSCR']
    assert var_1.sep == b'.'
    assert var_1.salt == b'.\x0c\n~tzj 5}wIiSCR'
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

def test_case_15():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'old-key', b'new-key']
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
    var_4 = 'test-value'
    var_5 = var_3.sign(var_4)
    assert var_5 == b'test-value.fQFR6UIQhcLjIj1zyIniPl-U5rE'
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test-value'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = b'\x8dv%\xff\xd6m\xe5'
    var_1 = 'Z+hPe7fu'
    var_2 = module_0.Signer(var_0, sep=var_0, key_derivation=var_1, digest_method=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'\x8dv%\xff\xd6m\xe5']
    assert var_2.sep == b'\x8dv%\xff\xd6m\xe5'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'Z+hPe7fu'
    assert var_2.digest_method == b'\x8dv%\xff\xd6m\xe5'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2.unsign(var_0)

def test_case_17():
    var_0 = 'sec^retkey'
    var_1 = module_0.Signer(var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'sec^retkey']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert var_1.digest_method == 'sec^retkey'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

def test_case_18():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'old-key', b'new-key']
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

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'secret-key'
    var_1 = 'my-salt'
    var_2 = module_0.Signer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'my-salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b'\x99i\xaf\n\xab\xa8p,q6\xf9`\xd7\xee<\xe1\x14H\xc5\xda'
    var_4 = len(var_3)
    var_5 = 'other-key'
    var_6 = var_2.derive_key(var_5)
    assert var_6 == b'\xd5\xb7\x9b\xe0\xbd\xc2A\x8a M\x9b\xe1,\x12Zb3\xcf9\xf5'
    var_7 = 'concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_8.secret_keys == [b'secret-key']
    assert var_8.sep == b'.'
    assert var_8.salt == b'my-salt'
    assert var_8.key_derivation == 'concat'
    assert f'{type(var_8.algorithm).__module__}.{type(var_8.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_9 = var_8.derive_key()
    assert var_9 == b"$\x1f\xa5\n'\x0e\xd8@<\x8f\xf2\xb5ynm2O\xb2\xe22"
    var_10 = 'hmac'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_11.secret_keys == [b'secret-key']
    assert var_11.sep == b'.'
    assert var_11.salt == b'my-salt'
    assert var_11.key_derivation == 'hmac'
    assert f'{type(var_11.algorithm).__module__}.{type(var_11.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_12 = var_11.derive_key()
    assert var_12 == b'>\x04\x88I/\x03\xcc\xf3Wb8\x0f\xaa\xd5,\xae|\xe0\x16Q'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_13.secret_keys == [b'secret-key']
    assert var_13.sep == b'.'
    assert var_13.salt == b'my-salt'
    assert var_13.key_derivation == 'concat'
    assert f'{type(var_13.algorithm).__module__}.{type(var_13.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_4.derive_key()
    assert var_14 == b'secret-key'

def test_case_20():
    var_0 = 'secret-key'
    var_1 = 'my-salt'
    var_2 = module_0.Signer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'my-salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b'\x99i\xaf\n\xab\xa8p,q6\xf9`\xd7\xee<\xe1\x14H\xc5\xda'
    var_4 = len(var_3)
    var_5 = 'other-key'
    var_6 = var_2.derive_key(var_5)
    assert var_6 == b'\xd5\xb7\x9b\xe0\xbd\xc2A\x8a M\x9b\xe1,\x12Zb3\xcf9\xf5'
    var_7 = 'concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_8.secret_keys == [b'secret-key']
    assert var_8.sep == b'.'
    assert var_8.salt == b'my-salt'
    assert var_8.key_derivation == 'concat'
    assert f'{type(var_8.algorithm).__module__}.{type(var_8.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_9 = var_8.derive_key()
    assert var_9 == b"$\x1f\xa5\n'\x0e\xd8@<\x8f\xf2\xb5ynm2O\xb2\xe22"
    var_10 = 'hmac'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_11.secret_keys == [b'secret-key']
    assert var_11.sep == b'.'
    assert var_11.salt == b'my-salt'
    assert var_11.key_derivation == 'hmac'
    assert f'{type(var_11.algorithm).__module__}.{type(var_11.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_12 = var_11.derive_key()
    assert var_12 == b'>\x04\x88I/\x03\xcc\xf3Wb8\x0f\xaa\xd5,\xae|\xe0\x16Q'
    var_13 = 'none'
    var_14 = module_0.Signer(var_0, var_1, key_derivation=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_14.secret_keys == [b'secret-key']
    assert var_14.sep == b'.'
    assert var_14.salt == b'my-salt'
    assert var_14.key_derivation == 'none'
    assert f'{type(var_14.algorithm).__module__}.{type(var_14.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_15 = var_14.derive_key()
    assert var_15 == b'secret-key'
    var_16 = 'invalid'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_17.secret_keys == [b'secret-key']
    assert var_17.sep == b'.'
    assert var_17.salt == b'my-salt'
    assert var_17.key_derivation == 'invalid'
    assert f'{type(var_17.algorithm).__module__}.{type(var_17.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    with pytest.raises(TypeError):
        var_17.derive_key()

def test_case_21():
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
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    assert var_3 == b'-XmL3-dwfsyW7pCZC0IPy4E6f8s'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = var_1.verify_signature(var_2, var_2)
    assert var_5 is False
    var_6 = 'test-value'
    var_7 = var_1.verify_signature(var_6, var_3)
    assert var_7 is True
    var_8 = 'old-key'
    var_9 = None
    var_10 = var_1.validate(var_3)
    assert var_10 is False
    var_11 = b'!!!invalid-bas\xf364!\xda!'
    var_12 = var_1.verify_signature(var_6, var_9)
    assert var_12 is False
    var_13 = module_0.Signer(var_8, sep=var_7, digest_method=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_13.secret_keys == [b'old-key']
    assert var_13.sep is True
    assert var_13.salt == b'itsdangerous.Signer'
    assert var_13.key_derivation == 'django-concat'
    assert f'{type(var_13.algorithm).__module__}.{type(var_13.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_14 = b''
    var_15 = var_1.get_signature(var_9)
    assert var_15 == b'b4-og8DGknlZ-AXdHwT5DOiFoFs'
    with pytest.raises(ValueError):
        module_0.Signer(var_11, var_3, var_14, var_6)