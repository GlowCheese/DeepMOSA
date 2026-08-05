# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous.exc as module_2
import hmac as module_3

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

def test_case_1():
    var_0 = b',X\xf4\x8a\x1c\xa03Bd9\x05u|.\x16'
    var_1 = module_0.Signer(var_0, sep=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b',X\xf4\x8a\x1c\xa03Bd9\x05u|.\x16']
    assert var_1.sep == b',X\xf4\x8a\x1c\xa03Bd9\x05u|.\x16'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'

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
    var_0 = module_0.NoneAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = var_0.get_signature(var_1, var_1)
    assert var_2 == b''
    var_3 = 'h~a'
    var_4 = module_0.Signer(var_2, key_derivation=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'']
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'h~a'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_4.derive_key()

def test_case_7():
    var_0 = []
    var_1 = b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"
    var_2 = module_0.NoneAlgorithm()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = module_0.Signer(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == []
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.derive_key(var_1)
    assert var_4 == b'\xa8\xbd\xa6\xa4\x82\xf2\xe3d=\xfa\xe8\xf7\x0e\xab\x19\xe9w\xbb:\x9b'
    var_5 = module_0.Signer(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"]
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = None
    var_7 = var_5.sign(var_1)
    assert var_7 == b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<.m8UP8EfS5TsCRiXv2fAsBBuN1II"
    var_8 = module_1.base64_encode(var_1)
    assert var_8 == b'J9iAAmcso6_jz1gk59KcYTw'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    var_9 = var_5.validate(var_8)
    assert var_9 is False
    var_10 = var_3.validate(var_7)
    assert var_10 is False
    var_11 = 'hiat'
    var_12 = var_5.verify_signature(var_6, var_6)
    assert var_12 is False
    var_13 = module_0.Signer(var_1, key_derivation=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_13.secret_keys == [b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"]
    assert var_13.sep == b'.'
    assert var_13.salt == b'itsdangerous.Signer'
    assert var_13.key_derivation == 'hiat'
    assert f'{type(var_13.algorithm).__module__}.{type(var_13.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_14 = module_0.Signer(var_8, algorithm=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_14.secret_keys == [b'J9iAAmcso6_jz1gk59KcYTw']
    assert var_14.sep == b'.'
    assert var_14.salt == b'itsdangerous.Signer'
    assert var_14.key_derivation == 'django-concat'
    assert f'{type(var_14.algorithm).__module__}.{type(var_14.algorithm).__qualname__}' == 'src.itsdangerous.signer.Signer'
    with pytest.raises(TypeError):
        var_13.derive_key()

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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = []
    var_1 = b'\nh\x9e\x08tQ\x1e\xc7\xc6\x8fZ\x81\x82\xfb^\xcd\xa2g9'
    var_2 = None
    var_3 = module_0.Signer(var_0, sep=var_1, digest_method=var_2, algorithm=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == []
    assert var_3.sep == b'\nh\x9e\x08tQ\x1e\xc7\xc6\x8fZ\x81\x82\xfb^\xcd\xa2g9'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3.sign(var_1)

def test_case_10():
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

def test_case_11():
    var_0 = b'|\xed\xe7\xad\xbd\xf1\xbc\x01O'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'|\xed\xe7\xad\xbd\xf1\xbc\x01O']
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

@pytest.mark.xfail(strict=True)
def test_case_14():
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

def test_case_15():
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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'D\x87llTb\xb5AT\x9a\xbc.-1nU5Ooxa70bkJYK4W887wLtuKs'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

def test_case_16():
    var_0 = []
    var_1 = b'\nh\x9e\x08tQ\x1e\xc6\x8fZ/\x81\x82.^\xcd\xa2g9'
    var_2 = None
    var_3 = module_0.Signer(var_0, sep=var_1, digest_method=var_2, algorithm=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == []
    assert var_3.sep == b'\nh\x9e\x08tQ\x1e\xc6\x8fZ/\x81\x82.^\xcd\xa2g9'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    with pytest.raises(module_2.BadSignature):
        var_3.unsign(var_1)

def test_case_17():
    var_0 = ''
    var_1 = None
    with pytest.raises(ValueError):
        module_0.Signer(var_0, sep=var_0, key_derivation=var_1, algorithm=var_1)

def test_case_18():
    var_0 = []
    var_1 = module_0.SigningAlgorithm(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = module_0.Signer(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == []
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = []
    var_1 = module_0.SigningAlgorithm(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"
    var_3 = module_0.NoneAlgorithm()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_4 = module_0.Signer(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == []
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.derive_key(var_2)
    assert var_5 == b'\xa8\xbd\xa6\xa4\x82\xf2\xe3d=\xfa\xe8\xf7\x0e\xab\x19\xe9w\xbb:\x9b'
    var_6 = module_0.Signer(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"]
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = None
    var_8 = var_6.sign(var_2)
    assert var_8 == b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<.m8UP8EfS5TsCRiXv2fAsBBuN1II"
    var_9 = '}vXl55(!US;cCnet'
    var_10 = module_2.BadData(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'src.itsdangerous.exc.BadData'
    assert var_10.message == '}vXl55(!US;cCnet'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    var_11 = {var_5, var_10}
    var_12 = module_0.Signer(var_11, var_4, digest_method=var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert f'{type(var_12.secret_keys).__module__}.{type(var_12.secret_keys).__qualname__}' == 'builtins.list'
    assert len(var_12.secret_keys) == 2
    assert var_12.sep == b'.'
    assert f'{type(var_12.salt).__module__}.{type(var_12.salt).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_12.key_derivation == 'django-concat'
    assert var_12.digest_method == b"'\xd8\x80\x02g,\xa3\xaf\xe3\xcfX$\xe7\xd2\x9ca<"
    assert f'{type(var_12.algorithm).__module__}.{type(var_12.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_13 = var_4.derive_key(var_8)
    assert var_13 == b'\x8c\x82\xd8Z\x18\xd9]\\\xd86\x96\x19n\xe9<\x0f0\xf4XD'
    module_3.new(var_7, digestmod=var_7)