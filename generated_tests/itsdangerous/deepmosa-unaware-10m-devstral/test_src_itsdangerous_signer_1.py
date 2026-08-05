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

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.HMACAlgorithm()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = module_0.NoneAlgorithm()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_2 = b'\xad\xb6\xecX\x83\xb6\xc4\x80\x83p\xa6\xcf\x96v\xbf\xcaW\xdb\xd2s'
    var_3 = None
    var_4 = b'\xb8\x08\xf4l\xb0\xb6T\x01\x84\x99'
    var_5 = var_1.verify_signature(var_3, var_3, var_4)
    assert var_5 is False
    var_6 = None
    var_7 = module_0.Signer(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_7.secret_keys == [b'\xad\xb6\xecX\x83\xb6\xc4\x80\x83p\xa6\xcf\x96v\xbf\xcaW\xdb\xd2s']
    assert var_7.sep == b'.'
    assert var_7.salt == b'itsdangerous.Signer'
    assert var_7.key_derivation == 'django-concat'
    assert f'{type(var_7.algorithm).__module__}.{type(var_7.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    module_0.SigningAlgorithm(*var_6)

def test_case_7():
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
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'not-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_12.secret_keys == [b'old-key', b'new-key']
    assert var_12.sep == b'.'
    assert var_12.salt == b'itsdangerous.Signer'
    assert var_12.key_derivation == 'django-concat'
    assert f'{type(var_12.algorithm).__module__}.{type(var_12.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_13 = b'test-value'
    var_14 = module_0.Signer(var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_14.secret_keys == [b'old-key']
    assert var_14.sep == b'.'
    assert var_14.salt == b'itsdangerous.Signer'
    assert var_14.key_derivation == 'django-concat'
    assert f'{type(var_14.algorithm).__module__}.{type(var_14.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_15 = var_14.get_signature(var_13)
    assert var_15 == b'5F6EtBobusUzfM2cZ9W9T5_Yr-0'
    var_16 = var_14.get_signature(var_13)
    assert var_16 == b'5F6EtBobusUzfM2cZ9W9T5_Yr-0'
    var_17 = var_12.verify_signature(var_13, var_15)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_13, var_16)
    assert var_18 is True
    var_19 = 'secret-key'
    var_20 = b'test-value'
    var_21 = var_1.verify_signature(var_20, var_17)
    assert var_21 is False
    var_22 = b'wrong-signature'
    var_23 = var_1.verify_signature(var_20, var_22)
    assert var_23 is False
    var_24 = module_0.NoneAlgorithm()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_25 = module_0.Signer(var_19, algorithm=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_25.secret_keys == [b'secret-key']
    assert var_25.sep == b'.'
    assert var_25.salt == b'itsdangerous.Signer'
    assert var_25.key_derivation == 'django-concat'
    assert f'{type(var_25.algorithm).__module__}.{type(var_25.algorithm).__qualname__}' == 'src.itsdangerous.signer.NoneAlgorithm'
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    assert var_27 == b''
    var_28 = var_25.verify_signature(var_26, var_22)
    assert var_28 is False

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
    var_3 = module_0.SigningAlgorithm()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.SigningAlgorithm'

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

@pytest.mark.xfail(strict=True)
def test_case_11():
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

def test_case_12():
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
def test_case_13():
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

def test_case_14():
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

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = b'D\x87llTb\xb5.AT\x9a\xbc'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'D\x87llTb\xb5.AT\x9a\xbc']
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
    var_3 = var_1.sign(var_0)
    assert var_3 == b'D\x87llTb\xb5.AT\x9a\xbc.bqKCLZaK4aaQGfaYsMpvWmHJoO4'
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'D\x87llTb\xb5.AT\x9a\xbc'
    var_5 = var_1.validate(var_4)
    assert var_5 is False
    var_6 = var_1.verify_signature(var_2, var_2)
    assert var_6 is False
    module_1.new(var_2, var_2, var_2)

def test_case_17():
    var_0 = ''
    var_1 = None
    with pytest.raises(ValueError):
        module_0.Signer(var_0, sep=var_0, key_derivation=var_1, algorithm=var_1)

def test_case_18():
    var_0 = ()
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == []
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
    var_3 = None
    with pytest.raises(TypeError):
        module_1.HMAC(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b'KF\x0b\xd4\xc0\xa1^\xb2T\x95\x032\x11F\xbav\xb6\x06b\xab'
    var_4 = len(var_3)
    var_5 = 'hmac'
    var_6 = module_0.Signer(var_0, key_derivation=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_6.secret_keys == [b'secret-key']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'hmac'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.derive_key()
    assert var_7 == b'\xb2*\x05=\xdaI\x92O\xdf\xe1\x05]\x9e\x81J\x9a\xe8\xad\x89n'
    var_4.get_signature(var_7, var_4)

def test_case_20():
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
    var_2 = var_1.derive_key()
    assert var_2 == b'\x95\xc1A>\xc5$\x00\xdb\x8a"\xb5\x0e/1\x8e\x92\x04cdH'
    var_3 = len(var_2)
    var_4 = 'concat'
    var_5 = module_0.Signer(var_0, key_derivation=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b'secret-key']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = var_5.derive_key()
    assert var_6 == b'KF\x0b\xd4\xc0\xa1^\xb2T\x95\x032\x11F\xbav\xb6\x06b\xab'
    var_7 = len(var_6)
    var_8 = 'hmac'
    var_9 = module_0.Signer(var_0, key_derivation=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_9.secret_keys == [b'secret-key']
    assert var_9.sep == b'.'
    assert var_9.salt == b'itsdangerous.Signer'
    assert var_9.key_derivation == 'hmac'
    assert f'{type(var_9.algorithm).__module__}.{type(var_9.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_10 = var_9.derive_key()
    assert var_10 == b'\xb2*\x05=\xdaI\x92O\xdf\xe1\x05]\x9e\x81J\x9a\xe8\xad\x89n'
    var_11 = 'none'
    var_12 = module_0.Signer(var_0, key_derivation=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_12.secret_keys == [b'secret-key']
    assert var_12.sep == b'.'
    assert var_12.salt == b'itsdangerous.Signer'
    assert var_12.key_derivation == 'none'
    assert f'{type(var_12.algorithm).__module__}.{type(var_12.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_13 = var_12.derive_key()
    assert var_13 == b'secret-key'
    var_14 = 'another-secret-key'
    var_15 = var_1.derive_key(var_14)
    assert var_15 == b'~d;\x04\xb67\x11\xf6b\xca\x86B\x89r<\xda>b\xfet'
    var_16 = len(var_15)
    var_17 = 'django-concat'
    var_18 = module_0.Signer(var_0, key_derivation=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_18.secret_keys == [b'secret-key']
    assert var_18.sep == b'.'
    assert var_18.salt == b'itsdangerous.Signer'
    assert var_18.key_derivation == 'django-concat'
    assert f'{type(var_18.algorithm).__module__}.{type(var_18.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_19 = var_18.derive_key()
    assert var_19 == b'\x95\xc1A>\xc5$\x00\xdb\x8a"\xb5\x0e/1\x8e\x92\x04cdH'
    var_20 = b'secret-key'
    var_21 = module_0.Signer(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_21.secret_keys == [b'secret-key']
    assert var_21.sep == b'.'
    assert var_21.salt == b'itsdangerous.Signer'
    assert var_21.key_derivation == 'django-concat'
    assert f'{type(var_21.algorithm).__module__}.{type(var_21.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_22 = var_21.derive_key()
    assert var_22 == b'\x95\xc1A>\xc5$\x00\xdb\x8a"\xb5\x0e/1\x8e\x92\x04cdH'
    var_23 = len(var_22)
    var_24 = 'different-salt'
    var_25 = module_0.Signer(var_0, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_25.secret_keys == [b'secret-key']
    assert var_25.sep == b'.'
    assert var_25.salt == b'different-salt'
    assert var_25.key_derivation == 'django-concat'
    assert f'{type(var_25.algorithm).__module__}.{type(var_25.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_26 = var_25.derive_key()
    assert var_26 == b'c\xfc\xe2_3~\xcb/\x0e\xfdZ\xd1\x1e\x96\xf7\xbb{\xcfM\xc2'
    var_27 = 'invalid'
    var_28 = module_0.Signer(var_0, key_derivation=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_28.secret_keys == [b'secret-key']
    assert var_28.sep == b'.'
    assert var_28.salt == b'itsdangerous.Signer'
    assert var_28.key_derivation == 'invalid'
    assert f'{type(var_28.algorithm).__module__}.{type(var_28.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    with pytest.raises(TypeError):
        var_28.derive_key()