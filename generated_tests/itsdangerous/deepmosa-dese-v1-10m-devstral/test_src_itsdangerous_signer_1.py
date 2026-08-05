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
    var_0 = 'VM\n|b?eP>rzm55M7h'
    var_1 = module_0.Signer(var_0, key_derivation=var_0, algorithm=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'VM\n|b?eP>rzm55M7h']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'VM\n|b?eP>rzm55M7h'
    assert var_1.algorithm == 'VM\n|b?eP>rzm55M7h'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

def test_case_8():
    var_0 = b'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'none'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.derive_key()
    assert var_3 == b'secret'

def test_case_9():
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
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b'old-key']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = var_5.get_signature(var_4)
    assert var_6 == b'5F6EtBobusUzfM2cZ9W9T5_Yr-0'
    var_7 = None
    var_8 = var_3.verify_signature(var_0, var_7)
    assert var_8 is False

def test_case_10():
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

def test_case_11():
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

def test_case_12():
    var_0 = '^/+F*Qwzz4">X=.\r'
    var_1 = module_0.Signer(var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'^/+F*Qwzz4">X=.\r']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == '^/+F*Qwzz4">X=.\r'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

def test_case_13():
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

def test_case_14():
    var_0 = 'sCD`cbet-key'
    var_1 = module_0.Signer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'sCD`cbet-key']
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
    var_2 = None
    var_1.validate(var_2)

def test_case_16():
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

def test_case_17():
    var_0 = 'D4[WOU. N'
    var_1 = module_0.Signer(var_0, var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'D4[WOU. N']
    assert var_1.sep == b'.'
    assert var_1.salt == b'D4[WOU. N'
    assert var_1.key_derivation == 'D4[WOU. N'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.validate(var_0)
    assert var_2 is False

def test_case_18():
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
def test_case_19():
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

def test_case_20():
    var_0 = 'I'
    var_1 = module_0.Signer(var_0, key_derivation=var_0, digest_method=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_1.secret_keys == [b'I']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'I'
    assert var_1.digest_method == 'I'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_0.Signer.secret_key).__module__}.{type(module_0.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.verify_signature(var_0, var_0)
    assert var_2 is False

def test_case_21():
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
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b'old-key']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = var_3.verify_signature(var_4, var_4)
    assert var_6 is False

def test_case_22():
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
    var_3 = 'django-concat'
    var_4 = module_0.Signer(var_0, key_derivation=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_4.secret_keys == [b'secret-key']
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_5 = b'test-value'
    var_6 = var_2.get_signature(var_5)
    assert var_6 == b'dNNtaSFy2TC11kdcU7OPACzO9Cw'
    var_7 = None
    var_8 = var_4.verify_signature(var_7, var_1)
    assert var_8 is False

def test_case_23():
    var_0 = 'secret'
    var_1 = 'a'
    with pytest.raises(ValueError):
        module_0.Signer(var_0, sep=var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_3 = var_2.digest_method
    var_4 = module_1.new(var_0, digestmod=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'hmac.HMAC'
    assert module_1.trans_5C == b'\\]^_XYZ[TUVWPQRSLMNOHIJKDEFG@ABC|}~\x7fxyz{tuvwpqrslmnohijkdefg`abc\x1c\x1d\x1e\x1f\x18\x19\x1a\x1b\x14\x15\x16\x17\x10\x11\x12\x13\x0c\r\x0e\x0f\x08\t\n\x0b\x04\x05\x06\x07\x00\x01\x02\x03<=>?89:;45670123,-./()*+$%&\' !"#\xdc\xdd\xde\xdf\xd8\xd9\xda\xdb\xd4\xd5\xd6\xd7\xd0\xd1\xd2\xd3\xcc\xcd\xce\xcf\xc8\xc9\xca\xcb\xc4\xc5\xc6\xc7\xc0\xc1\xc2\xc3\xfc\xfd\xfe\xff\xf8\xf9\xfa\xfb\xf4\xf5\xf6\xf7\xf0\xf1\xf2\xf3\xec\xed\xee\xef\xe8\xe9\xea\xeb\xe4\xe5\xe6\xe7\xe0\xe1\xe2\xe3\x9c\x9d\x9e\x9f\x98\x99\x9a\x9b\x94\x95\x96\x97\x90\x91\x92\x93\x8c\x8d\x8e\x8f\x88\x89\x8a\x8b\x84\x85\x86\x87\x80\x81\x82\x83\xbc\xbd\xbe\xbf\xb8\xb9\xba\xbb\xb4\xb5\xb6\xb7\xb0\xb1\xb2\xb3\xac\xad\xae\xaf\xa8\xa9\xaa\xab\xa4\xa5\xa6\xa7\xa0\xa1\xa2\xa3'
    assert module_1.trans_36 == b'67452301>?<=:;89&\'$%"# !./,-*+()\x16\x17\x14\x15\x12\x13\x10\x11\x1e\x1f\x1c\x1d\x1a\x1b\x18\x19\x06\x07\x04\x05\x02\x03\x00\x01\x0e\x0f\x0c\r\n\x0b\x08\tvwturspq~\x7f|}z{xyfgdebc`anolmjkhiVWTURSPQ^_\\]Z[XYFGDEBC@ANOLMJKHI\xb6\xb7\xb4\xb5\xb2\xb3\xb0\xb1\xbe\xbf\xbc\xbd\xba\xbb\xb8\xb9\xa6\xa7\xa4\xa5\xa2\xa3\xa0\xa1\xae\xaf\xac\xad\xaa\xab\xa8\xa9\x96\x97\x94\x95\x92\x93\x90\x91\x9e\x9f\x9c\x9d\x9a\x9b\x98\x99\x86\x87\x84\x85\x82\x83\x80\x81\x8e\x8f\x8c\x8d\x8a\x8b\x88\x89\xf6\xf7\xf4\xf5\xf2\xf3\xf0\xf1\xfe\xff\xfc\xfd\xfa\xfb\xf8\xf9\xe6\xe7\xe4\xe5\xe2\xe3\xe0\xe1\xee\xef\xec\xed\xea\xeb\xe8\xe9\xd6\xd7\xd4\xd5\xd2\xd3\xd0\xd1\xde\xdf\xdc\xdd\xda\xdb\xd8\xd9\xc6\xc7\xc4\xc5\xc2\xc3\xc0\xc1\xce\xcf\xcc\xcd\xca\xcb\xc8\xc9'
    assert module_1.digest_size is None
    assert module_1.HMAC.blocksize == 64
    assert f'{type(module_1.HMAC.name).__module__}.{type(module_1.HMAC.name).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.HMAC.block_size).__module__}.{type(module_1.HMAC.block_size).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.HMAC.digest_size).__module__}.{type(module_1.HMAC.digest_size).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_2.salt
    var_6 = var_2.derive_key()
    assert var_6 == b'(\xceIgt\xb6\xffz\xf9\xd1\xd4Mt\x02Ll\x1b\t\x82\xcf'
    module_1.digest()