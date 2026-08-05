# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'P_~h6kHs/`Q/I|j]W'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'P_~h6kHs/`Q/I|j]W']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False

def test_case_1():
    var_0 = 'D|N.Vkq@yy<B\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vkq@yy<B\\_?Fp.']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ()
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == []
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.loads_unsafe(var_1)

def test_case_3():
    var_0 = ()
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == []
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.TimedSerializer()

def test_case_5():
    var_0 = 'D|N.aVkq@y7B\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.aVkq@y7B\\_?Fp.']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False

def test_case_6():
    var_0 = ''
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'.amtPyQ.IABKTsQzEt6c7wzNAhixZS6ewsQ'
    var_3 = var_1.unsign(var_2, return_timestamp=var_0)
    assert var_3 == b''

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = ''
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_1.validate(var_2, var_2)

def test_case_8():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = ''
    var_2 = module_0.TimestampSigner(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_2.secret_keys == [b'']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.validate(var_1)
    assert var_3 is False
    var_4 = None
    var_5 = var_2.sign(var_0)
    assert var_5 == b'b\xbb\xb5\x90.amtPyQ.rSDDlPaG3VU17kgAYl4nPcNdtxM'
    var_6 = var_2.unsign(var_5, var_3, var_4)
    assert var_6 == b'b\xbb\xb5\x90'

def test_case_9():
    var_0 = 'm'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'm']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'm.amtPyQ.S3U9UGyScV8OWB-xCv5fj2SqVSg'
    var_3 = var_1.unsign(var_2, return_timestamp=var_0)

def test_case_10():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = 'D|".Vkqkyy<By\\_?Fp.'
    var_2 = 1390
    var_3 = None
    var_4 = module_0.TimedSerializer(var_1, signer_kwargs=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'D|".Vkqkyy<By\\_?Fp.']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_5 = var_4.loads_unsafe(var_0, var_2, var_3)
    var_6 = module_0.TimestampSigner(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_6.secret_keys == [b'D|".Vkqkyy<By\\_?Fp.']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.validate(var_1)
    assert var_7 is False
    var_8 = None
    var_9 = var_6.sign(var_0)
    assert var_9 == b'b\xbb\xb5\x90.amtPyQ.XYr5Je4at58ldjDJTHeey_NLkdQ'
    var_10 = var_4.loads_unsafe(var_1, salt=var_8)
    var_11 = var_6.validate(var_1)
    assert var_11 is False
    var_12 = var_4.dumps(var_3)
    assert var_12 == 'null.amtPyQ.ftcT73J-zkMXBH2GF65oyeOsXOs'
    var_13 = var_6.verify_signature(var_9, var_8)
    assert var_13 is False
    var_14 = var_4.loads(var_12, var_11, var_3)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = b'\x03sIaF\xdc\x96\x05\xe7\xa4\x9ciI\x01i'
    var_1 = 'D|=.VGq@1y<B5_? 2.'
    var_2 = -2555
    var_3 = None
    var_4 = module_0.TimedSerializer(var_1, signer_kwargs=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'D|=.VGq@1y<B5_? 2.']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_5 = var_4.loads_unsafe(var_0, var_2, var_3)
    var_6 = module_0.TimestampSigner(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_6.secret_keys == [b'D|=.VGq@1y<B5_? 2.']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.validate(var_1)
    assert var_7 is False
    var_8 = var_6.sign(var_0)
    assert var_8 == b'\x03sIaF\xdc\x96\x05\xe7\xa4\x9ciI\x01i.amtPyQ.NvoRM5lIUBbiAPvWpSX7Ce7znEY'
    var_9 = var_4.dumps(var_3)
    assert var_9 == 'null.amtPyQ.p8_NzhnI5leC7qOwKVoIg0JVYU0'
    var_10 = var_4.dumps(var_3)
    assert var_10 == 'null.amtPyQ.p8_NzhnI5leC7qOwKVoIg0JVYU0'
    var_11 = var_4.loads(var_10)
    var_4.loads(var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = 'D|N.Vkq@yy<B\\_?Fp.'
    var_2 = 1390
    var_3 = None
    var_4 = module_0.TimedSerializer(var_1, signer_kwargs=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'D|N.Vkq@yy<B\\_?Fp.']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_5 = var_4.loads_unsafe(var_0, var_2, var_3)
    var_6 = module_0.TimestampSigner(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_6.secret_keys == [b'D|N.Vkq@yy<B\\_?Fp.']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.validate(var_1)
    assert var_7 is False
    var_8 = var_6.sign(var_1)
    assert var_8 == b'D|N.Vkq@yy<B\\_?Fp..amtPyQ.XBPkpxjT-7jb-WgSJ24NZV7eKvU'
    var_9 = var_4.dumps(var_3, var_3)
    assert var_9 == 'null.amtPyQ.tvhTLY2c_o87RqKVL_ayO5CdGOg'
    var_10 = var_4.loads(var_9, return_timestamp=var_9)
    var_4.load_payload(var_3, var_9)