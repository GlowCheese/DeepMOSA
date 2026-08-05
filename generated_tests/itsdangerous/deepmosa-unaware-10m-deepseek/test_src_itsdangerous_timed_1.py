# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous.serializer as module_2

def test_case_0():
    var_0 = 'sF.dH '
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'sF.dH ']
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
    var_0 = 'D|N.Vkq@y<B\\_Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vkq@y<B\\_Fp.']
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
    var_0 = None
    var_1 = b'\xc1\x08\xdd=\x81\x9d\x95\xb0?\x90(\xcap\x0fO\xf0D+\xd1'
    var_2 = module_0.TimedSerializer(var_1, serializer_kwargs=var_0, signer_kwargs=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'\xc1\x08\xdd=\x81\x9d\x95\xb0?\x90(\xcap\x0fO\xf0D+\xd1']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.loads(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '=s*"E76z*jq?'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'=s*"E76z*jq?']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.loads(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x060\x9a;\xf3'
    module_0.TimedSerializer(var_0, serializer=var_0)

def test_case_5():
    var_0 = '*Rb&|1\x0b{i0%2(ok` WU'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'*Rb&|1\x0b{i0%2(ok` WU']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_0, var_0)

def test_case_6():
    var_0 = ' ='
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b' =']
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

def test_case_7():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret-key.ams4Jg.WtBlbAxs4C1Dxr_z9na5lzJxKO0'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'secret-key'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'D|N.Vk#0w@^y<B\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vk#0w@^y<B\\_?Fp.']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.derive_key()
    assert var_2 == b'1]\xect\xfb\x8c\xd4\x00;\\r([!NY\xa5+\x0e\xed'
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.sign(var_0)
    assert var_5 == b'D|N.Vk#0w@^y<B\\_?Fp..ams4Jg.dWnH1lmXVijDvCG2xeNJVnEOSV4'
    var_6 = var_1.validate(var_5, var_3)
    assert var_6 is True
    var_7 = module_0.TimedSerializer(var_2, serializer_kwargs=var_4, signer_kwargs=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'1]\xect\xfb\x8c\xd4\x00;\\r([!NY\xa5+\x0e\xed']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_1.get_timestamp()
    assert var_8 == 1785411622
    var_9 = var_7.loads_unsafe(var_5)
    var_1.validate(var_4)

def test_case_9():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret-key.ams4Jg.WtBlbAxs4C1Dxr_z9na5lzJxKO0'
    var_3 = False
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret-key'

def test_case_10():
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

def test_case_11():
    var_0 = None
    var_1 = b'\xc1\x08\xdd=\x81\x9d\x95\xb0?\x90(\xcap\x0fO\xf0D+\xd1'
    var_2 = module_0.TimedSerializer(var_1, serializer_kwargs=var_0, signer_kwargs=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'\xc1\x08\xdd=\x81\x9d\x95\xb0?\x90(\xcap\x0fO\xf0D+\xd1']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dumps(var_0)
    assert var_3 == 'null.ams4Jg.o__M-I7lxT6cnlndGLzxeni1_WE'
    var_4 = var_2.loads(var_3, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'secret']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'test message'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'test message.ams4Jg.InvtHpKhzTN-Znol1U90eGQC0lE'
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test message'
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)
    assert var_6 == b'test message'
    var_7 = var_1.sign(var_3)
    assert var_7 == b'test message.ams4Jg.InvtHpKhzTN-Znol1U90eGQC0lE.ams4Jg.WWlfObJ8IrFFObx00i_BrkvAK9g'
    var_8 = -16
    var_1.unsign(var_3, var_8)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'D|N.Vk#0w@^y<B\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vk#0w@^y<B\\_?Fp.']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = module_1.base64_encode(var_0)
    assert var_2 == b'RHxOLlZrIzB3QF55PEJcXz9GcC4'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.get_timestamp()
    assert var_5 == 1785411622
    var_6 = var_1.sign(var_0)
    assert var_6 == b'D|N.Vk#0w@^y<B\\_?Fp..ams4Jg.dWnH1lmXVijDvCG2xeNJVnEOSV4'
    var_7 = var_1.validate(var_6, var_5)
    assert var_7 is True
    var_8 = var_1.validate(var_0, var_5)
    assert var_8 is False
    var_9 = module_0.TimedSerializer(var_2, serializer_kwargs=var_4, signer_kwargs=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_9.secret_keys == [b'RHxOLlZrIzB3QF55PEJcXz9GcC4']
    assert var_9.salt == b'itsdangerous'
    assert var_9.is_text_serializer is True
    assert var_9.signer_kwargs == {}
    assert var_9.fallback_signers == []
    assert var_9.serializer_kwargs == {}
    var_10 = module_0.TimedSerializer(var_6, fallback_signers=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_10.secret_keys == [b'D|N.Vk#0w@^y<B\\_?Fp..ams4Jg.dWnH1lmXVijDvCG2xeNJVnEOSV4']
    assert var_10.salt == b'itsdangerous'
    assert var_10.is_text_serializer is True
    assert var_10.signer_kwargs == {}
    assert var_10.fallback_signers == []
    assert var_10.serializer_kwargs == {}
    var_11 = var_1.get_timestamp()
    assert var_11 == 1785411622
    var_12 = var_10.loads_unsafe(var_6)
    var_13 = var_10.dumps(var_4)
    assert var_13 == 'null.ams4Jg.Nvhy48tR1xY7GwTI5L8AkumUCXg'
    var_14 = var_10.loads(var_13, return_timestamp=var_13)
    var_14.dump_payload(var_14)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'D|N.Vk#0w@^y<B\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vk#0w@^y<B\\_?Fp.']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.derive_key()
    assert var_2 == b'1]\xect\xfb\x8c\xd4\x00;\\r([!NY\xa5+\x0e\xed'
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.sign(var_0)
    assert var_5 == b'D|N.Vk#0w@^y<B\\_?Fp..ams4Jg.dWnH1lmXVijDvCG2xeNJVnEOSV4'
    var_6 = var_1.validate(var_5, var_3)
    assert var_6 is True
    var_7 = var_1.verify_signature(var_0, var_4)
    assert var_7 is False
    var_8 = module_0.TimedSerializer(var_2, serializer_kwargs=var_4, signer_kwargs=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_8.secret_keys == [b'1]\xect\xfb\x8c\xd4\x00;\\r([!NY\xa5+\x0e\xed']
    assert var_8.salt == b'itsdangerous'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert var_8.fallback_signers == []
    assert var_8.serializer_kwargs == {}
    var_9 = b'\xbd\xad\xd2B0\\\xbc\x9f\x07n\x9a3'
    var_10 = module_0.TimedSerializer(var_9, fallback_signers=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_10.secret_keys == [b'\xbd\xad\xd2B0\\\xbc\x9f\x07n\x9a3']
    assert var_10.salt == b'itsdangerous'
    assert var_10.is_text_serializer is True
    assert var_10.signer_kwargs == {}
    assert var_10.fallback_signers == []
    assert var_10.serializer_kwargs == {}
    var_11 = var_1.validate(var_2)
    assert var_11 is False
    var_12 = var_10.loads_unsafe(var_5)
    var_13 = var_1.get_timestamp()
    assert var_13 == 1785411622
    var_14 = module_2.Serializer(var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_14.secret_keys == [b'\xbd\xad\xd2B0\\\xbc\x9f\x07n\x9a3']
    assert var_14.salt == b'itsdangerous'
    assert var_14.is_text_serializer is True
    assert var_14.signer_kwargs == {}
    assert var_14.fallback_signers == []
    assert var_14.serializer_kwargs == {}
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.Serializer.default_fallback_signers == []
    assert f'{type(module_2.Serializer.secret_key).__module__}.{type(module_2.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_15 = var_14.dumps(var_4)
    assert var_15 == 'null.Q88BWiq3DLU7T9kItQ1orjdPBCo'
    var_10.loads(var_15)