# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import builtins as module_1
import src.itsdangerous.serializer as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\r\x99\x8aT\x059\xd2'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\r\x99\x8aT\x059\xd2']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1.loads_unsafe(var_1, var_0)

def test_case_1():
    var_0 = b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\r\x99\x8aT\x059\xd2'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\r\x99\x8aT\x059\xd2']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.loads_unsafe(var_0)

def test_case_2():
    pass

def test_case_3():
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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'secret'

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_2 = b'test_value.sep.badtimestamp'
    var_1.unsign(var_2)

def test_case_5():
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
    var_2 = var_1.validate(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'a\x17fA-=\xf7\xc5\xddf\x16\x05\xda5)c\xc7gH'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, sep=var_0, algorithm=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_2.secret_keys == [b'a\x17fA-=\xf7\xc5\xddf\x16\x05\xda5)c\xc7gH']
    assert var_2.sep == b'a\x17fA-=\xf7\xc5\xddf\x16\x05\xda5)c\xc7gH'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.validate(var_1)

def test_case_7():
    var_0 = 'ecret'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'ecret']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'ecret.ampL7A.nbbxUtImUKJxcY01K4nKfA3woSk'
    var_3 = 0
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'ecret'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'sec\n2bre'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'sec\n2bre']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'sec\n2bre.ampL7A.gCdO_V7UNb467kYxBHHLwP62Dgo'
    var_3 = -2
    var_1.unsign(var_2, var_3)

def test_case_9():
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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk'
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = module_0.TimedSerializer(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = None
    var_6 = var_4.dumps(var_5)
    assert var_6 == 'null.ampL7A.bbRaMQlOkWujDeZ_KgBaT62eP24'
    var_7 = var_4.loads(var_6)
    var_8 = module_1.BaseException()

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_2 = b'tes\x9d_\xc8alue.se".badtimestamp'
    var_1.unsign(var_2)

def test_case_11():
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
    var_2 = var_1.get_timestamp()
    assert var_2 == 1785351148
    var_3 = var_1.sign(var_0)
    assert var_3 == b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk'
    var_4 = 0
    var_5 = var_1.verify_signature(var_0, var_0)
    assert var_5 is False
    var_6 = var_1.unsign(var_3, var_4)
    assert var_6 == b'secret'
    var_7 = var_1.sign(var_0)
    assert var_7 == b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk'
    var_8 = var_1.validate(var_0)
    assert var_8 is False
    var_9 = module_0.TimedSerializer(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_9.secret_keys == [b'secret.ampL7A.FL-X9G-fjPTdmBHKTlSMj1mPaIk']
    assert var_9.salt == b'itsdangerous'
    assert var_9.is_text_serializer is True
    assert var_9.signer_kwargs == {}
    assert var_9.fallback_signers == []
    assert var_9.serializer_kwargs == {}
    var_10 = None
    var_11 = var_9.loads_unsafe(var_7, var_10)
    var_12 = var_9.dumps(var_10)
    assert var_12 == 'null.ampL7A.bbRaMQlOkWujDeZ_KgBaT62eP24'
    var_13 = var_9.loads(var_12, var_4, var_12)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '!F%9.X\r._!VHD'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'!F%9.X\r._!VHD']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'!F%9.X\r._!VHD.ampL7A.jafTSLpfINRwsfQma6PPyALtyQA'
    var_3 = 12
    var_4 = None
    var_5 = var_1.verify_signature(var_4, var_0)
    assert var_5 is False
    var_6 = var_1.unsign(var_2, var_3)
    assert var_6 == b'!F%9.X\r._!VHD'
    var_7 = var_1.sign(var_0)
    assert var_7 == b'!F%9.X\r._!VHD.ampL7A.jafTSLpfINRwsfQma6PPyALtyQA'
    var_8 = var_1.validate(var_0)
    assert var_8 is False
    var_9 = 'NH'
    var_10 = var_1.validate(var_9, var_4)
    assert var_10 is False
    var_11 = module_0.TimedSerializer(var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_11.secret_keys == [b'!F%9.X\r._!VHD']
    assert var_11.salt == b'itsdangerous'
    assert var_11.is_text_serializer is True
    assert var_11.signer_kwargs == {}
    assert var_11.fallback_signers == []
    assert var_11.serializer_kwargs == {}
    var_12 = None
    var_13 = var_11.loads_unsafe(var_7)
    var_14 = module_2.Serializer(var_0, serializer_kwargs=var_12, signer_kwargs=var_4)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_14.secret_keys == [b'!F%9.X\r._!VHD']
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
    var_15 = var_14.dumps(var_12)
    assert var_15 == 'null.o5xhWw0Hpno-h7kDXEUdQ_88Wxk'
    var_11.loads(var_15)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = None
    var_2 = '`.ZY\r&%jm<+Qa=S./Cx'
    var_3 = b'\xe2\x9c\xf3\xebl\x9e&'
    var_4 = module_0.TimedSerializer(var_3, signer=var_1, signer_kwargs=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'\xe2\x9c\xf3\xebl\x9e&']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_5 = var_4.loads_unsafe(var_2)
    var_4.load(var_0)