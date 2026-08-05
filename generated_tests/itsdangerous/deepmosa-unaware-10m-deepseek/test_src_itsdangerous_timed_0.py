# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = b'`\xdf\x12\x91 \x84L\x19\x1e\x7f;\r\x99\x8aT\x059S'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'`\xdf\x12\x91 \x84L\x19\x1e\x7f;\r\x99\x8aT\x059S']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1.loads_unsafe(var_1)

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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.TimedSerializer(var_0, var_0, serializer_kwargs=var_0)

def test_case_3():
    var_0 = b'`\xdf\x12 \x84L\x19\x1e\x7f;\r\x99\x8aT\x059S'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'`\xdf\x12 \x84L\x19\x1e\x7f;\r\x99\x8aT\x059S']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'null.ams1yA.crl9RpwMK9-l8uNDXA4XBTa2WZc'
    var_4 = var_1.loads_unsafe(var_3)

def test_case_4():
    var_0 = 'x.k&*Xj8Z2K. r,0'
    var_1 = module_0.TimestampSigner(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'x.k&*Xj8Z2K. r,0']
    assert var_1.sep == b'.'
    assert var_1.salt == b'x.k&*Xj8Z2K. r,0'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'x.k&**Xj8Z2KNi. r,0'
    var_1 = module_0.TimestampSigner(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'x.k&**Xj8Z2KNi. r,0']
    assert var_1.sep == b'.'
    assert var_1.salt == b'x.k&**Xj8Z2KNi. r,0'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b't\x08st-value'
    var_3 = var_1.sign(var_2)
    assert var_3 == b't\x08st-value.ams1yA.NGKgemwCWqL7kp5ZFc2vbzCYlw0'
    var_4 = var_1.validate(var_2, var_1)
    assert var_4 is False
    var_5 = module_0.TimedSerializer(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b't\x08st-value.ams1yA.NGKgemwCWqL7kp5ZFc2vbzCYlw0']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = var_1.validate(var_3, var_4)
    assert var_6 is True
    var_7 = var_5.loads_unsafe(var_3)
    var_8 = var_1.validate(var_0)
    assert var_8 is False
    var_9 = None
    var_1.validate(var_9)

def test_case_6():
    var_0 = b';\\O \xc9[[\x96\\\x1f\xb7c._\xf2'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b';\\O \xc9[[\x96\\\x1f\xb7c._\xf2']
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

def test_case_7():
    var_0 = b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\x99\x8aT\x059\xd2'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, serializer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f;\x99\x8aT\x059\xd2']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dumps(var_1)
    assert var_3 == 'null.ams1yA.g0jqyb_2S_6kGsTJgseqygyKi1w'
    var_4 = var_2.loads(var_3, return_timestamp=var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f\x12\x99\x8aT\x059\xd2'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'`\xdf\x12\x91\x14 \xc8\x84L\x19\x1e\x7f\x12\x99\x8aT\x059\xd2']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'null.ams1yA.I0iwQEg8U0vmyflum-JpU_GXfbQ'
    var_1.loads_unsafe(var_3, var_3)

def test_case_9():
    var_0 = 'tst_jvale'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'tst_jvale']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'tst_jvale.ams1yA.1griX0TLnxBMua434UTkDf6wAZg'
    var_3 = 500
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'tst_jvale'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_2.secret_keys == [b'secret-key']
    assert var_2.sep == b'.'
    assert var_2.salt == b'test-salt'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    assert var_4 == b'test-value.ams1yA.tre9oGSnRwuss59qlnC3u8urNFk'
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = var_2.derive_key()
    assert var_6 == b"n\xa3\x9a:\xd4\xe0)2\xd8.}x9\xdb\x93\xc4\xf2'\xd0\\"
    var_7 = var_2.sign(var_3)
    assert var_7 == b'test-value.ams1yA.tre9oGSnRwuss59qlnC3u8urNFk'
    var_8 = 3600
    var_9 = var_2.unsign(var_7, var_8)
    assert var_9 == b'test-value'
    var_10 = var_2.get_timestamp
    var_11 = var_2.sign(var_3)
    assert var_11 == b'test-value.ams1yA.tre9oGSnRwuss59qlnC3u8urNFk'
    var_12 = -9
    var_2.unsign(var_11, var_12)

def test_case_11():
    var_0 = 'x.k&*Xj8Z2. r,0'
    var_1 = module_0.TimestampSigner(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'x.k&*Xj8Z2. r,0']
    assert var_1.sep == b'.'
    assert var_1.salt == b'x.k&*Xj8Z2. r,0'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False