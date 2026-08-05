# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import src.itsdangerous.serializer as module_1

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
    assert var_3 == 'null.amtlDQ.u6Jj8vqZkGDJtpzmnz1FC3hWcEs'
    var_4 = var_1.loads_unsafe(var_3)

def test_case_4():
    var_0 = None
    var_1 = b'\xbd\x9a[\xbf\xfeQj05\xfc\x17\xcc\x81\\z'
    var_2 = None
    var_3 = module_0.TimestampSigner(var_1, digest_method=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_3.secret_keys == [b'\xbd\x9a[\xbf\xfeQj05\xfc\x17\xcc\x81\\z']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = None
    var_5 = None
    var_6 = b'=\xa4\x98\x1f%\x89t\xf5\x80\x83'
    var_7 = module_0.TimedSerializer(var_6, serializer=var_4, fallback_signers=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'=\xa4\x98\x1f%\x89t\xf5\x80\x83']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_3.validate(var_6, var_5)
    assert var_8 is False

@pytest.mark.xfail(strict=True)
def test_case_5():
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

def test_case_6():
    var_0 = b"\xf5\x1d\x10<\x0c\x19/\xf2\xaa'\xe7~\xe9\x9e^"
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b"\xf5\x1d\x10<\x0c\x19/\xf2\xaa'\xe7~\xe9\x9e^"]
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.__str__()
    var_3 = var_1.loads_unsafe(var_2)

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
    assert var_3 == 'null.amtlDQ.na7L9wJG1w47blmWj_h2PBTALtA'
    var_4 = var_2.loads(var_3, return_timestamp=var_3)

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
    var_3 = var_1.dumps(var_2, var_0)
    assert var_3 == 'null.amtlDQ.nzZuSGjfY7Cr61J2t-DLqsN7KK8'
    var_4 = var_1.loads_unsafe(var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    assert var_3 == 'null.amtlDQ.rPFUSAGNZEH3tk6QCA8oihNVsyA'
    var_1.loads_unsafe(var_3, var_3)

def test_case_10():
    var_0 = b'\xce%\x00^\x9b\x953v'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'\xce%\x00^\x9b\x953v']
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
    assert var_3 == 'null.amtlDQ.I-BrLdYA3Ih64wPUJgteGDT8QVw'
    var_4 = module_0.TimedSerializer(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'null.amtlDQ.I-BrLdYA3Ih64wPUJgteGDT8QVw']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = False
    var_6 = var_1.loads_unsafe(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = b'secret-key'
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
    var_2 = b'hello-world'
    var_3 = b'.'
    var_4 = var_1.sign(var_2)
    assert var_4 == b'hello-world.amtlDQ.VO_Nx5Pp_xa-0yK7yCnfYX2sUlo'
    var_5 = var_1.unsign(var_4)
    assert var_5 == b'hello-world'
    var_6 = var_1.sign(var_2)
    assert var_6 == b'hello-world.amtlDQ.VO_Nx5Pp_xa-0yK7yCnfYX2sUlo'
    var_7 = 50
    var_8 = var_1.unsign(var_6, var_7)
    assert var_8 == b'hello-world'
    var_9 = str(var_7)
    var_10 = 100
    var_1.unsign(var_3, var_10)

def test_case_12():
    var_0 = b'L\xa3\xf8\xee'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'L\xa3\xf8\xee']
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
    var_3 = None
    var_4 = var_1.dumps(var_3)
    assert var_4 == 'null.amtlDQ.gKragNou_oWYjpTguSK7u4Uy9xg'
    var_5 = module_0.TimedSerializer(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'null.amtlDQ.gKragNou_oWYjpTguSK7u4Uy9xg']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = var_1.__str__()
    var_7 = var_1.loads_unsafe(var_4)
    var_8 = var_5.loads_unsafe(var_6, var_3)
    var_9 = False
    var_10 = var_5.loads_unsafe(var_4)
    var_11 = var_5.loads_unsafe(var_7)
    var_12 = var_5.loads_unsafe(var_0)
    var_13 = var_1.loads_unsafe(var_4, var_9)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = b'!e'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'!e']
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
    var_3 = module_1.Serializer(var_0, serializer=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'!e']
    assert var_3.salt == b'itsdangerous'
    assert f'{type(var_3.serializer).__module__}.{type(var_3.serializer).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.Serializer.default_fallback_signers == []
    assert f'{type(module_1.Serializer.secret_key).__module__}.{type(module_1.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.dumps(var_2)
    assert var_4 == 'null.amtlDQ.OpG6oHN-8Hjkc5N0pQDVIV5XK5A.X1UcQUiQc7Y5Wxjb6UBCSJ8WAM8'
    var_5 = module_0.TimedSerializer(var_4, signer=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'null.amtlDQ.OpG6oHN-8Hjkc5N0pQDVIV5XK5A.X1UcQUiQc7Y5Wxjb6UBCSJ8WAM8']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer == 'null.amtlDQ.OpG6oHN-8Hjkc5N0pQDVIV5XK5A.X1UcQUiQc7Y5Wxjb6UBCSJ8WAM8'
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = var_4.__str__()
    assert var_6 == 'null.amtlDQ.OpG6oHN-8Hjkc5N0pQDVIV5XK5A.X1UcQUiQc7Y5Wxjb6UBCSJ8WAM8'
    var_7 = var_1.loads_unsafe(var_4)
    var_1.loads_unsafe(var_2)