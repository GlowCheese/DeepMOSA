# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x060\x9a;\xf3'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x060\x9a;\xf3']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.loads_unsafe(var_0, salt=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = b'~A\xa8\xb9t\xd7'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'~A\xa8\xb9t\xd7']
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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x060\x9a;\xf3'
    module_0.TimedSerializer(var_0, serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'sebret'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'sebret']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'hello.notbae64!.signature'
    var_1.unsign(var_2)

def test_case_4():
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
    assert var_2 == b'secret.amrFpw.fOUJy2d-NsDMFs9rQA7LTMoV2kU'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'secret'

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    assert var_2 == b'.amrFpw.YhpB5tRYQ0wWxWgoKEfaw4VKJ5g'
    var_1.validate(var_2, var_1)

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
    var_2 = var_1.validate(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '.BZr8M.m!Ri1Oj\x0b#}4k'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'.BZr8M.m!Ri1Oj\x0b#}4k']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1.unsign(var_0)

def test_case_8():
    var_0 = b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x06\x9a;\xf3'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x06\x9a;\xf3']
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
    assert var_3 == 'null.amrFpw.Ak03_qmEj8kyMFem2fQcOXhCznA'
    var_4 = var_1.loads(var_3)

def test_case_9():
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
    assert var_2 == b'.amrFpw.YhpB5tRYQ0wWxWgoKEfaw4VKJ5g'
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = var_1.validate(var_2, var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '.k2BZr8M.m!Ti1O\x0b#}4k'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, serializer_kwargs=var_1, signer_kwargs=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'.k2BZr8M.m!Ti1O\x0b#}4k']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = module_0.TimestampSigner(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_3.secret_keys == [b'.k2BZr8M.m!Ti1O\x0b#}4k']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_4 = -5023
    var_5 = var_3.sign(var_0)
    assert var_5 == b'.k2BZr8M.m!Ti1O\x0b#}4k.amrFpw.9Ypi-u0AiagP2t3GjU9464QWGjM'
    var_6 = var_3.validate(var_0, var_1)
    assert var_6 is False
    var_7 = var_3.validate(var_5, var_6)
    assert var_7 is True
    var_8 = var_3.unsign(var_5)
    assert var_8 == b'.k2BZr8M.m!Ti1O\x0b#}4k'
    var_9 = b't\xe3\xee\xeaWRu\xdc$\xf9A\xe8j7A:\xfc'
    var_10 = var_3.validate(var_0)
    assert var_10 is False
    var_11 = var_3.sign(var_0)
    assert var_11 == b'.k2BZr8M.m!Ti1O\x0b#}4k.amrFpw.9Ypi-u0AiagP2t3GjU9464QWGjM'
    var_12 = var_2.iter_unsigners()
    var_13 = var_2.loads_unsafe(var_9)
    var_14 = module_0.TimedSerializer(var_9, var_1, signer=var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_14.secret_keys == [b't\xe3\xee\xeaWRu\xdc$\xf9A\xe8j7A:\xfc']
    assert var_14.salt is None
    assert var_14.is_text_serializer is True
    assert var_14.signer_kwargs == {}
    assert var_14.fallback_signers == []
    assert var_14.serializer_kwargs == {}
    var_15 = var_14.dumps(var_1)
    assert var_15 == 'null.amrFpw.l2gM_oz6TAMgVg-XCENWhyJHuQk'
    var_14.loads(var_15, var_4)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '.BZr8M.m!Ri1Oj\x0b#}4k'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, serializer_kwargs=var_1, signer_kwargs=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'.BZr8M.m!Ri1Oj\x0b#}4k']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = module_0.TimestampSigner(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_3.secret_keys == [b'.BZr8M.m!Ri1Oj\x0b#}4k']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_4 = None
    var_5 = var_2.dumps(var_4)
    assert var_5 == 'null.amrFpw.N1u0WxsQ2Bf6s30XDZh3UV6RaL8'
    var_6 = var_3.sign(var_0)
    assert var_6 == b'.BZr8M.m!Ri1Oj\x0b#}4k.amrFpw.YrMQkNkBvwNUwZVNGu6IdkXRCC4'
    var_7 = var_3.validate(var_0, var_4)
    assert var_7 is False
    var_8 = var_3.validate(var_6, var_7)
    assert var_8 is True
    var_9 = b'\xf3\xf8\xbc\xc1pk\xe0\xa0\xc0\x18'
    var_10 = var_3.validate(var_9, var_4)
    assert var_10 is False
    var_11 = var_2.loads(var_5, return_timestamp=var_5)
    var_12 = var_3.sign(var_0)
    assert var_12 == b'.BZr8M.m!Ri1Oj\x0b#}4k.amrFpw.YrMQkNkBvwNUwZVNGu6IdkXRCC4'
    var_2.loads_unsafe(var_4, var_1)