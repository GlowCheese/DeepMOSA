# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import datetime as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '^!)0'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'^!)0']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1.loads(var_0, var_0)

def test_case_1():
    var_0 = 'D|N.Vkq@yy<y\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.Vkq@yy<y\\_?Fp.']
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
    var_1 = '^!)0'
    var_2 = module_0.TimedSerializer(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'^!)0']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.loads(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\x86\xee\x07\xa8\x17\xc0\xb7Z\xf2\x060\x9a;\xf3'
    module_0.TimedSerializer(var_0, serializer=var_0)

def test_case_4():
    var_0 = 'D|N.V[q@yy<y\\_?Fp.'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'D|N.V[q@yy<y\\_?Fp.']
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

def test_case_5():
    var_0 = '^!)0'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'^!)0']
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
    assert var_2 == b'.amtnag.jYt6AbHeBsyVQn6Oy9sj8icwj1Y'
    var_3 = var_1.unsign(var_2, return_timestamp=var_0)
    assert var_3 == b''

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = module_0.TimedSerializer(var_0, var_2, var_3, signer=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'b\xbb\xb5\x90']
    assert var_4.salt is None
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_5 = var_4.loads_unsafe(var_0)
    var_6 = 'H.W[Elzo\x0bCJa|H._}6Y'
    var_7 = module_0.TimestampSigner(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_7.secret_keys == [b'H.W[Elzo\x0bCJa|H._}6Y']
    assert var_7.sep == b'.'
    assert var_7.salt == b'itsdangerous.Signer'
    assert var_7.key_derivation == 'django-concat'
    assert f'{type(var_7.algorithm).__module__}.{type(var_7.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_8 = var_7.validate(var_6)
    assert var_8 is False
    var_9 = module_0.TimedSerializer(var_0, signer=var_1, signer_kwargs=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_9.secret_keys == [b'b\xbb\xb5\x90']
    assert var_9.salt == b'itsdangerous'
    assert var_9.is_text_serializer is True
    assert var_9.signer_kwargs == {}
    assert var_9.fallback_signers == []
    assert var_9.serializer_kwargs == {}
    var_7.validate(var_3, var_1)

def test_case_8():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'b\xbb\xb5\x90']
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
    var_3 = var_1.sign(var_0)
    assert var_3 == b'b\xbb\xb5\x90.amtnag.hed_9gky9NzVSP_OjxsoNiqwnmM'
    var_4 = var_1.validate(var_3, var_2)
    assert var_4 is True

def test_case_9():
    var_0 = 'w1'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'w1']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'w1.amtnag.iwMBJAvqgV93JxzHQsN2LGMoD2A'
    var_3 = var_1.unsign(var_2, return_timestamp=var_0)

def test_case_10():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, var_0, var_1, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'b\xbb\xb5\x90']
    assert var_2.salt == b'b\xbb\xb5\x90'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_0)
    var_4 = None
    var_5 = module_0.TimedSerializer(var_0, serializer_kwargs=var_4, signer_kwargs=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'b\xbb\xb5\x90']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = var_2.dumps(var_1)
    assert var_6 == 'null.amtnag.IYtcoqLAaC19iAUWYUaN7QNvheQ'
    var_7 = var_2.loads(var_6)

def test_case_11():
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
    var_4 = var_2.sign(var_0)
    assert var_4 == b'b\xbb\xb5\x90.amtnag.rsHaMreLFBYHQwSvin42keVttfc'
    var_5 = -691
    var_6 = var_2.validate(var_4, var_5)
    assert var_6 is False

def test_case_12():
    var_0 = b'b\xbb\xb5\x90'
    var_1 = None
    var_2 = 'H.W[Elzo\x0bCJaO|H._}6Y'
    var_3 = module_0.TimestampSigner(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_3.secret_keys == [b'H.W[Elzo\x0bCJaO|H._}6Y']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.validate(var_2)
    assert var_4 is False
    var_5 = None
    var_6 = var_3.sign(var_0)
    assert var_6 == b'b\xbb\xb5\x90.amtnag.rmzpzgPyGGiBK8cqqvy8v_9-_H4'
    var_7 = module_0.TimedSerializer(var_0, serializer_kwargs=var_5, signer_kwargs=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'b\xbb\xb5\x90']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_3.validate(var_6, var_4)
    assert var_8 is True
    var_9 = var_7.dumps(var_1)
    assert var_9 == 'null.amtnag.1Wf38PJIjqWs_rRVdMh8aLx1pTU'
    var_10 = var_7.loads(var_9, return_timestamp=var_8)
    var_11 = 3428
    var_12 = var_3.timestamp_to_datetime(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.hour).__module__}.{type(module_1.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.minute).__module__}.{type(module_1.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.second).__module__}.{type(module_1.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.microsecond).__module__}.{type(module_1.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.tzinfo).__module__}.{type(module_1.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.fold).__module__}.{type(module_1.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.min).__module__}.{type(module_1.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.max).__module__}.{type(module_1.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.resolution).__module__}.{type(module_1.datetime.resolution).__qualname__}' == 'datetime.timedelta'