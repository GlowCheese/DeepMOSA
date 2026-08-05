# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import datetime as module_1

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
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
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
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = -40
    var_1.unsign(var_2, var_3)

def test_case_5():
    var_0 = 'i{0Odx<(;\x0b D*QbTx'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'i{0Odx<(;\x0b D*QbTx']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amrDSQ.IB30A_QFbsgJzvEcIG9eqToUj2U'
    var_4 = var_1.validate(var_2)
    assert var_4 is False

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
    var_0 = b'\xc3\xddM\x14.u\xc6\xb2'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'\xc3\xddM\x14.u\xc6\xb2']
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

def test_case_8():
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
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amrDSQ.XnzgjjREDTbwXP9_jJNirXbRhgM'
    var_4 = -5
    var_5 = var_3[:var_4]
    var_6 = var_1.validate(var_5)
    assert var_6 is False

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
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = 0
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret'

def test_case_10():
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
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '!n$8q.Y$.\rLYb7M'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'!n$8q.Y$.\rLYb7M']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1.unsign(var_0, var_0)

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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = 1
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret'
    var_5 = b'\xaf\x16\xc7\r\xd1\xcd\x9eU\xdf\xd5\xb8tO\xa6\xb9'
    var_6 = var_1.validate(var_0)
    assert var_6 is False
    var_7 = module_0.TimedSerializer(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'\xaf\x16\xc7\r\xd1\xcd\x9eU\xdf\xd5\xb8tO\xa6\xb9']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_7.dumps(var_0)
    assert var_8 == '"secret".amrDSQ.Naa4HUyXxT6V57vi5hjaZkuLZx0'
    var_9 = var_7.loads(var_8)
    assert var_9 == 'secret'
    var_7.loads(var_0, **var_8)

@pytest.mark.xfail(strict=True)
def test_case_13():
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
    assert var_2 == b'secret.amrDSQ.B1SshiKRGliUQ8qlRaM9-s0QDeo'
    var_3 = 1
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret'
    var_5 = b'"\x1c\xb75\xc0_\xba\x9a*'
    var_6 = b'\xaf\x16\xc7\r\xd1\xcd\x9eU\xdf\xd5\xb8tO\xa6\xb9'
    var_7 = 2647
    var_8 = var_1.timestamp_to_datetime(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.hour).__module__}.{type(module_1.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.minute).__module__}.{type(module_1.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.second).__module__}.{type(module_1.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.microsecond).__module__}.{type(module_1.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.tzinfo).__module__}.{type(module_1.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.fold).__module__}.{type(module_1.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.min).__module__}.{type(module_1.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.max).__module__}.{type(module_1.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.resolution).__module__}.{type(module_1.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_9 = var_1.validate(var_0)
    assert var_9 is False
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_10 = module_0.TimedSerializer(var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_10.secret_keys == [b'\xaf\x16\xc7\r\xd1\xcd\x9eU\xdf\xd5\xb8tO\xa6\xb9']
    assert var_10.salt == b'itsdangerous'
    assert var_10.is_text_serializer is True
    assert var_10.signer_kwargs == {}
    assert var_10.fallback_signers == []
    assert var_10.serializer_kwargs == {}
    var_11 = var_10.dumps(var_0)
    assert var_11 == '"secret".amrDSQ.Naa4HUyXxT6V57vi5hjaZkuLZx0'
    var_12 = None
    var_13 = var_10.dumps(var_12, var_11)
    assert var_13 == 'null.amrDSQ.Jjd7YRvpnmrL5JmL3sFkZ5USu3s'
    var_14 = var_10.loads(var_11, return_timestamp=var_13)
    var_11.loads(var_5, return_timestamp=var_11)