# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import datetime as module_1
import src.itsdangerous.signer as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '@.t>\x0crg>EjAZBM |'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_2.secret_keys == [b'@.t>\x0crg>EjAZBM |']
    assert var_2.sep == b'@.t>\x0crg>EjAZBM |'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2.unsign(var_0, return_timestamp=var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = b'payload.!!!!'
    var_1 = None
    var_2 = None
    var_3 = module_0.TimedSerializer(var_0, var_2, serializer_kwargs=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_3.secret_keys == [b'payload.!!!!']
    assert var_3.salt is None
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3.loads_unsafe(var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'r[ISq\n9~+w0h'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, serializer=var_1, serializer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'r[ISq\n9~+w0h']
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

def test_case_3():
    pass

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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amp-qw.EkNdRW3BkQNsQIeUsi6b5WtDEII'
    var_4 = -1
    var_1.unsign(var_3, var_4)

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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amp-qw.EkNdRW3BkQNsQIeUsi6b5WtDEII'
    var_4 = var_1.validate(var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
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
    assert var_2 == b'.amp-qw.2zMcfCiorz3ZByiCElovyRp3B_0'
    var_1.validate(var_2, var_2)

def test_case_7():
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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amp-qw.EkNdRW3BkQNsQIeUsi6b5WtDEII'
    var_4 = var_1.validate(var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.validate(var_5)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amp-qw.EkNdRW3BkQNsQIeUsi6b5WtDEII'
    var_4 = b'tampered'
    var_5 = var_3 + var_4
    var_1.unsign(var_5)

@pytest.mark.xfail(strict=True)
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
    var_2 = None
    var_3 = var_1.sign(var_0)
    assert var_3 == b'.amp-qw.2zMcfCiorz3ZByiCElovyRp3B_0'
    var_4 = -2060
    var_5 = var_1.timestamp_to_datetime(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.hour).__module__}.{type(module_1.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.minute).__module__}.{type(module_1.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.second).__module__}.{type(module_1.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.microsecond).__module__}.{type(module_1.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.tzinfo).__module__}.{type(module_1.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.fold).__module__}.{type(module_1.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.min).__module__}.{type(module_1.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.max).__module__}.{type(module_1.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.resolution).__module__}.{type(module_1.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = var_1.validate(var_3, var_2)
    assert var_6 is True
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_7 = module_0.TimedSerializer(var_0, var_2, signer_kwargs=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'']
    assert var_7.salt is None
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_7.loads(var_3, var_2)

@pytest.mark.xfail(strict=True)
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
    var_2 = b'hello'
    var_3 = b'.'
    var_4 = b'not-base64-data!!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = var_7 + var_2
    var_1.unsign(var_8)

@pytest.mark.xfail(strict=True)
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
    var_2 = module_2.Signer(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.sep == b'.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_2.Signer.secret_key).__module__}.{type(module_2.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)
    assert var_4 == b'hello.7KTthSs1fJgtbigPvFpQH1bpoGA'
    var_1.unsign(var_4)

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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'hello.amp-qw.EkNdRW3BkQNsQIeUsi6b5WtDEII'
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'hello'