# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import datetime as module_1
import src.itsdangerous.signer as module_2

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

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
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
    var_2 = b'test.sep.malformed'
    var_1.unsign(var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'se2:rket'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'se2:rket']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'se2:rket.amqt_g.BBIeYt-v4vdfiYrDoiRLPz6OLDo'
    var_1.unsign(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'o5.rCIQKjT.4h%l{'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'o5.rCIQKjT.4h%l{']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA.amqt_g.-o_5yWHeWs4ui8EHBA-Klg-UQcw'
    var_4 = var_1.validate(var_0)
    assert var_4 is False
    var_5 = module_0.TimedSerializer(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = None
    var_7 = 1124
    var_8 = var_1.unsign(var_2, var_7)
    assert var_8 == b'o5.rCIQKjT.4h%l{'
    var_9 = var_1.timestamp_to_datetime(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.hour).__module__}.{type(module_1.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.minute).__module__}.{type(module_1.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.second).__module__}.{type(module_1.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.microsecond).__module__}.{type(module_1.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.tzinfo).__module__}.{type(module_1.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.fold).__module__}.{type(module_1.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.datetime.min).__module__}.{type(module_1.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.max).__module__}.{type(module_1.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_1.datetime.resolution).__module__}.{type(module_1.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_10 = var_5.loads_unsafe(var_3, var_8)
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_11 = var_5.loads_unsafe(var_3)
    var_12 = var_5.iter_unsigners()
    var_13 = b'.\xa6M`#\x9e.\xa2\xbb^:\x84{\xe5-\xfd\xbc\x83'
    var_14 = var_1.validate(var_13)
    assert var_14 is False
    var_15 = False
    var_16 = var_1.timestamp_to_datetime(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'datetime.datetime'
    var_1.validate(var_6)

def test_case_6():
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
    var_2 = b'test'
    var_3 = b'\xc1\xba\x1e\xb3\x19\x80\xa5\x96e$\x82\xe0 \xf0y\xeb9'
    var_4 = var_1.validate(var_3)
    assert var_4 is False
    var_5 = var_1.sign(var_2)
    assert var_5 == b'test.amqt_g.me_li5OFurmF7TVfrnXsFeUv1ec'
    var_6 = 3600
    var_7 = var_1.unsign(var_5, var_6)
    assert var_7 == b'test'

@pytest.mark.xfail(strict=True)
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
    var_2 = b'test.sep.12345.invalid'
    var_1.unsign(var_2)

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
    assert var_3 == 'null.amqt_g.CuDbFw1a-9P7qjhY9DsVnR-Vl8I'
    var_4 = var_1.loads(var_3)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'test.amqt_g.me_li5OFurmF7TVfrnXsFeUv1ec'
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'test.amqt_g.me_li5OFurmF7TVfrnXsFeUv1ec'
    var_4 = -1
    var_1.unsign(var_3, var_4)

def test_case_11():
    var_0 = 'se2:rket'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'se2:rket']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'se2:rket.amqt_g.BBIeYt-v4vdfiYrDoiRLPz6OLDo'
    var_3 = 1078
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'se2:rket'

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
    var_2 = b'hello.1234567890.invalid_signature'
    var_1.unsign(var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'secrAet'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'secrAet']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secrAet.amqt_g.YinoO36WvKjBmd8VmXthbsjpGAU'
    var_3 = module_2.Signer(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_3.secret_keys == [b'secrAet']
    assert var_3.sep == b'.'
    assert var_3.salt == b'itsdangerous.Signer'
    assert var_3.key_derivation == 'django-concat'
    assert f'{type(var_3.algorithm).__module__}.{type(var_3.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_2.Signer.secret_key).__module__}.{type(module_2.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.sign(var_0)
    assert var_4 == b'secrAet.st5BH9IF9-TfTSWrHPtjN9StGb4'
    var_5 = var_1.validate(var_4)
    assert var_5 is False
    var_6 = None
    var_7 = module_0.TimedSerializer(var_4, serializer=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'secrAet.st5BH9IF9-TfTSWrHPtjN9StGb4']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_1.unsign(var_2)
    assert var_8 == b'secrAet'
    var_9 = var_7.loads_unsafe(var_0, var_6)
    var_7.loads_unsafe(var_6, var_0)

def test_case_14():
    var_0 = 'o5.rCIQKjT.4h%l{'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'o5.rCIQKjT.4h%l{']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.sign(var_0)
    assert var_2 == b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA.amqt_g.-o_5yWHeWs4ui8EHBA-Klg-UQcw'
    var_4 = var_1.validate(var_0)
    assert var_4 is False
    var_5 = module_0.TimedSerializer(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = None
    var_7 = 1124
    var_8 = var_1.unsign(var_2, var_7)
    assert var_8 == b'o5.rCIQKjT.4h%l{'
    var_9 = var_5.loads_unsafe(var_3, var_8)
    var_10 = var_5.loads_unsafe(var_3)
    var_11 = var_5.iter_unsigners()
    var_12 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_12.secret_keys == [b'o5.rCIQKjT.4h%l{']
    assert var_12.sep == b'.'
    assert var_12.salt == b'itsdangerous.Signer'
    assert var_12.key_derivation == 'django-concat'
    assert f'{type(var_12.algorithm).__module__}.{type(var_12.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_13 = b'.\xa6M`#\x9e.\xa2\xbb^:\x84{\xe5-\xfd\xbc\x83'
    var_14 = var_1.validate(var_13)
    assert var_14 is False
    var_15 = var_1.validate(var_3)
    assert var_15 is True
    var_16 = None
    var_17 = var_1.unsign(var_2)
    assert var_17 == b'o5.rCIQKjT.4h%l{'
    var_18 = module_0.TimestampSigner(var_2, algorithm=var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_18.secret_keys == [b'o5.rCIQKjT.4h%l{.amqt_g.I9F3kUi7O2WlO9BpUNamIaOFXHA']
    assert var_18.sep == b'.'
    assert var_18.salt == b'itsdangerous.Signer'
    assert var_18.key_derivation == 'django-concat'
    assert f'{type(var_18.algorithm).__module__}.{type(var_18.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_19 = None
    var_20 = var_5.dumps(var_19)
    assert var_20 == 'null.amqt_g.AR7pTjjczge2makFkZAvCnfj9kI'
    var_21 = var_5.loads(var_20, var_14, var_15, var_6)
    var_22 = var_5.loads_unsafe(var_20)