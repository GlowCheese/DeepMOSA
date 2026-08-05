# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.timed as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.serializer as module_2

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
    assert var_2 == b'secret.amqT-A.1WY7ONV5Y0qynudKC2ySc70VWIM'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'secret'

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
    assert var_2 == b'secret.amqT-A.1WY7ONV5Y0qynudKC2ySc70VWIM'
    var_3 = var_2 + var_2
    var_1.unsign(var_3)

def test_case_5():
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
def test_case_6():
    var_0 = 'secret'
    var_1 = None
    var_2 = {}
    var_3 = module_0.TimedSerializer(var_0, var_1, signer_kwargs=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_3.secret_keys == [b'secret']
    assert var_3.salt is None
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.loads_unsafe(var_0)
    var_5 = module_0.TimestampSigner(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_5.secret_keys == [b'secret']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = var_5.sign(var_0)
    assert var_6 == b'secret.amqT-A.1WY7ONV5Y0qynudKC2ySc70VWIM'
    var_7 = var_5.unsign(var_6)
    assert var_7 == b'secret'
    var_5.validate(var_1)

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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret.amqT-A.1WY7ONV5Y0qynudKC2ySc70VWIM'
    var_3 = 3600
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret'

def test_case_9():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0, sep=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'secret']
    assert var_1.sep == b'secret'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'testsecretamqT-AsecretDlEc9qiUHgfRMLZx5p5Um418HNQ'
    var_4 = None
    var_5 = var_1.unsign(var_3, var_4, var_1)

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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    assert var_3 == b'test.amqT-A.EvCJL1WEd4ivB8iU0pX08TkhKDk'
    var_4 = -3000
    var_1.unsign(var_3, var_4)

def test_case_11():
    var_0 = 'secret'
    var_1 = None
    var_2 = {}
    var_3 = module_0.TimedSerializer(var_0, var_1, signer_kwargs=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_3.secret_keys == [b'secret']
    assert var_3.salt is None
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.loads_unsafe(var_0)
    var_5 = module_0.TimestampSigner(var_0, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_5.secret_keys == [b'secret']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_6 = module_0.TimestampSigner(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_6.secret_keys == [b'secret']
    assert var_6.sep == b'.'
    assert var_6.salt == b'itsdangerous.Signer'
    assert var_6.key_derivation == 'django-concat'
    assert f'{type(var_6.algorithm).__module__}.{type(var_6.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_7 = var_6.sign(var_0)
    assert var_7 == b'secret.amqT-A.1WY7ONV5Y0qynudKC2ySc70VWIM'
    var_8 = var_5.get_timestamp()
    assert var_8 == 1785369592
    var_9 = -2347
    with pytest.raises(module_1.SignatureExpired):
        var_3.loads(var_7, var_9)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, var_1, signer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dumps(var_1)
    assert var_3 == 'null.amqT-A.Yrz-Kawgl_J6NUxkWm0WIP35vYY'
    var_4 = var_2.loads_unsafe(var_3)
    var_5 = module_0.TimestampSigner(var_0, digest_method=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_5.secret_keys == [b'secret']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous.Signer'
    assert var_5.key_derivation == 'django-concat'
    assert var_5.digest_method == 'secret'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_3.loads(var_3)

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
    var_2 = var_1.sep
    var_3 = var_2 + var_2
    var_4 = b'bad_timestamp'
    var_5 = var_3 + var_4
    var_6 = var_5 + var_2
    var_1.unsign(var_6)

def test_case_14():
    var_0 = ')?>[gM.'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, var_1, signer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b')?>[gM.']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dumps(var_1)
    assert var_3 == 'null.amqT-A.EVdquZwOSZQiSsbQn0KliCCPpmc'
    var_4 = var_2.loads_unsafe(var_0, salt=var_3)
    var_5 = var_2.dumps(var_3)
    assert var_5 == '"null.amqT-A.EVdquZwOSZQiSsbQn0KliCCPpmc".amqT-A.Ss5B_6E34vtNmakr-UXWO9N0P2w'
    var_6 = var_2.loads_unsafe(var_3, salt=var_3)
    var_7 = module_0.TimestampSigner(var_0, var_1, var_0, algorithm=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_7.secret_keys == [b')?>[gM.']
    assert var_7.sep == b')?>[gM.'
    assert var_7.salt == b'itsdangerous.Signer'
    assert var_7.key_derivation == 'django-concat'
    assert f'{type(var_7.algorithm).__module__}.{type(var_7.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_8 = var_7.sign(var_3)
    assert var_8 == b'null.amqT-A.EVdquZwOSZQiSsbQn0KliCCPpmc)?>[gM.amqT-A)?>[gM.7U-OyjlRSNgl4FZWFXr4S_lLL3o'
    var_9 = 42
    var_10 = var_2.loads(var_3, var_9, var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = ')?>[gM.'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1, var_0, algorithm=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_2.secret_keys == [b')?>[gM.']
    assert var_2.sep == b')?>[gM.'
    assert var_2.salt == b'itsdangerous.Signer'
    assert var_2.key_derivation == 'django-concat'
    assert f'{type(var_2.algorithm).__module__}.{type(var_2.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.sign(var_0)
    assert var_3 == b')?>[gM.)?>[gM.amqT-A)?>[gM.Xl5RIHypPreR9_m5KuOTdrUp9uo'
    var_4 = module_0.TimestampSigner(var_0, algorithm=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_4.secret_keys == [b')?>[gM.']
    assert var_4.sep == b'.'
    assert var_4.salt == b'itsdangerous.Signer'
    assert var_4.key_derivation == 'django-concat'
    assert f'{type(var_4.algorithm).__module__}.{type(var_4.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    var_4.unsign(var_3)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'Re?m.<O?%jt58uv'
    var_1 = None
    var_2 = module_0.TimedSerializer(var_0, serializer_kwargs=var_1, signer=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_2.secret_keys == [b'Re?m.<O?%jt58uv']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = module_2.Serializer(var_0, fallback_signers=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'Re?m.<O?%jt58uv']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.Serializer.default_fallback_signers == []
    assert f'{type(module_2.Serializer.secret_key).__module__}.{type(module_2.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.dumps(var_1)
    assert var_4 == 'null.b7syg6c4F4FPerLvb9fA50zW-GY'
    var_5 = var_2.loads_unsafe(var_4)
    var_4.loads_unsafe(var_1, salt=var_1)