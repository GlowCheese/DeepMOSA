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
    assert var_2 == b'secret.amqroQ.VQiv5YTYDFmC_XfYRyQKL32Pq_s'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'secret'

def test_case_4():
    var_0 = 'QFH.x\r^-\n@+]uMw+m.6'
    var_1 = module_0.TimestampSigner(var_0, var_0, key_derivation=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'QFH.x\r^-\n@+]uMw+m.6']
    assert var_1.sep == b'.'
    assert var_1.salt == b'QFH.x\r^-\n@+]uMw+m.6'
    assert var_1.key_derivation == 'QFH.x\r^-\n@+]uMw+m.6'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = var_1.validate(var_0)
    assert var_2 is False

def test_case_5():
    var_0 = 'n[0.#g\r.qS$PUTt\x0b'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'n[0.#g\r.qS$PUTt\x0b']
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
    var_0 = 'FDC4v0\x0bX[l'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.verify_signature(var_0, var_0)
    assert var_3 is False
    var_4 = module_0.TimedSerializer(var_0, var_2, signer_kwargs=var_2, fallback_signers=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_4.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_4.salt is None
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = var_4.loads_unsafe(var_0)
    var_6 = module_0.TimedSerializer(var_0, signer_kwargs=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_6.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_6.salt == b'itsdangerous'
    assert var_6.is_text_serializer is True
    assert var_6.signer_kwargs == {}
    assert var_6.fallback_signers == []
    assert var_6.serializer_kwargs == {}
    var_7 = var_6.dumps(var_2)
    assert var_7 == 'null.amqroQ.dtJGf21XFq8NYjxWjyK_-LJ2eL8'
    var_1.validate(var_2)

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
    var_0 = b'\x0b\xdf\xea\x91 \x84\x19\x1e\x7f;\r\x99\xbdT\x059S'
    var_1 = module_0.TimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_1.secret_keys == [b'\x0b\xdf\xea\x91 \x84\x19\x1e\x7f;\r\x99\xbdT\x059S']
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
    assert var_3 == 'null.amqroQ.mOJk99aIuMXNjdHlhLuyw99Riz8'
    var_4 = var_1.loads_unsafe(var_3)

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
    assert var_2 == b'secret.amqroQ.VQiv5YTYDFmC_XfYRyQKL32Pq_s'
    var_3 = 100
    var_4 = var_1.unsign(var_2, var_3)
    assert var_4 == b'secret'

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
    var_2 = var_1.sign(var_0)
    assert var_2 == b'secret.amqroQ.VQiv5YTYDFmC_XfYRyQKL32Pq_s'
    var_3 = -1
    var_1.unsign(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'FDC4v0\x0bX[l'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.get_signature(var_2)
    assert var_3 == b'ibYlJFyg-vEjPqAadvOYP0KNQ8Y'
    var_4 = var_1.validate(var_0)
    assert var_4 is False
    var_5 = module_0.TimedSerializer(var_0, var_2, signer_kwargs=var_2, fallback_signers=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_5.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_5.salt is None
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = 'iNY=X7/r'
    var_7 = var_5.loads_unsafe(var_0)
    var_8 = var_1.sign(var_6)
    assert var_8 == b'iNY=X7/r.amqroQ.i458VrrYIhc-cEpbKfbjjhE3Vt4'
    var_9 = module_0.TimedSerializer(var_0, signer_kwargs=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_9.secret_keys == [b'FDC4v0\x0bX[l']
    assert var_9.salt == b'itsdangerous'
    assert var_9.is_text_serializer is True
    assert var_9.signer_kwargs == {}
    assert var_9.fallback_signers == []
    assert var_9.serializer_kwargs == {}
    var_10 = var_9.dumps(var_2)
    assert var_10 == 'null.amqroQ.dtJGf21XFq8NYjxWjyK_-LJ2eL8'
    var_11 = var_9.loads(var_10)
    var_12 = True
    var_13 = var_9.loads(var_10, return_timestamp=var_12)
    var_10.validate(var_11)

def test_case_12():
    var_0 = '2[0.#g\r.qS\x0bPUTt\x0b'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'2[0.#g\r.qS\x0bPUTt\x0b']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = var_1.derive_key()
    assert var_4 == b'\xf3\xfa\xcdRZH\xba\x89\xea\xb3\xe4\xe04\xab\\Mv\x9cU2'
    var_5 = var_1.get_timestamp()
    assert var_5 == 1785375649
    var_6 = '\x0cPhg\rDE*s}P'
    var_7 = module_0.TimedSerializer(var_6, signer=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_7.secret_keys == [b'\x0cPhg\rDE*s}P']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == {}
    var_8 = var_1.get_timestamp()
    assert var_8 == 1785375649
    var_9 = False
    var_10 = var_7.dumps(var_3)
    assert var_10 == 'false.amqroQ.CDzv8iSPtak-xMI2O9vdSvX37-k'
    var_11 = 'iNY/r'
    var_12 = var_7.loads_unsafe(var_11, salt=var_2)
    var_13 = var_1.sign(var_6)
    assert var_13 == b'\x0cPhg\rDE*s}P.amqroQ.K7I9iGBpkF35meFkPseOciPhsDA'
    var_14 = var_7.dumps(var_2)
    assert var_14 == 'null.amqroQ.mUeoz-cM1M5x_4ZcCi4xDjh4iGg'
    var_15 = False
    var_16 = var_7.loads(var_14, var_9, var_15)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'n[0.#g\r.qS$PUTt\x0b'
    var_1 = module_0.TimestampSigner(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.timed.TimestampSigner'
    assert var_1.secret_keys == [b'n[0.#g\r.qS$PUTt\x0b']
    assert var_1.sep == b'.'
    assert var_1.salt == b'itsdangerous.Signer'
    assert var_1.key_derivation == 'django-concat'
    assert f'{type(var_1.algorithm).__module__}.{type(var_1.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = None
    var_3 = var_1.validate(var_0)
    assert var_3 is False
    var_4 = module_1.Serializer(var_0, var_2, signer=var_2, signer_kwargs=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'n[0.#g\r.qS$PUTt\x0b']
    assert var_4.salt is None
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.Serializer.default_fallback_signers == []
    assert f'{type(module_1.Serializer.secret_key).__module__}.{type(module_1.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.dumps(var_2)
    assert var_5 == 'null.2vRL1lla7or0hfWJmGcOj5rTbh0'
    var_6 = var_1.validate(var_5)
    assert var_6 is False
    var_7 = '\x0cPhg\rDE*s}P'
    var_8 = module_0.TimedSerializer(var_7, serializer_kwargs=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.timed.TimedSerializer'
    assert var_8.secret_keys == [b'\x0cPhg\rDE*s}P']
    assert var_8.salt == b'itsdangerous'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert var_8.fallback_signers == []
    assert var_8.serializer_kwargs == {}
    var_8.loads(var_5, **var_5)