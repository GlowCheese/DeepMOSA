# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.signer as module_2
import re as module_3
import json as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._PDataSerializer()

def test_case_1():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'secret']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2.loads(var_0, var_0)

def test_case_3():
    var_0 = 'scret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'scret']
    assert var_1.salt == b'scret'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = module_0.is_text_serializer(var_1)
    assert var_2 is True

def test_case_4():
    var_0 = 'scret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'scret']
    assert var_1.salt == b'scret'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'scret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'scret']
    assert var_1.salt == b'scret'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = None
    var_1.loads_unsafe(var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = None
    var_2.load(var_3)

def test_case_7():
    var_0 = 'scret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'scret']
    assert var_1.salt == b'scret'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = None
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5Q_'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5Q_']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_1, var_0)
    var_4 = None
    var_2.load_unsafe(var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = b"O\xd5{\xe7\x85&'\x88\xf3\xc6\x9b."
    var_1 = module_0.Serializer(var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b"O\xd5{\xe7\x85&'\x88\xf3\xc6\x9b."]
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b"O\xd5{\xe7\x85&'\x88\xf3\xc6\x9b."
    assert var_1.serializer_kwargs == b"O\xd5{\xe7\x85&'\x88\xf3\xc6\x9b."
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_1.loads_unsafe(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = b'k\xb8\x15\x1a/0O\xc6U\xc6\x9b\x1b\x91\xd6\x10in'
    var_2 = b'\x11\x17\xd6T\x98oS\xdf\xe9\xd0\xcaG'
    var_3 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    module_0.Serializer(var_3, var_2, var_1, signer=var_0, fallback_signers=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_1, var_0)
    var_4 = None
    var_2.dump(var_4, var_4)

def test_case_12():
    var_0 = b'key1'
    var_1 = [var_0]
    var_2 = module_0.Serializer(var_0, var_0, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'key1']
    assert var_2.salt == b'key1'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == [b'key1']
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.iter_unsigners()
    with pytest.raises(TypeError):
        var_4 = list(var_3)

def test_case_13():
    var_0 = b"O\xd5{\x03\xe7\x85&'\x88\xf3\xc6\x9b."
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer_kwargs=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b"O\xd5{\x03\xe7\x85&'\x88\xf3\xc6\x9b."]
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_0, var_1)

def test_case_14():
    var_0 = 'secret'
    var_1 = 'some'
    var_2 = {var_1: var_1}
    var_3 = module_0.Serializer(var_0, signer_kwargs=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'secret']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {'some': 'some'}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = b'\x16\x9f\x90\x94\xc8\xb0f'
    var_2.load_payload(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, serializer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    var_4 = var_3.dumps(var_0, var_0)
    assert var_4 == 'null.vgNNipWvQUjLRBAFhoazTxri7Xo'
    var_5 = module_0.Serializer(var_4, signer=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_5.secret_keys == [b'null.vgNNipWvQUjLRBAFhoazTxri7Xo']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = var_3.loads_unsafe(var_1, var_0)
    var_7 = var_3.make_signer(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_7.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_7.sep == b'.'
    assert var_7.salt == b'itsdangerous'
    assert var_7.key_derivation == 'django-concat'
    assert f'{type(var_7.algorithm).__module__}.{type(var_7.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_2.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_2.Signer.secret_key).__module__}.{type(module_2.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_8 = None
    var_9 = module_3.RegexFlag.LOCALE
    var_10 = None
    var_11 = module_4.dumps(var_8, skipkeys=var_10, allow_nan=var_0, default=var_10)
    assert var_11 == 'null'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_9.seek(var_9)

def test_case_17():
    var_0 = b'key1'
    var_1 = 'extra'
    var_2 = {var_1: var_1}
    var_3 = [var_2]
    var_4 = module_0.Serializer(var_0, var_0, fallback_signers=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'key1']
    assert var_4.salt == b'key1'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == [{'extra': 'extra'}]
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.iter_unsigners()
    with pytest.raises(TypeError):
        var_6 = list(var_5)

def test_case_18():
    var_0 = b'key1'
    var_1 = {}
    var_2 = [var_1]
    var_3 = module_0.Serializer(var_0, var_0, fallback_signers=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'key1']
    assert var_3.salt == b'key1'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == [{}]
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2