# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import src.itsdangerous.exc as module_1
import builtins as module_2
import src.itsdangerous.signer as module_3
import re as module_4
import json as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._PDataSerializer()

def test_case_1():
    var_0 = 'custiom_salt'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'custiom_salt']
    assert var_1.salt == b'custiom_salt'
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

def test_case_2():
    var_0 = 'scret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'scret']
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
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_1)

def test_case_3():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'secret']
    assert var_1.salt == b'secret'
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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'secret']
    assert var_1.salt == b'secret'
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
    var_0 = 'sec.Lt'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'sec.Lt']
    assert var_1.salt == b'sec.Lt'
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
    var_1.loads_unsafe(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2']
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
    var_2.load(var_1)

def test_case_7():
    var_0 = 'sec.Lt'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'sec.Lt']
    assert var_1.salt == b'sec.Lt'
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
def test_case_8():
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
    var_2 = None
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'null.mzpArLzCx4J5X46ifzkqnvDZKH8'
    var_1.load_unsafe(var_2)

def test_case_9():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'secret']
    assert var_1.salt == b'secret'
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
    var_2 = var_1.loads_unsafe(var_0, var_0)

def test_case_10():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'loads'
    var_3 = 'dumps'
    var_4 = lambda self, x: x
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_2.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = {}
    var_13 = module_0.Serializer(var_11, serializer=var_10, serializer_kwargs=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_13.secret_keys == [b'secret']
    assert var_13.salt == b'itsdangerous'
    assert var_13.is_text_serializer is False
    assert var_13.signer_kwargs == {}
    assert var_13.fallback_signers == []
    assert var_13.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_14 = b'test'
    var_15 = var_13.load_payload(var_14)
    assert var_15 == b'test'

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

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2'
    var_1 = module_0.Serializer(var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2'
    assert var_1.serializer_kwargs == b'\xe8\x9a\xb8\xdc\xaa\xd8C\x98!.\xac\xa2'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_1.loads_unsafe(var_0, var_0)

def test_case_13():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'secret']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {'key_derivation': 'hmac'}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.signer_kwargs

@pytest.mark.xfail(strict=True)
def test_case_14():
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
    var_3 = var_2.dumps(var_0, var_0)
    assert var_3 == 'null.vgNNipWvQUjLRBAFhoazTxri7Xo'
    var_4 = var_2.loads_unsafe(var_1, var_0)
    var_5 = var_2.make_signer(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.signer.Signer'
    assert var_5.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_5.sep == b'.'
    assert var_5.salt == b'itsdangerous'
    assert var_5.key_derivation == 'django-concat'
    assert f'{type(var_5.algorithm).__module__}.{type(var_5.algorithm).__qualname__}' == 'src.itsdangerous.signer.HMACAlgorithm'
    assert module_3.Signer.default_key_derivation == 'django-concat'
    assert f'{type(module_3.Signer.secret_key).__module__}.{type(module_3.Signer.secret_key).__qualname__}' == 'builtins.property'
    var_6 = None
    var_7 = module_0.Serializer(var_3, serializer_kwargs=var_3, signer=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_7.secret_keys == [b'null.vgNNipWvQUjLRBAFhoazTxri7Xo']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer == 'null.vgNNipWvQUjLRBAFhoazTxri7Xo'
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == []
    assert var_7.serializer_kwargs == 'null.vgNNipWvQUjLRBAFhoazTxri7Xo'
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    var_8 = module_4.RegexFlag.LOCALE
    var_9 = None
    var_10 = module_5.dumps(var_6, skipkeys=var_9, allow_nan=var_0, default=var_9)
    assert var_10 == 'null'
    assert module_4.ASCII == module_4.RegexFlag.ASCII
    assert module_4.A == module_4.RegexFlag.ASCII
    assert module_4.IGNORECASE == module_4.RegexFlag.IGNORECASE
    assert module_4.I == module_4.RegexFlag.IGNORECASE
    assert module_4.LOCALE == module_4.RegexFlag.LOCALE
    assert module_4.L == module_4.RegexFlag.LOCALE
    assert module_4.UNICODE == module_4.RegexFlag.UNICODE
    assert module_4.U == module_4.RegexFlag.UNICODE
    assert module_4.MULTILINE == module_4.RegexFlag.MULTILINE
    assert module_4.M == module_4.RegexFlag.MULTILINE
    assert module_4.DOTALL == module_4.RegexFlag.DOTALL
    assert module_4.S == module_4.RegexFlag.DOTALL
    assert module_4.VERBOSE == module_4.RegexFlag.VERBOSE
    assert module_4.X == module_4.RegexFlag.VERBOSE
    assert module_4.TEMPLATE == module_4.RegexFlag.TEMPLATE
    assert module_4.T == module_4.RegexFlag.TEMPLATE
    assert module_4.DEBUG == module_4.RegexFlag.DEBUG
    var_8.seek(var_8)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = b'secret'
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
    var_2 = None
    var_1.load_payload(var_2, var_0)

def test_case_16():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_3.secret_keys == [b'old_key', b'new_key']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'

def test_case_17():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_5.secret_keys == [b'secret-key']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == [{'key': 'value'}]
    assert var_5.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_6 = var_5.iter_unsigners()
    with pytest.raises(TypeError):
        var_7 = list(var_6)

def test_case_18():
    var_0 = b'test_secret'
    var_1 = b'test_salt'
    var_2 = {}
    var_3 = [var_2]
    var_4 = module_0.Serializer(var_0, var_1, fallback_signers=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'test_secret']
    assert var_4.salt == b'test_salt'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == [{}]
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_5 = var_4.iter_unsigners(var_1)
    var_6 = list(var_5)
    var_7 = 1
    var_8 = var_6[var_7]