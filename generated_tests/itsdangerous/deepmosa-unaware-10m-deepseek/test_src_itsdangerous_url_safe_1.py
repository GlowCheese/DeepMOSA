# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = b';\xb2\xed\xa5\xa63\x8a\xc9\xe9\xd7\x15\xa2\xdb\xfc\x10\xae'
    var_1 = module_0.URLSafeSerializerMixin(var_0, var_0, signer=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b';\xb2\xed\xa5\xa63\x8a\xc9\xe9\xd7\x15\xa2\xdb\xfc\x10\xae']
    assert var_1.salt == b';\xb2\xed\xa5\xa63\x8a\xc9\xe9\xd7\x15\xa2\xdb\xfc\x10\xae'
    assert var_1.is_text_serializer is True
    assert var_1.signer == b';\xb2\xed\xa5\xa63\x8a\xc9\xe9\xd7\x15\xa2\xdb\xfc\x10\xae'
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_1():
    var_0 = None
    var_1 = b'.'
    var_2 = '7Af'
    var_3 = module_0.URLSafeTimedSerializer(var_1, serializer=var_0, signer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_3.secret_keys == [b'.']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = module_0.URLSafeSerializer(var_2, fallback_signers=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_4.secret_keys == [b'7Af']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == '7Af'
    assert var_4.serializer_kwargs == {}
    var_5 = var_4.dump_payload(var_0)
    assert var_5 == b'bnVsbA'
    var_6 = var_4.load_payload(var_5, serializer=var_4)

def test_case_2():
    var_0 = '\x0bU1tzKUbJUP7XHUbqg'
    var_1 = None
    var_2 = module_0.URLSafeTimedSerializer(var_0, var_0, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_2.secret_keys == [b'\x0bU1tzKUbJUP7XHUbqg']
    assert var_2.salt == b'\x0bU1tzKUbJUP7XHUbqg'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dump_payload(var_1)
    assert var_3 == b'bnVsbA'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Y;5)EwYh'
    var_1 = None
    module_0.URLSafeSerializer(var_0, var_1, var_0, signer_kwargs=var_1)

def test_case_4():
    var_0 = b'.'
    var_1 = module_0.URLSafeSerializer(var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'.']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b'.'
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_5():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'test-secret']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = 'message'
    var_3 = 'value'
    var_4 = 'hello'
    var_5 = 42
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.dump_payload(var_6)
    assert var_7 == b'eyJtZXNzYWdlIjoiaGVsbG8iLCJ2YWx1ZSI6NDJ9'
    var_8 = module_2.base64_decode(var_7)
    assert var_8 == b'{"message":"hello","value":42}'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    var_9 = 'a'
    var_10 = 100
    var_11 = var_9 * var_10
    var_12 = 'data'
    var_13 = {var_12: var_11}
    var_14 = var_1.dump_payload(var_13)
    assert var_14 == b'.eJyrVkpJLElUslJKpANQqgUA6jEpOQ'
    var_15 = 1
    var_16 = var_14[var_15:]
    var_17 = module_2.base64_decode(var_16)
    assert var_17 == b'x\x9c\xabVJI,IT\xb2RJ\xa4\x03P\xaa\x05\x00\xea1)9'
    with pytest.raises(module_1.BadData):
        module_2.base64_decode(var_14)
    assert var_18 == b'{"x":1}'