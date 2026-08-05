# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = b'{"tKst": "d5ta"}'
    var_1 = module_0.URLSafeSerializer(var_0, var_0, serializer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'{"tKst": "d5ta"}']
    assert var_1.salt == b'{"tKst": "d5ta"}'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == b'{"tKst": "d5ta"}'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_1():
    var_0 = b'\xc0\x84\x8e\xc5\xf0v\x9d'
    var_1 = module_0.URLSafeSerializer(var_0, var_0, serializer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'\xc0\x84\x8e\xc5\xf0v\x9d']
    assert var_1.salt == b'\xc0\x84\x8e\xc5\xf0v\x9d'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == b'\xc0\x84\x8e\xc5\xf0v\x9d'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0, serializer=var_0)

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
    var_4 = module_0.URLSafeSerializer(var_3, signer_kwargs=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_4.secret_keys == [b'bnVsbA']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Y;5)EwYh'
    var_1 = None
    module_0.URLSafeSerializer(var_0, var_1, var_0, signer_kwargs=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'gano]:EFcAhMr'
    var_1 = [var_0, var_0]
    var_2 = module_0.URLSafeSerializer(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'gano]:EFcAhMr']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dump_payload(var_1)
    assert var_3 == b'.eJyLVkpPzMuPtXJ1S3bM8C1S0kHjxwIAtOkKkQ'
    var_4 = var_2.loads_unsafe(var_3, var_3)
    var_0.loads(var_3)

def test_case_5():
    var_0 = b'.'
    var_1 = module_0.URLSafeSerializer(var_0, signer=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'.']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer == b'.'
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)