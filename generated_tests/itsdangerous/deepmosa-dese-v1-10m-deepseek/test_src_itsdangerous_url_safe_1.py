# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = module_0.URLSafeSerializer(var_0, var_0, serializer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'invalid!!!']
    assert var_1.salt == b'invalid!!!'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == b'invalid!!!'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_1():
    var_0 = b'\x03h\x03'
    var_1 = module_0.URLSafeSerializer(var_0, var_0, serializer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'\x03h\x03']
    assert var_1.salt == b'\x03h\x03'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == b'\x03h\x03'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_2():
    var_0 = None
    var_1 = b'q:\xfa\t\xaf\x9c\x991\x94\xea\xa2\xfc\xcc\xe8'
    var_2 = module_0.URLSafeSerializerMixin(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b'q:\xfa\t\xaf\x9c\x991\x94\xea\xa2\xfc\xcc\xe8']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = module_0.URLSafeSerializerMixin(var_1, var_0, var_2, var_0, fallback_signers=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_3.secret_keys == [b'q:\xfa\t\xaf\x9c\x991\x94\xea\xa2\xfc\xcc\xe8']
    assert var_3.salt is None
    assert f'{type(var_3.serializer).__module__}.{type(var_3.serializer).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Y;5)EwYh'
    var_1 = None
    module_0.URLSafeSerializer(var_0, var_1, var_0, signer_kwargs=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = b'\xd3\xf5s\xa5\x81\x0e\x86\xa0\x80\x9d'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = module_0.URLSafeTimedSerializer(var_1, signer=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_3.secret_keys == [b'\xd3\xf5s\xa5\x81\x0e\x86\xa0\x80\x9d']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.dump_payload(var_2)
    assert var_4 == b'.eJyLzivNydHBQsQCAIfxChA'
    var_5 = [var_0, var_0, var_0]
    var_3.load_payload(var_4, *var_5)

def test_case_5():
    var_0 = ''
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'.'
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2)