# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = b'^\xb3\xe4\\\x84m\xbf\x10 0\xcc'
    var_1 = module_0.URLSafeSerializerMixin(var_0, signer=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'^\xb3\xe4\\\x84m\xbf\x10 0\xcc']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer == b'^\xb3\xe4\\\x84m\xbf\x10 0\xcc'
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b'^\xb3\xe4\\\x84m\xbf\x10 0\xcc'
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_1():
    var_0 = None
    var_1 = b"\xa3\x0c\xb0L[\xe4'\x85\x9b\xcd\x8bm\x03`\x11\xba\x0cc"
    var_2 = ''
    var_3 = module_0.URLSafeSerializer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_3.secret_keys == [b'']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.dump_payload(var_0)
    assert var_4 == b'bnVsbA'
    with pytest.raises(module_1.BadPayload):
        var_3.load_payload(var_1)

def test_case_2():
    var_0 = None
    var_1 = b'.j\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd'
    var_2 = module_0.URLSafeSerializerMixin(var_1, serializer=var_0, signer=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b'.j\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.dump_payload(var_0)
    assert var_3 == b'bnVsbA'

def test_case_3():
    var_0 = None
    var_1 = b'.j\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd'
    var_2 = module_0.URLSafeSerializerMixin(var_1, serializer=var_0, signer=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b'.j\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_1, var_0)