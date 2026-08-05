# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = ')/\x0cfap\tA?ajU7E\r'
    var_1 = None
    var_2 = module_0.URLSafeSerializer(var_0, serializer=var_1, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b')/\x0cfap\tA?ajU7E\r']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = b'=Z\xb3a!\x96\xfbVw\xd8\xb7L\x8bP\xa9h'
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_3)

def test_case_1():
    var_0 = None
    var_1 = b'.h\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd'
    var_2 = module_0.URLSafeSerializerMixin(var_1, serializer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b'.h\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd']
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
    var_4 = var_2.load_payload(var_3)
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_1, var_0)

def test_case_2():
    var_0 = 't7S+6N\n+jr5Vg&iO-\t'
    var_1 = None
    var_2 = module_0.URLSafeSerializerMixin(var_0, serializer=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b't7S+6N\n+jr5Vg&iO-\t']
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
    assert var_3 == b'bnVsbA'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.URLSafeSerializerMixin(var_0, var_0, var_0, fallback_signers=var_0)

def test_case_4():
    var_0 = None
    var_1 = b'.h\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd'
    var_2 = module_0.URLSafeSerializerMixin(var_1, serializer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b'.h\xeb\x13?\xc4\x7f\xad\x18\xe3T\x06H\xae\xfb\xae\x1fd']
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

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'value'
    var_1 = None
    var_2 = module_0.URLSafeTimedSerializer(var_0, serializer=var_1, serializer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_2.secret_keys == [b'value']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = 'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = var_2.dump_payload(var_5)
    assert var_6 == b'.eJxTqhgFo2AUDHugBAAFGdUU'
    var_5.load_payload(var_1, *var_5, **var_5)