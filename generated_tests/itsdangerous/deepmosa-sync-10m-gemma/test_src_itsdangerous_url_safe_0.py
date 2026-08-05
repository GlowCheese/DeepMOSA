# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = b'\xb9u\x1f\x92\xee\xbc\x14\xc2\xa4\xee'
    var_1 = module_0.URLSafeSerializer(var_0, signer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_1.secret_keys == [b'\xb9u\x1f\x92\xee\xbc\x14\xc2\xa4\xee']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == b'\xb9u\x1f\x92\xee\xbc\x14\xc2\xa4\xee'
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
    var_1 = b'\xb9u\x1f\x92\xee\xbc\x14\xc2\xa4\xee'
    var_2 = module_0.URLSafeSerializer(var_1, signer_kwargs=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'\xb9u\x1f\x92\xee\xbc\x14\xc2\xa4\xee']
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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.URLSafeSerializer(var_0, serializer_kwargs=var_0, signer=var_0, signer_kwargs=var_0, fallback_signers=var_0)

def test_case_3():
    var_0 = b'\x0c\xc5/\xf8>\xfe\xbe\xb1z\xd8y\xcd\xbb\x16'
    var_1 = None
    var_2 = module_0.URLSafeSerializer(var_0, signer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'\x0c\xc5/\xf8>\xfe\xbe\xb1z\xd8y\xcd\xbb\x16']
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
        var_2.load_payload(var_0)

def test_case_4():
    var_0 = b'.'
    var_1 = module_0.URLSafeTimedSerializer(var_0, var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'.']
    assert var_1.salt == b'.'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b'.'
    assert var_1.serializer_kwargs == b'.'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0, var_0)

def test_case_5():
    var_0 = b'.'
    var_1 = module_0.URLSafeSerializerMixin(var_0, signer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'.']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == b'.'
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = {}
    var_3 = '\x0cE\r6#=}LTD\x0c9qW'
    var_4 = {var_3: var_3}
    var_5 = var_1.dump_payload(var_4)
    assert var_5 == b'.eJyrVopJc40pMlO2rfUJcYlJsywMV7LCIlYLAA0kDNk'
    var_6 = [var_4, var_4]
    var_7 = module_0.URLSafeTimedSerializer(var_0, var_0, serializer_kwargs=var_4, fallback_signers=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_7.secret_keys == [b'.']
    assert var_7.salt == b'.'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {}
    assert var_7.fallback_signers == [{'\x0cE\r6#=}LTD\x0c9qW': '\x0cE\r6#=}LTD\x0c9qW'}, {'\x0cE\r6#=}LTD\x0c9qW': '\x0cE\r6#=}LTD\x0c9qW'}]
    assert var_7.serializer_kwargs == {'\x0cE\r6#=}LTD\x0c9qW': '\x0cE\r6#=}LTD\x0c9qW'}
    with pytest.raises(module_1.BadPayload):
        var_7.load_payload(var_0, **var_2)