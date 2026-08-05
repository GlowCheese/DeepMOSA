# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1
import builtins as module_2

def test_case_0():
    var_0 = b'\nw\x86\t\xd79\x02\x81\xdf(h\xe2'
    var_1 = module_0.URLSafeTimedSerializer(var_0, signer=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'\nw\x86\t\xd79\x02\x81\xdf(h\xe2']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer == b'\nw\x86\t\xd79\x02\x81\xdf(h\xe2'
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b'\nw\x86\t\xd79\x02\x81\xdf(h\xe2'
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0, var_0)

def test_case_1():
    var_0 = b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2'
    var_1 = None
    var_2 = module_0.URLSafeTimedSerializer(var_0, signer=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_2.secret_keys == [b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2']
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

def test_case_2():
    var_0 = b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2'
    var_1 = None
    var_2 = module_0.URLSafeSerializer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2']
    assert var_2.salt is None
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

def test_case_3():
    var_0 = b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2'
    var_1 = None
    var_2 = module_0.URLSafeSerializer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'G\xb7\xe8\xe7\x83j9g\x0fdI\xab\xe0\x14=8\xe4\xa2']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = ''
    var_2 = module_0.URLSafeSerializer(var_1, serializer=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'']
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
    var_4 = b'4\x80\xc1e\x9f\x04\xcd]e.\xf5\x81E\xd5'
    var_5 = None
    var_6 = var_2.loads_unsafe(var_4)
    var_7 = module_2.int
    var_8 = module_0.URLSafeSerializerMixin(var_4, serializer_kwargs=var_4, signer=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_8.secret_keys == [b'4\x80\xc1e\x9f\x04\xcd]e.\xf5\x81E\xd5']
    assert var_8.salt == b'itsdangerous'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert var_8.fallback_signers == []
    assert var_8.serializer_kwargs == b'4\x80\xc1e\x9f\x04\xcd]e.\xf5\x81E\xd5'
    var_9 = [var_6, var_5, var_6]
    var_10 = var_2.dump_payload(var_9)
    assert var_10 == b'.eJyLjk5LzClO1ckrzcmJBZM6yCKxAMeKDCA'
    var_2.load(var_5)

def test_case_5():
    var_0 = b'4\x80\xc1e\x9f\x04\xcd]e.\xf5\x81E\x10'
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'4\x80\xc1e\x9f\x04\xcd]e.\xf5\x81E\x10']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'.XD\xf6\x81\xbaI'
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2)