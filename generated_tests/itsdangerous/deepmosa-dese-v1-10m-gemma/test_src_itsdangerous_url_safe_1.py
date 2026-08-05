# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = b'\xaf\x90\xda\xde\xb9\xb9\xe5l;D\xe3\xa0'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'\xaf\x90\xda\xde\xb9\xb9\xe5l;D\xe3\xa0']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
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
    var_0 = b'\xef\xaa\tG\xb6 _n\xed\x7f\xe6^_\xff\x83\xee'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'\xef\xaa\tG\xb6 _n\xed\x7f\xe6^_\xff\x83\xee']
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
    var_3 = var_1.dump_payload(var_2)
    assert var_3 == b'bnVsbA'
    var_4 = b'.\xcf\xa3'
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.URLSafeSerializer(var_0, var_0, var_0, signer_kwargs=var_0)

def test_case_3():
    var_0 = b')\x19O`\xa0\x89h\x98\t\x9bB\x92\x02\xc5\xb6'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b')\x19O`\xa0\x89h\x98\t\x9bB\x92\x02\xc5\xb6']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'\x04\x0f\xcd\xfesHb\x7fa\xdf\xb1\xba\xd0:'
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2)

def test_case_4():
    var_0 = None
    var_1 = b')\x19O`\xa0\x89h&\x98\t\x9bB\x92\x02\xc5\xb6'
    var_2 = module_0.URLSafeSerializerMixin(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_2.secret_keys == [b')\x19O`\xa0\x89h&\x98\t\x9bB\x92\x02\xc5\xb6']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = '\\GvT'
    var_4 = 'fCU11\x0b$]#Y1'
    var_5 = ':&9\tuJqU:k^aq-8xc3\r'
    var_6 = {var_3: var_0, var_3: var_0, var_4: var_0, var_5: var_0}
    var_7 = var_2.dump_payload(var_6)
    assert var_7 == b'.eJyrVoqJcS8LUbLKK83J0VFKcw41NIwpNTAwSFKJVY40hIlbqVnGlJR6FYZaZcclFupaVCQbxxRBJGsBtXoVGg'
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_1, serializer=var_0)

def test_case_5():
    var_0 = b'\xef\xaa\tG\xb6 _n\xed\x7f\xe6^_\xff\x83\xee'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_1.secret_keys == [b'\xef\xaa\tG\xb6 _n\xed\x7f\xe6^_\xff\x83\xee']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_2 = b'.\xcf\xa3'
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2)