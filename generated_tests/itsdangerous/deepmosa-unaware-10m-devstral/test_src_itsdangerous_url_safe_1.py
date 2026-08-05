# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = None
    var_1 = '.FK_i01f'
    var_2 = module_0.URLSafeSerializer(var_1, serializer_kwargs=var_1, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'.FK_i01f']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == '.FK_i01f'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_1, var_0)

def test_case_1():
    var_0 = None
    var_1 = '._iS01.'
    var_2 = module_0.URLSafeSerializer(var_1, serializer_kwargs=var_1, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'._iS01.']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == '._iS01.'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_1, var_0)

def test_case_2():
    var_0 = ''
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
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
    var_2 = None
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'bnVsbA.amtUgw.Lx2CBH3D6vfgV6yKOM_xep2vYqI'

def test_case_3():
    var_0 = ''
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
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

def test_case_4():
    var_0 = None
    var_1 = '.F_iS01.'
    var_2 = module_0.URLSafeSerializer(var_1, serializer_kwargs=var_1, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'.F_iS01.']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == '.F_iS01.'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = '.F_i01f'
    var_2 = module_0.URLSafeSerializer(var_1, serializer_kwargs=var_1, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_2.secret_keys == [b'.F_i01f']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == '.F_i01f'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_3 = var_2.loads_unsafe(var_1, var_0)
    var_4 = module_0.URLSafeSerializer(var_1, signer=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_4.secret_keys == [b'.F_i01f']
    assert var_4.salt == b'itsdangerous'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = module_0.URLSafeSerializerMixin(var_1, fallback_signers=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_5.secret_keys == [b'.F_i01f']
    assert var_5.salt == b'itsdangerous'
    assert var_5.is_text_serializer is True
    assert var_5.signer_kwargs == {}
    assert var_5.fallback_signers == []
    assert var_5.serializer_kwargs == {}
    var_6 = ''
    var_7 = var_5.dump_payload(var_0)
    assert var_7 == b'bnVsbA'
    var_8 = 'S:A2'
    var_9 = {var_6: var_0, var_1: var_0, var_1: var_0, var_8: var_0}
    var_10 = var_4.dump_payload(var_9)
    assert var_10 == b'.eJyrVlKyyivNydFR0nOLzzQwTINxg60cjSDsWgDFEgr_'
    var_10.loads(var_0)