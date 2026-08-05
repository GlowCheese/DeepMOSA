# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = b'\x03H>\xda\x00\xe1\x02bTI\x14+\x8cm\x14\xb8'
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'\x03H>\xda\x00\xe1\x02bTI\x14+\x8cm\x14\xb8']
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
    var_0 = b''
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
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

def test_case_2():
    var_0 = b'\x03H>\xda\x00\xe1\x02bTI\x14+\x8cm\x14\xb8'
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'\x03H>\xda\x00\xe1\x02bTI\x14+\x8cm\x14\xb8']
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
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.URLSafeSerializer(var_0, var_0, signer_kwargs=var_0)

def test_case_4():
    var_0 = '9wfM)O08ow"W)-z$+Rp'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = None
    var_3 = module_0.URLSafeTimedSerializer(var_0, serializer_kwargs=var_2, signer=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_3.secret_keys == [b'9wfM)O08ow"W)-z$+Rp']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_4 = var_3.dumps(var_1)
    assert var_4 == '.eJyrVrIsT_PV9DewyC-PUQrX1K1S0Q4qULLCLlwLAEWJDXk.ampVYw.V3-hVhHUdBnuzpQ6qve8gn1n9_A'
    var_5 = module_2.base64_decode(var_4)
    assert var_5 == b'x\x9c\xabV\xb2,O\xf3\xd5\xf47\xb0\xc8/\x8fQ\n\xd7\xd4\xadR\xd1\x0e*P\xb2\xc2.\\\x0b\x00E\x89\ry\x1a\x9a\x95X\xc1]\xfe\x85XGQ\xd0g\xbb:P\xea\xab\xde\xf2\t\xf5\x9f\xdf\xc0'
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    var_6 = {var_0: var_0}
    var_7 = module_0.URLSafeSerializer(var_0, signer_kwargs=var_6, fallback_signers=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_7.secret_keys == [b'9wfM)O08ow"W)-z$+Rp']
    assert var_7.salt == b'itsdangerous'
    assert var_7.is_text_serializer is True
    assert var_7.signer_kwargs == {'9wfM)O08ow"W)-z$+Rp': '9wfM)O08ow"W)-z$+Rp'}
    assert var_7.fallback_signers == '9wfM)O08ow"W)-z$+Rp'
    assert var_7.serializer_kwargs == {}
    var_8 = b'"\x97\x8e\x80|\xfe$w\\L\xbdF$*\x01n'
    with pytest.raises(module_1.BadPayload):
        var_7.load_payload(var_8)

def test_case_5():
    var_0 = b'.\x19\xfc\xd5\xc3\x98\x0c,\xf1\x08\x8a\x13\x99'
    var_1 = module_0.URLSafeTimedSerializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_1.secret_keys == [b'.\x19\xfc\xd5\xc3\x98\x0c,\xf1\x08\x8a\x13\x99']
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