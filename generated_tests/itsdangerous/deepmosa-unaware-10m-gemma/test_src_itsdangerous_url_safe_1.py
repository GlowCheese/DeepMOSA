# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.url_safe as module_0
import src.itsdangerous.exc as module_1
import typing as module_2
import src.itsdangerous.serializer as module_3

def test_case_0():
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

def test_case_1():
    var_0 = b'\x1d\xfb\x05J'
    var_1 = ''
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_1}
    var_3 = module_0.URLSafeSerializer(var_1, signer=var_1, signer_kwargs=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_3.secret_keys == [b'']
    assert var_3.salt == b'itsdangerous'
    assert var_3.is_text_serializer is True
    assert var_3.signer == ''
    assert var_3.signer_kwargs == {}
    assert var_3.fallback_signers == []
    assert var_3.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    with pytest.raises(module_1.BadPayload):
        var_3.load_payload(var_0, **var_2)

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

def test_case_4():
    var_0 = b'.\xe9'
    var_1 = None
    var_2 = module_0.URLSafeTimedSerializer(var_0, signer=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeTimedSerializer'
    assert var_2.secret_keys == [b'.\xe9']
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
        var_2.load_payload(var_0, serializer=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = None
    var_2 = 'ywmw\x0bo45\x0c<\x0b\x0b-xs,|ST'
    var_3 = module_2.Protocol
    var_4 = 'DBRZ~\n~'
    var_5 = {var_4: var_2, var_2: var_2}
    var_6 = (var_3, var_5)
    var_7 = [var_6]
    var_8 = module_0.URLSafeSerializer(var_2, serializer=var_1, serializer_kwargs=var_0, signer_kwargs=var_1, fallback_signers=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializer'
    assert var_8.secret_keys == [b'ywmw\x0bo45\x0c<\x0b\x0b-xs,|ST']
    assert var_8.salt == b'itsdangerous'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert f'{type(var_8.fallback_signers).__module__}.{type(var_8.fallback_signers).__qualname__}' == 'builtins.list'
    assert len(var_8.fallback_signers) == 1
    assert var_8.serializer_kwargs == {}
    assert module_2.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_2.T).__module__}.{type(module_2.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT).__module__}.{type(module_2.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.V_co).__module__}.{type(module_2.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_contra).__module__}.{type(module_2.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.CT_co).__module__}.{type(module_2.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.AnyStr).__module__}.{type(module_2.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_9 = var_8.dump_payload(var_1)
    assert var_9 == b'bnVsbA'
    var_10 = var_8.dump_payload(var_5)
    assert var_10 == b'.eJyrVnJxCoqqi8mrU7JSqizPLY8pNTAwSMo3MY1Js4GwIaRuRbFOTXCIkg5RqogzqxYA68Inrw'
    var_11 = module_3.Serializer
    var_12 = module_0.URLSafeSerializerMixin(var_9, signer=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'src.itsdangerous.url_safe.URLSafeSerializerMixin'
    assert var_12.secret_keys == [b'bnVsbA']
    assert var_12.salt == b'itsdangerous'
    assert var_12.is_text_serializer is True
    assert var_12.signer_kwargs == {}
    assert var_12.fallback_signers == []
    assert var_12.serializer_kwargs == {}
    assert f'{type(module_3.annotations).__module__}.{type(module_3.annotations).__qualname__}' == '__future__._Feature'
    assert module_3.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_3.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_3.annotations.compiler_flag == 16777216
    module_0.URLSafeTimedSerializer(var_0)