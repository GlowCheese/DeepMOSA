# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import src.itsdangerous.exc as module_1
import json.scanner as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._PDataSerializer()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.is_text_serializer(var_0)

def test_case_2():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'

def test_case_3():
    var_0 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2.loads(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_1 = module_0.Serializer(var_0, var_0, signer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_1.salt == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_1.fallback_signers == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_1.loads_unsafe(var_0)

def test_case_6():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.dumps(var_0, var_0)
    assert var_3 == 'null.SYwehGOtfeIR22Tah00tBlWf0m8'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = None
    var_2.load(var_3)

def test_case_8():
    var_0 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = None
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5Q_'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5Q_']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_1, var_0)
    var_4 = None
    var_2.load_unsafe(var_4)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = b'\xc2\x12\rD)F\x04\xa6\xb1\xfe\x13=\xbb'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xc2\x12\rD)F\x04\xa6\xb1\xfe\x13=\xbb']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.dumps(var_0)
    assert var_3 == 'null.NU5VD8l1-bZEwbvCEf9_NgsEtEU'
    var_4 = var_2.loads_unsafe(var_3, var_1)
    var_2.seek(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = b'k\xb8\x15\x1a/0O\xc6U\xc6\x9b\x1b\x91\xd6\x10in'
    var_2 = b'\x11\x17\xd6T\x98oS\xdf\xe9\xd0\xcaG'
    var_3 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    module_0.Serializer(var_3, var_2, var_1, signer=var_0, fallback_signers=var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_1, var_0)
    var_4 = None
    var_2.dump(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0'
    var_1 = module_0.Serializer(var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0'
    assert var_1.serializer_kwargs == b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_1.loads_unsafe(var_0, var_0)

def test_case_14():
    var_0 = None
    var_1 = b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0\xd1'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b's\xc1\t\xbew.\x9c}\xd2^Q\xd6\x9f\xfbJ}\\\xe0\xd1']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_3 = var_2.loads_unsafe(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_0, signer_kwargs=var_1, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '?\x0c"E\tmjvCoIfeU>j;Q'
    var_1 = module_0.Serializer(var_0, var_0, signer=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'?\x0c"E\tmjvCoIfeU>j;Q']
    assert var_1.salt == b'?\x0c"E\tmjvCoIfeU>j;Q'
    assert var_1.is_text_serializer is True
    assert var_1.signer == '?\x0c"E\tmjvCoIfeU>j;Q'
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = None
    module_2.py_make_scanner(var_2)

def test_case_17():
    var_0 = None
    var_1 = b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\x81^\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    with pytest.raises(module_1.BadPayload):
        var_2.load_payload(var_1, var_2)

def test_case_18():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'secret-key']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 'custom-salt'
    var_8 = module_0.Serializer(var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_8.secret_keys == [b'secret-key']
    assert var_8.salt == b'custom-salt'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert var_8.fallback_signers == []
    assert var_8.serializer_kwargs == {}
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = var_8.iter_unsigners()
    var_12 = list(var_11)
    var_13 = var_12[var_5]
    var_14 = 'digest_method'
    var_15 = 'sha256'
    var_16 = {var_14: var_15}
    var_17 = module_0.Serializer(var_0, signer_kwargs=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_17.secret_keys == [b'secret-key']
    assert var_17.salt == b'itsdangerous'
    assert var_17.is_text_serializer is True
    assert var_17.signer_kwargs == {'digest_method': 'sha256'}
    assert var_17.fallback_signers == []
    assert var_17.serializer_kwargs == {}
    var_18 = var_17.iter_unsigners()
    var_19 = list(var_18)
    var_20 = {var_14: var_15}
    var_21 = [var_20]
    var_22 = module_0.Serializer(var_0, fallback_signers=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_22.secret_keys == [b'secret-key']
    assert var_22.salt == b'itsdangerous'
    assert var_22.is_text_serializer is True
    assert var_22.signer_kwargs == {}
    assert var_22.fallback_signers == [{'digest_method': 'sha256'}]
    assert var_22.serializer_kwargs == {}
    var_23 = var_22.iter_unsigners()
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_22.iter_unsigners()
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = var_22.iter_unsigners()
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 1
    var_33 = var_30[var_32]
    var_34 = var_22.iter_unsigners()
    var_35 = list(var_34)
    var_36 = len(var_35)
    var_37 = 2
    with pytest.raises(IndexError):
        var_38 = var_35[var_37]