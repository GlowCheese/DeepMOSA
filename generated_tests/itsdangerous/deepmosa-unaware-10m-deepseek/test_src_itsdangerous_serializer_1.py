# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import src.itsdangerous.exc as module_1

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

def test_case_4():
    var_0 = 'Test that text serializer decodes UTF-8 properly.'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'Test that text serializer decodes UTF-8 properly.']
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
    with pytest.raises(module_1.BadPayload):
        var_1.load_payload(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'Ii|B [*O.G'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'Ii|B [*O.G']
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
    var_1.loads_unsafe(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
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

def test_case_7():
    var_0 = 'Ii|B [*O.G'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'Ii|B [*O.G']
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
def test_case_8():
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

def test_case_9():
    var_0 = 'Ii|B [*O.G'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'Ii|B [*O.G']
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
    var_2 = var_1.loads_unsafe(var_0, var_0)

def test_case_10():
    var_0 = 'I99[O.G'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_0, serializer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'I99[O.G']
    assert var_2.salt == b'I99[O.G'
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
    var_3 = var_2.dumps(var_1)
    assert var_3 == 'null.ukNmsF4xlqXIwgO5oSC1dDgspnY'
    var_4 = module_0.Serializer(var_0, serializer=var_2, fallback_signers=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'I99[O.G']
    assert var_4.salt == b'itsdangerous'
    assert f'{type(var_4.serializer).__module__}.{type(var_4.serializer).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = var_2.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = b'.\x15\xd4A\xfb\xf5'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'.\x15\xd4A\xfb\xf5']
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
    var_4 = var_2.loads_unsafe(var_1, var_1)
    var_2.dump(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1, signer_kwargs=var_1, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe8\xdc\xaa\xb4C6!/\x11\xa2\xf5']
    assert var_2.salt is None
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
    var_2.loads_unsafe(var_0, var_0)

def test_case_13():
    var_0 = 'Ig99[O.G'
    var_1 = module_0.Serializer(var_0, var_0, serializer_kwargs=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'Ig99[O.G']
    assert var_1.salt == b'Ig99[O.G'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == []
    assert var_1.serializer_kwargs == 'Ig99[O.G'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = var_1.loads_unsafe(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'Ig99[O.G'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_0, serializer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'Ig99[O.G']
    assert var_2.salt == b'Ig99[O.G'
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
    var_3 = var_2.dumps(var_1, var_1)
    assert var_3 == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    var_4 = module_0.Serializer(var_0, var_3, serializer_kwargs=var_3, signer=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'Ig99[O.G']
    assert var_4.salt == b'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    assert var_4.is_text_serializer is True
    assert var_4.signer == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    var_2.loads_unsafe(var_1, var_3)

def test_case_15():
    var_0 = 'Ig99[O.G'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_0, serializer_kwargs=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'Ig99[O.G']
    assert var_2.salt == b'Ig99[O.G'
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
    var_3 = var_2.dumps(var_1, var_1)
    assert var_3 == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    var_4 = module_0.Serializer(var_3, var_1, serializer_kwargs=var_3, signer=var_1, signer_kwargs=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'null.1fyibUnsL2DCmcNzWc3qQMJXPVs']
    assert var_4.salt is None
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == 'null.1fyibUnsL2DCmcNzWc3qQMJXPVs'
    var_5 = b'\x10'
    var_6 = var_2.loads_unsafe(var_0, var_5)

def test_case_16():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'test-secret']
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
    assert var_8.secret_keys == [b'test-secret']
    assert var_8.salt == b'custom-salt'
    assert var_8.is_text_serializer is True
    assert var_8.signer_kwargs == {}
    assert var_8.fallback_signers == []
    assert var_8.serializer_kwargs == {}
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'digest_method'
    var_13 = 'sha256'
    var_14 = {var_12: var_13}
    var_15 = [var_14]
    var_16 = module_0.Serializer(var_0, fallback_signers=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_16.secret_keys == [b'test-secret']
    assert var_16.salt == b'itsdangerous'
    assert var_16.is_text_serializer is True
    assert var_16.signer_kwargs == {}
    assert var_16.fallback_signers == [{'digest_method': 'sha256'}]
    assert var_16.serializer_kwargs == {}
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_18[var_5]
    var_21 = 1
    var_22 = var_18[var_21]
    var_23 = var_16.iter_unsigners()
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_24[var_21]
    var_27 = var_16.iter_unsigners()
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_28[var_21]
    var_31 = 'old-key'
    var_32 = 'new-key'
    var_33 = [var_31, var_32]
    var_34 = module_0.Serializer(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_34.secret_keys == [b'old-key', b'new-key']
    assert var_34.salt == b'itsdangerous'
    assert var_34.is_text_serializer is True
    assert var_34.signer_kwargs == {}
    assert var_34.fallback_signers == []
    assert var_34.serializer_kwargs == {}
    var_35 = var_34.iter_unsigners()
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = [var_31, var_32]
    var_39 = {var_12: var_13}
    var_40 = [var_39]
    var_41 = module_0.Serializer(var_38, fallback_signers=var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_41.secret_keys == [b'old-key', b'new-key']
    assert var_41.salt == b'itsdangerous'
    assert var_41.is_text_serializer is True
    assert var_41.signer_kwargs == {}
    assert var_41.fallback_signers == [{'digest_method': 'sha256'}]
    assert var_41.serializer_kwargs == {}
    var_42 = var_41.iter_unsigners()
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 3
    var_45 = None
    var_46 = module_0.Serializer(var_0, var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_46.secret_keys == [b'test-secret']
    assert var_46.salt is None
    assert var_46.is_text_serializer is True
    assert var_46.signer_kwargs == {}
    assert var_46.fallback_signers == []
    assert var_46.serializer_kwargs == {}
    var_47 = var_46.iter_unsigners(var_45)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = module_0.Serializer(var_0)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_50.secret_keys == [b'test-secret']
    assert var_50.salt == b'itsdangerous'
    assert var_50.is_text_serializer is True
    assert var_50.signer_kwargs == {}
    assert var_50.fallback_signers == []
    assert var_50.serializer_kwargs == {}
    var_51 = 'override-salt'
    var_52 = var_50.iter_unsigners(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1