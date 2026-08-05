# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import enum as module_1
import src.itsdangerous.exc as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._PDataSerializer()

def test_case_1():
    var_0 = None
    var_1 = '6jhpHQ+WiE|n\x0c'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'6jhpHQ+WiE|n\x0c']
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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = '*P+2]s~'
    var_2 = module_1.Enum
    var_3 = [var_2]
    var_4 = module_0.Serializer(var_1, var_0, serializer_kwargs=var_0, signer_kwargs=var_0, fallback_signers=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'*P+2]s~']
    assert var_4.salt is None
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert f'{type(var_4.fallback_signers).__module__}.{type(var_4.fallback_signers).__qualname__}' == 'builtins.list'
    assert len(var_4.fallback_signers) == 1
    assert var_4.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_4.loads_unsafe(var_1)

def test_case_3():
    var_0 = '/=.\r]:u!@YN+58_:#'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'/=.\r]:u!@YN+58_:#']
    assert var_1.salt == b'/=.\r]:u!@YN+58_:#'
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
    module_0.is_text_serializer(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = '/bR^^'
    var_2 = module_0.Serializer(var_1, serializer_kwargs=var_1, fallback_signers=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'/bR^^']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == '/bR^^'
    assert var_2.serializer_kwargs == '/bR^^'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2.dump_payload(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'secret']
    assert var_2.salt == b'salt'
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
    var_2.loads_unsafe(var_3, var_1)

def test_case_7():
    var_0 = '/=.\r]:u!@YN+58_:#'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'/=.\r]:u!@YN+58_:#']
    assert var_1.salt == b'/=.\r]:u!@YN+58_:#'
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
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'null.mrod0gA9B2aB-ecB6sQUfIDMsC4'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '/bR^^'
    var_1 = module_0.Serializer(var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'/bR^^']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == '/bR^^'
    assert var_1.serializer_kwargs == '/bR^^'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = None
    var_1.load_unsafe(var_2)

def test_case_9():
    var_0 = '[@'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'[@']
    assert var_1.salt == b'[@'
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
def test_case_10():
    var_0 = None
    var_1 = '/bR^^'
    var_2 = module_0.Serializer(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'/bR^^']
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
    var_3 = var_2.dumps(var_0)
    assert var_3 == 'null.JwujuKUnvaOBNgBFwmPSZi8DToQ'
    var_4 = var_2.loads_unsafe(var_3, var_3)
    var_5 = var_2.loads_unsafe(var_1)
    var_2.dump(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '/bR^^'
    var_1 = module_0.Serializer(var_0, serializer_kwargs=var_0, fallback_signers=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'/bR^^']
    assert var_1.salt == b'itsdangerous'
    assert var_1.is_text_serializer is True
    assert var_1.signer_kwargs == {}
    assert var_1.fallback_signers == '/bR^^'
    assert var_1.serializer_kwargs == '/bR^^'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2 = False
    var_3 = (var_2, var_2)
    var_4 = {var_0: var_0, var_2: var_1, var_0: var_1, var_1: var_0}
    var_5 = (var_3, var_4)
    var_1.load(var_5, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = '/bR^^'
    module_0.Serializer(var_1, serializer=var_1, signer_kwargs=var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = '/=.\r]:u!@YN+58_:#'
    var_2 = module_0.Serializer(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'/=.\r]:u!@YN+58_:#']
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
    var_3 = var_2.dumps(var_0)
    assert var_3 == 'null.SQ1mMG0U-ZyoGBSYqaihOxICyVQ'
    var_4 = var_2.loads_unsafe(var_3, var_3)
    var_5 = var_2.loads_unsafe(var_1)
    var_2.load_payload(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = '/bR^^'
    var_2 = module_0.Serializer(var_1, var_0, signer=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'/bR^^']
    assert var_2.salt is None
    assert var_2.is_text_serializer is True
    assert var_2.signer == '/bR^^'
    assert var_2.signer_kwargs == {}
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_2.dumps(var_0)

def test_case_15():
    var_0 = None
    var_1 = '/=.\r]:u!@YN+58_:#'
    var_2 = module_0.Serializer(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'/=.\r]:u!@YN+58_:#']
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
    var_3 = '*P+s~'
    var_4 = var_2.dumps(var_0)
    assert var_4 == 'null.SQ1mMG0U-ZyoGBSYqaihOxICyVQ'
    var_5 = var_2.loads_unsafe(var_4, var_4)
    var_6 = module_0.Serializer(var_4, var_3, signer=var_4, signer_kwargs=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_6.secret_keys == [b'null.SQ1mMG0U-ZyoGBSYqaihOxICyVQ']
    assert var_6.salt == b'*P+s~'
    assert var_6.is_text_serializer is True
    assert var_6.signer == 'null.SQ1mMG0U-ZyoGBSYqaihOxICyVQ'
    assert var_6.signer_kwargs == 'null.SQ1mMG0U-ZyoGBSYqaihOxICyVQ'
    assert var_6.fallback_signers == []
    assert var_6.serializer_kwargs == {}
    var_7 = var_2.loads_unsafe(var_1)
    with pytest.raises(module_2.BadPayload):
        var_2.load_payload(var_0)

def test_case_16():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'extra_arg'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_6.secret_keys == [b'secret']
    assert var_6.salt == b'salt'
    assert var_6.is_text_serializer is True
    assert var_6.signer_kwargs == {}
    assert var_6.fallback_signers == [{'extra_arg': 'value'}]
    assert var_6.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    var_7 = var_6.iter_unsigners()
    with pytest.raises(TypeError):
        var_8 = list(var_7)