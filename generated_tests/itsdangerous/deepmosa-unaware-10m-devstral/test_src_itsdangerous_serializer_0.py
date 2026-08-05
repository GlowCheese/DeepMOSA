# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous.serializer as module_0
import datetime as module_1
import src.itsdangerous.exc as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._PDataSerializer()

def test_case_1():
    var_0 = ',V87.Kx1'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b',V87.Kx1']
    assert var_1.salt == b',V87.Kx1'
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

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_3.reset()

def test_case_3():
    var_0 = ',V87.Kx1'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b',V87.Kx1']
    assert var_1.salt == b',V87.Kx1'
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
    var_2.loads(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '/bR^^'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'/bR^^']
    assert var_1.salt == b'/bR^^'
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
    var_1.loads_unsafe(var_2)

def test_case_8():
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
    var_3 = var_2.dumps(var_0)
    assert var_3 == 'null.cUsBfsj6xjHAXlz-ROzjEL2XnRI'

def test_case_9():
    var_0 = ',V87~Kx1'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b',V87~Kx1']
    assert var_1.salt == b',V87~Kx1'
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
    var_2 = var_1.dumps(var_0, var_0)
    assert var_2 == '",V87~Kx1".9BR5q1PCwpW7Vs4jUfLQEU2WJao'

@pytest.mark.xfail(strict=True)
def test_case_10():
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

def test_case_11():
    var_0 = ',V87Kx'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b',V87Kx']
    assert var_1.salt == b',V87Kx'
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
def test_case_12():
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
def test_case_13():
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
def test_case_14():
    var_0 = None
    var_1 = '/bR^^'
    module_0.Serializer(var_1, serializer=var_1, signer_kwargs=var_0)

def test_case_15():
    var_0 = '+0r\\jVL_[-$'
    var_1 = module_0.Serializer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_1.secret_keys == [b'+0r\\jVL_[-$']
    assert var_1.salt == b'+0r\\jVL_[-$'
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

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    var_4 = 'k\x0b:V%'
    var_5 = var_2.loads_unsafe(var_4)
    var_2.load_payload(var_0, var_3)

def test_case_17():
    var_0 = None
    var_1 = '+0r\\jVL_[-$'
    var_2 = module_0.Serializer(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'+0r\\jVL_[-$']
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
    var_3 = 'c^c_QO'
    var_4 = module_0.Serializer(var_3, serializer=var_2, signer=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.secret_keys == [b'c^c_QO']
    assert var_4.salt == b'itsdangerous'
    assert f'{type(var_4.serializer).__module__}.{type(var_4.serializer).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_4.is_text_serializer is True
    assert var_4.signer_kwargs == {}
    assert var_4.fallback_signers == []
    assert var_4.serializer_kwargs == {}
    var_5 = module_1.date
    var_6 = module_0.Serializer(var_1, serializer=var_0, signer=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_6.secret_keys == [b'+0r\\jVL_[-$']
    assert var_6.salt == b'itsdangerous'
    assert var_6.is_text_serializer is True
    assert var_6.signer_kwargs == {}
    assert var_6.fallback_signers == []
    assert var_6.serializer_kwargs == {}
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_7 = var_4.dumps(var_0, var_3)
    assert var_7 == 'null.CIHbEW3_YpEXbOwZRB0RtxNUo48.PgmpJCiBPGwz6fdQenNcF1ySXjo'
    var_8 = var_2.loads_unsafe(var_1, var_0)
    var_9 = var_2.loads_unsafe(var_7, var_7)
    with pytest.raises(module_2.BadPayload):
        var_4.load_payload(var_0)

def test_case_18():
    var_0 = b'\xe2\xd7\xcb\x1e\xedw\xe0\x842\xf73*;\x8c\xc6\xbf\xe8\xabj'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1, signer_kwargs=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_2.secret_keys == [b'\xe2\xd7\xcb\x1e\xedw\xe0\x842\xf73*;\x8c\xc6\xbf\xe8\xabj']
    assert var_2.salt == b'itsdangerous'
    assert var_2.is_text_serializer is True
    assert var_2.signer_kwargs == b'\xe2\xd7\xcb\x1e\xedw\xe0\x842\xf73*;\x8c\xc6\xbf\xe8\xabj'
    assert var_2.fallback_signers == []
    assert var_2.serializer_kwargs == {}
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.Serializer.default_fallback_signers == []
    assert f'{type(module_0.Serializer.secret_key).__module__}.{type(module_0.Serializer.secret_key).__qualname__}' == 'builtins.property'
    with pytest.raises(module_2.BadPayload):
        var_2.load_payload(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'digest_method'
    var_13 = 'sha256'
    var_14 = {var_12: var_13}
    var_15 = [var_14]
    var_16 = module_0.Serializer(var_0, fallback_signers=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'src.itsdangerous.serializer.Serializer'
    assert var_16.secret_keys == [b'secret-key']
    assert var_16.salt == b'itsdangerous'
    assert var_16.is_text_serializer is True
    assert var_16.signer_kwargs == {}
    assert var_16.fallback_signers == [{'digest_method': 'sha256'}]
    assert var_16.serializer_kwargs == {}
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_16.iter_unsigners()
    var_21 = list(var_20)
    var_22 = var_16.iter_unsigners()
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_10.iter_unsigners()