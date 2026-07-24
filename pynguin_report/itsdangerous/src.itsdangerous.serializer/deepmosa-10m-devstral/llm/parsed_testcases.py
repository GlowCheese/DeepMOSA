####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = ';'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': ';'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = ';'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = (var_8, var_11)
    var_13 = [var_2, var_12]
    var_14 = 'secret'
    var_15 = module_1.Serializer(var_14, fallback_signers=var_13)
    var_16 = var_15.secret_keys
    var_17 = bool(var_15.secret_keys == [b'secret'])
    assert var_17 is True
    var_18 = var_15.salt
    assert var_18 == b'itsdangerous'
    var_19 = var_15.serializer
    var_20 = var_15.is_text_serializer
    assert var_20 is True
    var_21 = var_15.signer
    var_22 = var_15.signer_kwargs
    var_23 = bool(var_15.signer_kwargs == {})
    assert var_23 is True
    var_24 = var_15.fallback_signers
    var_25 = bool(var_15.fallback_signers == var_13)
    assert var_25 is True
    var_26 = var_15.serializer_kwargs
    var_27 = bool(var_15.serializer_kwargs == {})
    assert var_27 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 2/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 2/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 1/8 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 2/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_8 is True

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_fallback_signers. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = b'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_error. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Could not load the payload'



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers is not None)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x.encode()
    var_3 = lambda x: x.decode()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = ':'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': ':'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = ':'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 11/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = 'loads'
    var_4 = 1
    var_5 = 0
    var_6 = var_4 / var_5
    var_7 = lambda x: var_6
    var_8 = {var_3: var_7}
    var_9 = var_1.load_payload(var_2, var_8)
    var_10 = str(var_3)
    var_11 = 'Could not load the payload because an exception occurred on unserializing the data.'
    var_12 = bool('Could not load the payload because an exception occurred on unserializing the data.' in var_10)
    assert var_12 is True



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_custom_serializer_text. Retrieved 1/8 statements.
# Partially parsed test_serializer_custom_serializer_bytes. Retrieved 1/8 statements.
# Partially parsed test_serializer_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key': 'value'}])
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 5/19 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1
    var_4 = 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_8 is True

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_invalid_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Could not load the payload'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #18
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_invalid_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Could not load the payload'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = lambda x: str(x)
    var_4 = lambda x: x
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, serializer=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = lambda x: str(x)
    var_11 = lambda x: x
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = var_6.serializer
    var_14 = bool(var_6.serializer == var_12)
    assert var_14 is True
    var_15 = var_6.is_text_serializer
    assert var_15 is True
    var_16 = var_6.signer
    var_17 = var_6.signer_kwargs
    var_18 = bool(var_6.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_6.fallback_signers
    var_20 = bool(var_6.fallback_signers == [])
    assert var_20 is True
    var_21 = var_6.serializer_kwargs
    var_22 = bool(var_6.serializer_kwargs == {})
    assert var_22 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = lambda x: x.encode()
    var_4 = lambda x: x.decode()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, serializer=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = lambda x: x.encode()
    var_11 = lambda x: x.decode()
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = var_6.serializer
    var_14 = bool(var_6.serializer == var_12)
    assert var_14 is True
    var_15 = var_6.is_text_serializer
    assert var_15 is False
    var_16 = var_6.signer
    var_17 = var_6.signer_kwargs
    var_18 = bool(var_6.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_6.fallback_signers
    var_20 = bool(var_6.fallback_signers == [])
    assert var_20 is True
    var_21 = var_6.serializer_kwargs
    var_22 = bool(var_6.serializer_kwargs == {})
    assert var_22 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/11 statements.
# Partially parsed test_dumps_with_serializer_kwargs. Retrieved 9/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = {}
    var_7 = module_0.loads(var_5, **var_6)
    var_8 = bool(var_7 == var_3)
    assert var_8 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = {}
    var_10 = module_0.loads(var_8, **var_9)
    var_11 = bool(var_10 == var_6)
    assert var_11 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = {}
    var_8 = var_2.loads(var_6, var_1, **var_7)
    var_9 = bool(var_8 == var_5)
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'salt2'
    var_8 = {}
    var_9 = var_2.loads(var_6, var_7, **var_8)
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = {}
    var_8 = var_2.loads(var_6, var_1, **var_7)
    var_9 = bool(var_8 == var_5)
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)
    var_8 = {}
    var_9 = var_3.loads(var_7, **var_8)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = {}
    var_10 = module_0.loads(var_8, **var_9)
    var_11 = bool(var_10 == var_6)
    assert var_11 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'binary data'
    var_3 = var_1.dumps(var_2)
    var_4 = {}
    var_5 = var_1.loads(var_3, **var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'unicode data: ☃'
    var_3 = var_1.dumps(var_2)
    var_4 = {}
    var_5 = var_1.loads(var_3, **var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 1/3 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.secret_key
    assert var_4 == b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'my_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'my_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_raises_badpayload_on_exception. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '?'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = ':'
    var_2 = {var_0: var_1}
    var_3 = ';'
    var_4 = {var_0: var_3}
    var_5 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 5/18 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_and_key_rotation. Retrieved 6/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[0].secret_keys
    var_9 = bool(var_4[0].secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_4[0].salt
    assert var_10 == b'custom-salt'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1
    var_4 = 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_5[0].secret_keys
    var_10 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_10 is True
    var_11 = var_5[0].salt
    assert var_11 == b'itsdangerous'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 4/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer
    var_3 = var_1.signer_kwargs
    var_4 = bool(var_1.signer_kwargs == {})
    assert var_4 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = 'Could not load the payload because an exception occurred on unserializing the data.'



# Parsed testcases at query #31
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dumps_with_bytes_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'old'
    var_5 = 'new'
    var_6 = [var_4, var_5]
    var_7 = 'custom'
    var_8 = 'protocol'
    var_9 = 2
    var_10 = {var_8: var_9}
    var_11 = {var_0: var_1}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_iter_unsigners_with_dict_fallback. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 1
    var_12 = var_7[var_11]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #36
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer is var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: x
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer is var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer is var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = [var_2, var_8]
    var_10 = 'secret'
    var_11 = module_1.Serializer(var_10, fallback_signers=var_9)
    var_12 = var_11.secret_keys
    var_13 = bool(var_11.secret_keys == [b'secret'])
    assert var_13 is True
    var_14 = var_11.salt
    assert var_14 == b'itsdangerous'
    var_15 = var_11.serializer
    var_16 = var_11.is_text_serializer
    assert var_16 is True
    var_17 = var_11.signer
    var_18 = var_11.signer_kwargs
    var_19 = bool(var_11.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_11.fallback_signers
    var_21 = bool(var_11.fallback_signers == var_9)
    assert var_21 is True
    var_22 = var_11.serializer_kwargs
    var_23 = bool(var_11.serializer_kwargs == {})
    assert var_23 is True



# Parsed testcases at query #37
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #39
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = 'custom'
    var_4 = lambda x: var_3
    var_5 = lambda x: var_3
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = module_0.Serializer(var_0, serializer=var_6)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == {'dumps': lambda x: 'custom', 'loads': lambda x: 'custom'})
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'SHA256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = b'custom'
    var_4 = lambda x: var_3
    var_5 = 'custom'
    var_6 = lambda x: var_5
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = module_0.Serializer(var_0, serializer=var_7)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == {'dumps': lambda x: b'custom', 'loads': lambda x: 'custom'})
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = b'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.fallback_signers
    var_9 = bool(var_7.fallback_signers == var_5)
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key1': 'value1'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key1': 'value1'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 2/12 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-secret', b'new-secret'])
    assert var_8 is True

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'42'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_isinstance_fallback_dict. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 1
    var_12 = var_7[var_11]



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 8/9 statements.
# Partially parsed test_dumps_with_serializer_kwargs. Retrieved 9/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = b'key'
    var_8 = bool(b'key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'
    var_6 = var_1.dumps(var_4, var_5)
    var_7 = 'key'
    var_8 = bool('key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'different-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)
    var_8 = 'key'
    var_9 = bool('key' in var_7)
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)
    var_9 = 'key'
    var_10 = bool('key' in var_8)
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_text_serializer_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dumps_with_text_serializer. Retrieved 6/9 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 7/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_different_object. Retrieved 8/10 statements.
# Partially parsed test_dumps_empty_object. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = '{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'
    var_6 = var_1.dumps(var_4, var_5)
    var_7 = '{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.dumps(var_5)
    var_7 = '["a", "b", "c"]'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    var_4 = '{}'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = {}
    var_2 = [var_1]
    var_3 = module_0.Serializer(var_0, fallback_signers=var_2)
    var_4 = var_3.fallback_signers
    var_5 = bool(var_3.fallback_signers is not None)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = {var_0: var_1}
    var_10 = (var_8, var_9)
    var_11 = [var_2, var_10]
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, fallback_signers=var_11)
    var_14 = var_13.secret_keys
    var_15 = bool(var_13.secret_keys == [b'secret'])
    assert var_15 is True
    var_16 = var_13.salt
    assert var_16 == b'itsdangerous'
    var_17 = var_13.serializer
    var_18 = var_13.is_text_serializer
    assert var_18 is True
    var_19 = var_13.signer
    var_20 = var_13.signer_kwargs
    var_21 = bool(var_13.signer_kwargs == {})
    assert var_21 is True
    var_22 = var_13.fallback_signers
    var_23 = bool(var_13.fallback_signers == var_11)
    assert var_23 is True
    var_24 = var_13.serializer_kwargs
    var_25 = bool(var_13.serializer_kwargs == {})
    assert var_25 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: x.decode(var_2)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_invalid_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_key
    assert var_7 == b'secret-key'
    var_8 = var_3[0].salt
    assert var_8 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_key
    assert var_7 == b'new-key'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_with_fallback_signers. Retrieved 2/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_6 is True

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 2/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 2/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signer_class. Retrieved 2/9 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 3/13 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 3

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_8 is True

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_invalid_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Could not load the payload'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers is not None)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 9/10 statements.
# Partially parsed test_dumps_with_none_salt. Retrieved 6/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'key'
    var_8 = bool('key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'different-secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'data'
    var_8 = bool('data' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = 'list'
    var_6 = 'keys'
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)
    var_9 = 'list'
    var_10 = bool('list' in var_8)
    assert var_10 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'salt'
    var_4 = {var_3: var_1}
    var_5 = var_2.dumps(var_4)
    var_6 = 'salt'
    var_7 = bool('salt' in var_5)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer is var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: int(x)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer is var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(str(x), var_2)
    var_4 = lambda x: x.decode(var_2)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/8 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_data'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"text": "data"}'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'binary_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loads_deserializes_payload_correctly. Retrieved 2/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = {}
    var_2 = module_0.loads(var_0, **var_1)
    assert var_2 == 'HELLO'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key1': 'value1'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key1': 'value1'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: x
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_none_data. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_empty_dict. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_nested_data. Retrieved 8/9 statements.
# Partially parsed test_dumps_with_bytes_key. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_different_serializer. Retrieved 3/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'data'
    var_8 = bool('data' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.dumps(var_5)
    var_7 = '1'
    var_8 = bool('1' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)
    var_4 = 'null'
    var_5 = bool('null' in var_3)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    assert var_3 == 'e30.{}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'outer'
    var_3 = 'inner'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_1.dumps(var_6)
    var_8 = 'outer'
    var_9 = bool('outer' in var_7)
    assert var_9 is True
    var_10 = 'inner'
    var_11 = bool('inner' in var_7)
    assert var_11 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'test'
    var_7 = bool('test' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-data'
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)
    var_4 = 'custom-test-data'
    var_5 = bool('custom-test-data' in var_3)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
    var_6 = var_2.serializer
    assert var_6 == b'custom_serializer'
    var_7 = var_2.is_text_serializer
    assert var_7 is False
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key1': 'value1'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key1': 'value1'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 9/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = b'key'
    var_10 = bool(b'key' in var_8)
    assert var_10 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'key'
    var_8 = bool('key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'different-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'test'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'test'
    var_8 = bool('test' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'list-salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = 'list'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)
    var_9 = 'list'
    var_10 = bool('list' in var_8)
    assert var_10 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_dumps_with_empty_data. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_none_data. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_nested_data. Retrieved 8/9 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True
    var_8 = 'value'
    var_9 = bool('value' in var_5)
    assert var_9 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = b'key'
    var_10 = bool(b'key' in var_8)
    assert var_10 is True
    var_11 = b'value'
    var_12 = bool(b'value' in var_8)
    assert var_12 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'another-salt'
    var_7 = var_2.dumps(var_5, var_6)
    var_8 = 'key'
    var_9 = bool('key' in var_7)
    assert var_9 is True
    var_10 = 'value'
    var_11 = bool('value' in var_7)
    assert var_11 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    var_4 = bool(var_3 != '')
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)
    var_4 = bool(var_3 != '')
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.dumps(var_5)
    var_7 = '1'
    var_8 = bool('1' in var_6)
    assert var_8 is True
    var_9 = '2'
    var_10 = bool('2' in var_6)
    assert var_10 is True
    var_11 = '3'
    var_12 = bool('3' in var_6)
    assert var_12 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'outer'
    var_3 = 'inner'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_1.dumps(var_6)
    var_8 = 'outer'
    var_9 = bool('outer' in var_7)
    assert var_9 is True
    var_10 = 'inner'
    var_11 = bool('inner' in var_7)
    assert var_11 is True
    var_12 = 'value'
    var_13 = bool('value' in var_7)
    assert var_13 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'digest_method'
    var_4 = 'SHA256'
    var_5 = {var_3: var_4}
    var_6 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loads_with_valid_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_with_empty_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_with_none_payload. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'some_serialized_data'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b''
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 6/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 1
    var_12 = var_7[var_11]
    var_13 = var_7[1].signer_kwargs
    var_14 = bool(var_7[1].signer_kwargs == {'key_derivation': 'hmac'})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = 'concat'
    var_5 = {var_1: var_4}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'key_derivation'
    var_4 = 'hmac'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_9[0].secret_keys
    var_12 = bool(var_9[0].secret_keys == [b'old-key', b'new-key'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == [b'old-key'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == [b'new-key'])
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = 'custom'
    var_4 = lambda x: var_3
    var_5 = {}
    var_6 = lambda x: var_5
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = module_0.Serializer(var_0, serializer=var_7)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == {'dumps': lambda x: 'custom', 'loads': lambda x: {}})
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'SHA256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/3 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/8 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"text": "data"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'text': 'data'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'\x00\x01\x02'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 10/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'sep'
    var_3 = '?'
    var_4 = {var_2: var_3}
    var_5 = var_1.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 1
    var_9 = var_6[var_8]
    var_10 = var_6[1].sep
    assert var_10 == '?'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 2/9 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 7/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[1].signer_kwargs['digest_method']
    assert var_9 == 'sha256'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = 'sha512'
    var_5 = {var_1: var_4}
    var_6 = 3

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].secret_keys
    var_8 = bool(var_5[0].secret_keys == [b'old-key', b'new-key'])
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_9[0].secret_keys
    var_12 = bool(var_9[0].secret_keys == [b'old-key', b'new-key'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == [b'old-key'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == [b'new-key'])
    assert var_16 is True



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = 'custom'
    var_4 = lambda x: var_3
    var_5 = {}
    var_6 = lambda x: var_5
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = module_0.Serializer(var_0, serializer=var_7)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == {'dumps': lambda x: 'custom', 'loads': lambda x: {}})
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = b'custom'
    var_4 = lambda x: var_3
    var_5 = {}
    var_6 = lambda x: var_5
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = module_0.Serializer(var_0, serializer=var_7)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == {'dumps': lambda x: b'custom', 'loads': lambda x: {}})
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'CustomSigner'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.type(*var_4, **var_5)
    var_7 = module_1.Serializer(var_0, signer=var_6)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = ()
    var_14 = {}
    var_15 = [var_1, var_13, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_7.signer
    var_19 = bool(var_7.signer == var_17)
    assert var_19 is True
    var_20 = var_7.signer_kwargs
    var_21 = bool(var_7.signer_kwargs == {})
    assert var_21 is True
    var_22 = var_7.fallback_signers
    var_23 = bool(var_7.fallback_signers == [])
    assert var_23 is True
    var_24 = var_7.serializer_kwargs
    var_25 = bool(var_7.serializer_kwargs == {})
    assert var_25 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'sha256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'sha256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = 'indent'
    var_3 = True
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, serializer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {'sort_keys': True, 'indent': 4})
    assert var_18 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret1'
    var_7 = 'secret2'
    var_8 = [var_6, var_7]
    var_9 = 'custom-salt'
    var_10 = 'dumps'
    var_11 = 'loads'
    var_12 = 'custom'
    var_13 = lambda x: var_12
    var_14 = {}
    var_15 = lambda x: var_14
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = 'sort_keys'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = 'CustomSigner'
    var_21 = ()
    var_22 = {}
    var_23 = [var_20, var_21, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = {var_0: var_2}
    var_27 = module_1.Serializer(var_8, var_9, var_16, var_19, var_25, var_26, var_5)
    var_28 = var_27.secret_keys
    var_29 = bool(var_27.secret_keys == [b'secret1', b'secret2'])
    assert var_29 is True
    var_30 = var_27.salt
    assert var_30 == b'custom-salt'
    var_31 = var_27.serializer
    var_32 = bool(var_27.serializer == {'dumps': lambda x: 'custom', 'loads': lambda x: {}})
    assert var_32 is True
    var_33 = var_27.is_text_serializer
    assert var_33 is True
    var_34 = ()
    var_35 = {}
    var_36 = [var_20, var_34, var_35]
    var_37 = {}
    var_38 = module_0.type(*var_36, **var_37)
    var_39 = var_27.signer
    var_40 = bool(var_27.signer == var_38)
    assert var_40 is True
    var_41 = var_27.signer_kwargs
    var_42 = bool(var_27.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_42 is True
    var_43 = var_27.fallback_signers
    var_44 = bool(var_27.fallback_signers == var_5)
    assert var_44 is True
    var_45 = var_27.serializer_kwargs
    var_46 = bool(var_27.serializer_kwargs == {'sort_keys': True})
    assert var_46 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'some_serialized_data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dumps_with_bytes_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/8 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"text": "data"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'text': 'data'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'binary_data'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer is var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is True
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer is var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True



# Parsed testcases at query #27
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret-key'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = ';'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': ';'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = ';'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = ':'
    var_10 = {var_0: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_2, var_11]
    var_13 = 'secret-key'
    var_14 = module_1.Serializer(var_13, fallback_signers=var_12)
    var_15 = var_14.secret_keys
    var_16 = bool(var_14.secret_keys == [b'secret-key'])
    assert var_16 is True
    var_17 = var_14.salt
    assert var_17 == b'itsdangerous'
    var_18 = var_14.serializer
    var_19 = var_14.is_text_serializer
    assert var_19 is True
    var_20 = var_14.signer
    var_21 = var_14.signer_kwargs
    var_22 = bool(var_14.signer_kwargs == {})
    assert var_22 is True
    var_23 = var_14.fallback_signers
    var_24 = bool(var_14.fallback_signers == var_12)
    assert var_24 is True
    var_25 = var_14.serializer_kwargs
    var_26 = bool(var_14.serializer_kwargs == {})
    assert var_26 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/16 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_loads_deserializes_payload_correctly. Retrieved 2/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = {}
    var_2 = module_0.loads(var_0, **var_1)
    assert var_2 == 'deserialized_test_payload'



# Parsed testcases at query #30
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True
    var_4 = var_1.secret_keys
    var_5 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_5 is True
    var_6 = var_1.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer == var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is True
    var_10 = var_6.secret_keys
    var_11 = bool(var_6.secret_keys == [b'secret-key'])
    assert var_11 is True
    var_12 = var_6.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = '__init__'
    var_3 = None
    var_4 = lambda self, *args, **kwargs: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = b'secret-key'
    var_10 = module_1.Serializer(var_9, signer=var_8)
    var_11 = var_10.signer
    var_12 = bool(var_10.signer == var_8)
    assert var_12 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: x.decode(var_2)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = b'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.serializer
    var_9 = bool(var_7.serializer == var_5)
    assert var_9 is True
    var_10 = var_7.is_text_serializer
    assert var_10 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_data'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #32
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'sha256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'sha256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serializer_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_with_fallback_signers. Retrieved 7/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key1': 'value1'})
    assert var_6 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'key1': 'value1'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #39
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #41
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = {var_2: var_4}
    var_6 = lambda x: var_5
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'secret'
    var_9 = module_0.Serializer(var_8, serializer=var_7)
    var_10 = var_9.secret_keys
    var_11 = bool(var_9.secret_keys == [b'secret'])
    assert var_11 is True
    var_12 = var_9.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_9.serializer
    var_14 = bool(var_9.serializer == var_7)
    assert var_14 is True
    var_15 = var_9.is_text_serializer
    assert var_15 is True
    var_16 = var_9.signer
    var_17 = var_9.signer_kwargs
    var_18 = bool(var_9.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_9.fallback_signers
    var_20 = bool(var_9.fallback_signers == [])
    assert var_20 is True
    var_21 = var_9.serializer_kwargs
    var_22 = bool(var_9.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = 'custom'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': 'custom'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = 'custom'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = lambda x: var_6
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'secret'
    var_10 = module_0.Serializer(var_9, serializer=var_8)
    var_11 = var_10.secret_keys
    var_12 = bool(var_10.secret_keys == [b'secret'])
    assert var_12 is True
    var_13 = var_10.salt
    assert var_13 == b'itsdangerous'
    var_14 = var_10.serializer
    var_15 = bool(var_10.serializer == var_8)
    assert var_15 is True
    var_16 = var_10.is_text_serializer
    assert var_16 is False
    var_17 = var_10.signer
    var_18 = var_10.signer_kwargs
    var_19 = bool(var_10.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_10.fallback_signers
    var_21 = bool(var_10.fallback_signers == [])
    assert var_21 is True
    var_22 = var_10.serializer_kwargs
    var_23 = bool(var_10.serializer_kwargs == {})
    assert var_23 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'SHA256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #43
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'invalid_json'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer is var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_raises_bad_payload. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 42

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = 'Could not load the payload'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret-key'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret-key'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer is var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret-key'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret-key'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer is var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'SHA256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '--'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '--'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '--'
    var_2 = {var_0: var_1}
    var_3 = '++'
    var_4 = {var_0: var_3}
    var_5 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'key2'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_6 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = 'test'
    var_4 = lambda x: var_3
    var_5 = lambda x: x
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = module_0.Serializer(var_0, serializer=var_6)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == {'dumps': lambda x: 'test', 'loads': lambda x: x})
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'SHA256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = var_6.is_text_serializer
    assert var_11 is True
    var_12 = var_6.signer
    var_13 = var_6.signer_kwargs
    var_14 = bool(var_6.signer_kwargs == {'key_derivation': 'hmac', 'digest_method': 'SHA256'})
    assert var_14 is True
    var_15 = var_6.fallback_signers
    var_16 = bool(var_6.fallback_signers == [])
    assert var_16 is True
    var_17 = var_6.serializer_kwargs
    var_18 = bool(var_6.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'SHA256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = var_7.signer_kwargs
    var_15 = bool(var_7.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_7.fallback_signers
    var_17 = bool(var_7.fallback_signers == var_5)
    assert var_17 is True
    var_18 = var_7.serializer_kwargs
    var_19 = bool(var_7.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #56
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



# Parsed testcases at query #57
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = {var_2: var_4}
    var_6 = lambda x: var_5
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'secret-key'
    var_9 = module_0.Serializer(var_8, serializer=var_7)
    var_10 = var_9.secret_keys
    var_11 = bool(var_9.secret_keys == [b'secret-key'])
    assert var_11 is True
    var_12 = var_9.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_9.serializer
    var_14 = bool(var_9.serializer == var_7)
    assert var_14 is True
    var_15 = var_9.is_text_serializer
    assert var_15 is True
    var_16 = var_9.signer
    var_17 = var_9.signer_kwargs
    var_18 = bool(var_9.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_9.fallback_signers
    var_20 = bool(var_9.fallback_signers == [])
    assert var_20 is True
    var_21 = var_9.serializer_kwargs
    var_22 = bool(var_9.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = lambda x: var_6
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'secret-key'
    var_10 = module_0.Serializer(var_9, serializer=var_8)
    var_11 = var_10.secret_keys
    var_12 = bool(var_10.secret_keys == [b'secret-key'])
    assert var_12 is True
    var_13 = var_10.salt
    assert var_13 == b'itsdangerous'
    var_14 = var_10.serializer
    var_15 = bool(var_10.serializer == var_8)
    assert var_15 is True
    var_16 = var_10.is_text_serializer
    assert var_16 is False
    var_17 = var_10.signer
    var_18 = var_10.signer_kwargs
    var_19 = bool(var_10.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_10.fallback_signers
    var_21 = bool(var_10.fallback_signers == [])
    assert var_21 is True
    var_22 = var_10.serializer_kwargs
    var_23 = bool(var_10.serializer_kwargs == {})
    assert var_23 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_dumps_with_bytes_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_iter_unsigners_with_dict_fallback. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 1
    var_12 = var_7[var_11]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 4/6 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test-data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = '{"key": "value"}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test-data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #64
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret-key'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret-key'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #67
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = {}
    var_2 = [var_1]
    var_3 = module_0.Serializer(var_0, fallback_signers=var_2)
    var_4 = var_3.fallback_signers
    var_5 = bool(var_3.fallback_signers is not None)
    assert var_5 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'42'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #70
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer is var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is True
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer is var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer is var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #71
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is False
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_loads_deserializes_payload. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'some_serialized_data'
    var_4 = var_2.loads(var_3)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #76
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_iter_unsigners_with_dict_fallback. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 1
    var_12 = var_7[var_11]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret-key'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret-key'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret-key'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key1': 'value1'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key1': 'value1'})
    assert var_16 is True



# Parsed testcases at query #81
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #84
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = {var_2: var_4}
    var_6 = lambda x: var_5
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'secret'
    var_9 = module_0.Serializer(var_8, serializer=var_7)
    var_10 = var_9.secret_keys
    var_11 = bool(var_9.secret_keys == [b'secret'])
    assert var_11 is True
    var_12 = var_9.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_9.serializer
    var_14 = bool(var_9.serializer == var_7)
    assert var_14 is True
    var_15 = var_9.is_text_serializer
    assert var_15 is True
    var_16 = var_9.signer
    var_17 = var_9.signer_kwargs
    var_18 = bool(var_9.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_9.fallback_signers
    var_20 = bool(var_9.fallback_signers == [])
    assert var_20 is True
    var_21 = var_9.serializer_kwargs
    var_22 = bool(var_9.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = lambda x: var_6
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'secret'
    var_10 = module_0.Serializer(var_9, serializer=var_8)
    var_11 = var_10.secret_keys
    var_12 = bool(var_10.secret_keys == [b'secret'])
    assert var_12 is True
    var_13 = var_10.salt
    assert var_13 == b'itsdangerous'
    var_14 = var_10.serializer
    var_15 = bool(var_10.serializer == var_8)
    assert var_15 is True
    var_16 = var_10.is_text_serializer
    assert var_16 is False
    var_17 = var_10.signer
    var_18 = var_10.signer_kwargs
    var_19 = bool(var_10.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_10.fallback_signers
    var_21 = bool(var_10.fallback_signers == [])
    assert var_21 is True
    var_22 = var_10.serializer_kwargs
    var_23 = bool(var_10.serializer_kwargs == {})
    assert var_23 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'



# Parsed testcases at query #86
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = lambda x: {var_4: x}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)
    var_9 = var_8.secret_keys
    var_10 = bool(var_8.secret_keys == [b'secret'])
    assert var_10 is True
    var_11 = var_8.salt
    assert var_11 == b'itsdangerous'
    var_12 = var_8.serializer
    var_13 = bool(var_8.serializer == var_6)
    assert var_13 is True
    var_14 = var_8.is_text_serializer
    assert var_14 is False
    var_15 = var_8.signer
    var_16 = var_8.signer_kwargs
    var_17 = bool(var_8.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_8.fallback_signers
    var_19 = bool(var_8.fallback_signers == [])
    assert var_19 is True
    var_20 = var_8.serializer_kwargs
    var_21 = bool(var_8.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'



# Parsed testcases at query #88
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #90
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret-key-1', b'secret-key-2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret-key'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret-key'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret-key'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = {var_0: var_1}
    var_10 = (var_8, var_9)
    var_11 = [var_2, var_10]
    var_12 = 'secret-key'
    var_13 = module_1.Serializer(var_12, fallback_signers=var_11)
    var_14 = var_13.secret_keys
    var_15 = bool(var_13.secret_keys == [b'secret-key'])
    assert var_15 is True
    var_16 = var_13.salt
    assert var_16 == b'itsdangerous'
    var_17 = var_13.serializer
    var_18 = var_13.is_text_serializer
    assert var_18 is True
    var_19 = var_13.signer
    var_20 = var_13.signer_kwargs
    var_21 = bool(var_13.signer_kwargs == {})
    assert var_21 is True
    var_22 = var_13.fallback_signers
    var_23 = bool(var_13.fallback_signers == var_11)
    assert var_23 is True
    var_24 = var_13.serializer_kwargs
    var_25 = bool(var_13.serializer_kwargs == {})
    assert var_25 is True



# Parsed testcases at query #91
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = bool(var_7.serializer == var_5)
    assert var_12 is True
    var_13 = var_7.is_text_serializer
    assert var_13 is True
    var_14 = var_7.signer
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = 'secret'
    var_7 = module_1.Serializer(var_6, signer=var_5)
    var_8 = var_7.secret_keys
    var_9 = bool(var_7.secret_keys == [b'secret'])
    assert var_9 is True
    var_10 = var_7.salt
    assert var_10 == b'itsdangerous'
    var_11 = var_7.serializer
    var_12 = var_7.is_text_serializer
    assert var_12 is True
    var_13 = var_7.signer
    var_14 = bool(var_7.signer == var_5)
    assert var_14 is True
    var_15 = var_7.signer_kwargs
    var_16 = bool(var_7.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_7.fallback_signers
    var_18 = bool(var_7.fallback_signers == [])
    assert var_18 is True
    var_19 = var_7.serializer_kwargs
    var_20 = bool(var_7.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '?'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = '!'
    var_10 = {var_0: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_2, var_11]
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, fallback_signers=var_12)
    var_15 = var_14.secret_keys
    var_16 = bool(var_14.secret_keys == [b'secret'])
    assert var_16 is True
    var_17 = var_14.salt
    assert var_17 == b'itsdangerous'
    var_18 = var_14.serializer
    var_19 = var_14.is_text_serializer
    assert var_19 is True
    var_20 = var_14.signer
    var_21 = var_14.signer_kwargs
    var_22 = bool(var_14.signer_kwargs == {})
    assert var_22 is True
    var_23 = var_14.fallback_signers
    var_24 = bool(var_14.fallback_signers == var_12)
    assert var_24 is True
    var_25 = var_14.serializer_kwargs
    var_26 = bool(var_14.serializer_kwargs == {})
    assert var_26 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: {var_2: x}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'CustomSigner'
    var_7 = ()
    var_8 = {}
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'sep'
    var_13 = '?'
    var_14 = {var_12: var_13}
    var_15 = ()
    var_16 = {}
    var_17 = [var_6, var_15, var_16]
    var_18 = {}
    var_19 = module_0.type(*var_17, **var_18)
    var_20 = '!'
    var_21 = {var_12: var_20}
    var_22 = (var_19, var_21)
    var_23 = [var_14, var_22]
    var_24 = 'secret1'
    var_25 = 'secret2'
    var_26 = [var_24, var_25]
    var_27 = 'custom-salt'
    var_28 = 'indent'
    var_29 = 4
    var_30 = {var_28: var_29}
    var_31 = {var_12: var_13}
    var_32 = module_1.Serializer(var_26, var_27, var_5, var_30, var_11, var_31, var_23)
    var_33 = var_32.secret_keys
    var_34 = bool(var_32.secret_keys == [b'secret1', b'secret2'])
    assert var_34 is True
    var_35 = var_32.salt
    assert var_35 == b'custom-salt'
    var_36 = var_32.serializer
    var_37 = bool(var_32.serializer == var_5)
    assert var_37 is True
    var_38 = var_32.is_text_serializer
    assert var_38 is True
    var_39 = var_32.signer
    var_40 = bool(var_32.signer == var_11)
    assert var_40 is True
    var_41 = var_32.signer_kwargs
    var_42 = bool(var_32.signer_kwargs == {'sep': '?'})
    assert var_42 is True
    var_43 = var_32.fallback_signers
    var_44 = bool(var_32.fallback_signers == var_23)
    assert var_44 is True
    var_45 = var_32.serializer_kwargs
    var_46 = bool(var_32.serializer_kwargs == {'indent': 4})
    assert var_46 is True



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 5/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = str(var_3)
    var_5 = 'Could not load the payload because an exception occurred on unserializing the data.'
    var_6 = bool('Could not load the payload because an exception occurred on unserializing the data.' in var_4)
    assert var_6 is True



# Parsed testcases at query #94
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True



# Parsed testcases at query #96
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/16 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_6 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_key
    assert var_5 == b'key3'



# Parsed testcases at query #99
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret-key'



# Parsed testcases at query #101
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_bytes_input. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_empty_object. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_list_object. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_none_object. Retrieved 4/5 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'key'
    var_7 = bool('key' in var_5)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = b'key'
    var_8 = bool(b'key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'
    var_6 = var_1.dumps(var_4, var_5)
    var_7 = 'key'
    var_8 = bool('key' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'bytes-input'
    var_3 = var_1.dumps(var_2)
    var_4 = 'bytes-input'
    var_5 = bool('bytes-input' in var_3)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.dumps(var_5)
    var_7 = '1'
    var_8 = bool('1' in var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)
    var_4 = 'null'
    var_5 = bool('null' in var_3)
    assert var_5 is True



# Parsed testcases at query #103
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid_payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_6.salt
    assert var_9 == b'itsdangerous'
    var_10 = var_6.serializer
    var_11 = bool(var_6.serializer == var_4)
    assert var_11 is True
    var_12 = var_6.is_text_serializer
    assert var_12 is False
    var_13 = var_6.signer
    var_14 = var_6.signer_kwargs
    var_15 = bool(var_6.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_6.fallback_signers
    var_17 = bool(var_6.fallback_signers == [])
    assert var_17 is True
    var_18 = var_6.serializer_kwargs
    var_19 = bool(var_6.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret-key'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_iter_unsigners_default. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 10/24 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 17/20 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = var_3[0].secret_keys
    var_8 = bool(var_3[0].secret_keys == ['secret-key'])
    assert var_8 is True
    var_9 = var_3[0].salt
    assert var_9 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[0].secret_keys
    var_9 = bool(var_4[0].secret_keys == ['secret-key'])
    assert var_9 is True
    var_10 = var_4[0].salt
    assert var_10 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'md5'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = var_7[0].secret_keys
    var_12 = bool(var_7[0].secret_keys == ['secret-key'])
    assert var_12 is True
    var_13 = var_7[0].salt
    assert var_13 == b'itsdangerous'
    var_14 = 1
    var_15 = var_7[var_14]
    var_16 = var_7[1].secret_keys
    var_17 = bool(var_7[1].secret_keys == ['secret-key'])
    assert var_17 is True
    var_18 = var_7[1].salt
    assert var_18 == b'itsdangerous'
    var_19 = var_7[1].digest_method.name
    assert var_19 == 'md5'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'md5'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'md5'
    var_3 = {var_1: var_2}
    var_4 = 'sha256'
    var_5 = {var_1: var_4}
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = 3

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_5[0].secret_keys
    var_10 = bool(var_5[0].secret_keys == ['old-key', 'new-key'])
    assert var_10 is True
    var_11 = var_5[0].salt
    assert var_11 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'
    var_4 = 'md5'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = var_9[0].secret_keys
    var_14 = bool(var_9[0].secret_keys == ['old-key', 'new-key'])
    assert var_14 is True
    var_15 = var_9[0].salt
    assert var_15 == b'itsdangerous'
    var_16 = 1
    var_17 = var_9[var_16]
    var_18 = var_9[1].secret_keys
    var_19 = bool(var_9[1].secret_keys == ['old-key'])
    assert var_19 is True
    var_20 = var_9[1].salt
    assert var_20 == b'itsdangerous'
    var_21 = var_9[1].digest_method.name
    assert var_21 == 'md5'
    var_22 = 2
    var_23 = var_9[var_22]
    var_24 = var_9[2].secret_keys
    var_25 = bool(var_9[2].secret_keys == ['new-key'])
    assert var_25 is True
    var_26 = var_9[2].salt
    assert var_26 == b'itsdangerous'
    var_27 = var_9[2].digest_method.name
    assert var_27 == 'md5'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = {var_2: var_4}
    var_6 = lambda x: var_5
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'secret-key'
    var_9 = module_0.Serializer(var_8, serializer=var_7)
    var_10 = var_9.secret_keys
    var_11 = bool(var_9.secret_keys == [b'secret-key'])
    assert var_11 is True
    var_12 = var_9.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_9.serializer
    var_14 = bool(var_9.serializer == var_7)
    assert var_14 is True
    var_15 = var_9.is_text_serializer
    assert var_15 is True
    var_16 = var_9.signer
    var_17 = var_9.signer_kwargs
    var_18 = bool(var_9.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_9.fallback_signers
    var_20 = bool(var_9.fallback_signers == [])
    assert var_20 is True
    var_21 = var_9.serializer_kwargs
    var_22 = bool(var_9.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_16 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = ';'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'sep': ';'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

def test_case_0():
    var_0 = 'sep'
    var_1 = ';'
    var_2 = {var_0: var_1}
    var_3 = ':'
    var_4 = {var_0: var_3}
    var_5 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'bytes'
    var_3 = lambda x: var_2
    var_4 = 'bytes'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = lambda x: var_6
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'secret-key'
    var_10 = module_0.Serializer(var_9, serializer=var_8)
    var_11 = var_10.secret_keys
    var_12 = bool(var_10.secret_keys == [b'secret-key'])
    assert var_12 is True
    var_13 = var_10.salt
    assert var_13 == b'itsdangerous'
    var_14 = var_10.serializer
    var_15 = bool(var_10.serializer == var_8)
    assert var_15 is True
    var_16 = var_10.is_text_serializer
    assert var_16 is False
    var_17 = var_10.signer
    var_18 = var_10.signer_kwargs
    var_19 = bool(var_10.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_10.fallback_signers
    var_21 = bool(var_10.fallback_signers == [])
    assert var_21 is True
    var_22 = var_10.serializer_kwargs
    var_23 = bool(var_10.serializer_kwargs == {})
    assert var_23 is True



# Parsed testcases at query #107
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #108
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
    assert var_9 is True
    var_10 = var_1.fallback_signers
    var_11 = bool(var_1.fallback_signers == [])
    assert var_11 is True
    var_12 = var_1.serializer_kwargs
    var_13 = bool(var_1.serializer_kwargs == {})
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = var_3.is_text_serializer
    assert var_8 is True
    var_9 = var_3.signer
    var_10 = var_3.signer_kwargs
    var_11 = bool(var_3.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_3.fallback_signers
    var_13 = bool(var_3.fallback_signers == [])
    assert var_13 is True
    var_14 = var_3.serializer_kwargs
    var_15 = bool(var_3.serializer_kwargs == {})
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = {var_2: var_4}
    var_6 = lambda x: var_5
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = 'secret-key'
    var_9 = module_0.Serializer(var_8, serializer=var_7)
    var_10 = var_9.secret_keys
    var_11 = bool(var_9.secret_keys == [b'secret-key'])
    assert var_11 is True
    var_12 = var_9.salt
    assert var_12 == b'itsdangerous'
    var_13 = var_9.serializer
    var_14 = bool(var_9.serializer == var_7)
    assert var_14 is True
    var_15 = var_9.is_text_serializer
    assert var_15 is True
    var_16 = var_9.signer
    var_17 = var_9.signer_kwargs
    var_18 = bool(var_9.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_9.fallback_signers
    var_20 = bool(var_9.fallback_signers == [])
    assert var_20 is True
    var_21 = var_9.serializer_kwargs
    var_22 = bool(var_9.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = 'custom'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = lambda x: var_6
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = 'secret-key'
    var_10 = module_0.Serializer(var_9, serializer=var_8)
    var_11 = var_10.secret_keys
    var_12 = bool(var_10.secret_keys == [b'secret-key'])
    assert var_12 is True
    var_13 = var_10.salt
    assert var_13 == b'itsdangerous'
    var_14 = var_10.serializer
    var_15 = bool(var_10.serializer == var_8)
    assert var_15 is True
    var_16 = var_10.is_text_serializer
    assert var_16 is False
    var_17 = var_10.signer
    var_18 = var_10.signer_kwargs
    var_19 = bool(var_10.signer_kwargs == {})
    assert var_19 is True
    var_20 = var_10.fallback_signers
    var_21 = bool(var_10.fallback_signers == [])
    assert var_21 is True
    var_22 = var_10.serializer_kwargs
    var_23 = bool(var_10.serializer_kwargs == {})
    assert var_23 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {'key': 'value'})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = var_5.is_text_serializer
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == var_3)
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = var_4.is_text_serializer
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = var_2.is_text_serializer
    assert var_7 is True
    var_8 = var_2.signer
    var_9 = var_2.signer_kwargs
    var_10 = bool(var_2.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_2.fallback_signers
    var_12 = bool(var_2.fallback_signers == [])
    assert var_12 is True
    var_13 = var_2.serializer_kwargs
    var_14 = bool(var_2.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



