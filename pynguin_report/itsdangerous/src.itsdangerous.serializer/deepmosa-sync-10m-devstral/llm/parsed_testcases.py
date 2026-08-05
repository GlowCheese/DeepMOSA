####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 6/14 statements.
# Partially parsed test_iter_unsigners_custom_signer_class. Retrieved 2/10 statements.


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
    var_6 = bool(var_0)
    assert var_6 is True

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
    var_8 = bool(var_5[0].secret_keys == ['old-key', 'new-key'])
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
    var_12 = bool(var_9[0].secret_keys == ['old-key', 'new-key'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == ['old-key'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == ['new-key'])
    assert var_16 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loads_deserializes_payload. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_signer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_text_serializer. Retrieved 1/4 statements.
# Partially parsed test_serializer_bytes_serializer. Retrieved 1/10 statements.
# Partially parsed test_serializer_signer_custom. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

def test_case_0():
    var_0 = b'secret'

def test_case_0():
    var_0 = b'secret'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer_kwargs
    var_3 = bool(var_1.signer_kwargs == {})
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer_kwargs
    var_3 = bool(var_1.serializer_kwargs == {})
    assert var_3 is True

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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_binary_serializer. Retrieved 2/10 statements.


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
    var_1 = b'custom_data'

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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
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



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-json-data'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 12/17 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key_derivation'
    var_3 = 'hmac'
    var_4 = {var_2: var_3}
    var_5 = var_1.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = 1
    var_11 = var_6[var_10]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_8 = bool(var_6.serializer == var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
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
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

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
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serializer_constructor_custom_values. Retrieved 11/12 statements.
# Partially parsed test_serializer_constructor_bytes_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_fallback_signers_custom. Retrieved 6/9 statements.


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

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = 'indent'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'sep'
    var_6 = '|'
    var_7 = {var_5: var_6}
    var_8 = ':'
    var_9 = {var_5: var_8}
    var_10 = [var_9]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = ':'
    var_3 = {var_1: var_2}
    var_4 = '|'
    var_5 = {var_1: var_4}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_exception_raises_bad_payload. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 9/10 statements.
# Partially parsed test_dumps_empty_object. Retrieved 4/5 statements.
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
    var_9 = b'key'
    var_10 = bool(b'key' in var_8)
    assert var_10 is True

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
    var_0 = 'another-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'test'
    var_4 = 123
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'salt'
    var_4 = {var_3: var_1}
    var_5 = var_2.dumps(var_4)
    var_6 = 'salt'
    var_7 = bool('salt' in var_5)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_dumps_with_bytes_data. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_empty_data. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_nested_data. Retrieved 8/9 statements.
# Partially parsed test_dumps_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_none_data. Retrieved 4/5 statements.


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
    var_6 = 'another-salt'
    var_7 = var_2.dumps(var_5, var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'binary-data'
    var_3 = var_1.dumps(var_2)

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
    var_2 = 'nested'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_1.dumps(var_6)
    var_8 = 'nested'
    var_9 = bool('nested' in var_7)
    assert var_9 is True

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



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = {}
    var_2 = [var_1]
    var_3 = module_0.Serializer(var_0, fallback_signers=var_2)
    var_4 = var_3.fallback_signers
    var_5 = bool(var_3.fallback_signers is not None)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


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



# Parsed testcases at query #23
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'invalid_json'
    var_4 = var_2.loads(var_3)



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #25
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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
    var_1 = b'custom-serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
    var_6 = var_2.serializer
    assert var_6 == b'custom-serializer'
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



# Parsed testcases at query #28
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'secret'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_loads_with_valid_payload. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b''
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'invalid_json'
    var_4 = var_2.loads(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 2/10 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'data'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 6/14 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 11/13 statements.


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
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[0].secret_key
    assert var_8 == b'secret-key'
    var_9 = var_4[0].salt
    assert var_9 == b'custom-salt'

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
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_5[0].secret_key
    assert var_9 == b'new-key'

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
    var_11 = var_9[0].secret_key
    assert var_11 == b'new-key'
    var_12 = var_9[1].secret_key
    assert var_12 == b'old-key'
    var_13 = var_9[2].secret_key
    assert var_13 == b'new-key'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_string_secret_key. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_secret_key. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_list_secret_key. Retrieved 4/5 statements.
# Partially parsed test_serializer_constructor_with_none_salt. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 2/8 statements.
# Partially parsed test_serializer_constructor_with_serializer_kwargs. Retrieved 5/6 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_signer_kwargs. Retrieved 5/6 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'

def test_case_0():
    var_0 = 'secret'
    var_1 = None

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'indent'
    var_3 = 4
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'key_derivation'
    var_3 = 'hmac'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'
    var_5 = 'salt'

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



# Parsed testcases at query #3
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
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
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



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_payload'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_raises_badpayload. Retrieved 2/6 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'invalid json'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    assert var_3 == 'custom_serializer'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_signer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/11 statements.


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
    var_7 = {}
    var_8 = var_2.loads(var_6, var_1, **var_7)
    var_9 = bool(var_8 == var_5)
    assert var_9 is True
    var_10 = 'salt2'
    var_11 = {}
    var_12 = var_2.loads(var_6, var_10, **var_11)
    var_13 = bool(False)
    assert var_13 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    var_4 = {}
    var_5 = var_1.loads(var_3, **var_4)
    var_6 = bool(var_5 == {})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)
    var_4 = {}
    var_5 = var_1.loads(var_3, **var_4)
    assert var_5 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_4, var_9: var_5}
    var_11 = {var_2: var_7, var_3: var_10}
    var_12 = var_1.dumps(var_11)
    var_13 = {}
    var_14 = var_1.loads(var_12, **var_13)
    var_15 = bool(var_14 == var_11)
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = {}
    var_7 = var_1.loads(var_5, **var_6)
    var_8 = bool(var_7 == var_4)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.
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

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 4/7 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_invalid_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'
    var_2 = 'Could not load the payload'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 6/14 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallbacks. Retrieved 17/20 statements.


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
    var_6 = bool(var_0)
    assert var_6 is True

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
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = var_9[0].secret_keys
    var_14 = bool(var_9[0].secret_keys == [b'old-key', b'new-key'])
    assert var_14 is True
    var_15 = 1
    var_16 = var_9[var_15]
    var_17 = var_9[1].secret_keys
    var_18 = bool(var_9[1].secret_keys == [b'old-key'])
    assert var_18 is True
    var_19 = 2
    var_20 = var_9[var_19]
    var_21 = var_9[2].secret_keys
    var_22 = bool(var_9[2].secret_keys == [b'new-key'])
    assert var_22 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_signer_kwargs. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_custom_fallback_signers. Retrieved 2/5 statements.


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

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True
    var_5 = var_2.secret_keys
    var_6 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_2.salt
    assert var_7 == b'custom-salt'
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
    var_0 = b'secret-key'

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = b'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer
    var_6 = var_4.is_text_serializer
    assert var_6 is True
    var_7 = var_4.secret_keys
    var_8 = bool(var_4.secret_keys == [b'secret-key'])
    assert var_8 is True
    var_9 = var_4.salt
    assert var_9 == b'itsdangerous'
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
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.serializer
    var_5 = var_3.is_text_serializer
    assert var_5 is True
    var_6 = var_3.secret_keys
    var_7 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_7 is True
    var_8 = var_3.salt
    assert var_8 == b'itsdangerous'
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
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True
    var_5 = var_2.secret_keys
    var_6 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_6 is True
    var_7 = var_2.salt
    assert var_7 is None
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



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 is None



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_fallback_signers. Retrieved 5/8 statements.


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
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer == var_4)
    assert var_8 is True
    var_9 = var_6.is_text_serializer
    assert var_9 is False

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = b'secret'

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
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_signer_is_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret-key'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_binary_serializer. Retrieved 3/11 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 3/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'data'
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_iter_unsigners_with_dict_fallback. Retrieved 13/15 statements.


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



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.is_text_serializer
    assert var_2 is False
    var_3 = b'test'
    var_4 = var_1.load_payload(var_3)
    assert var_4 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_iter_unsigners_with_dict_fallback. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
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



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.


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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #25
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
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
    var_2 = module_0.Serializer(var_0, signer=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
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
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
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
    var_2 = module_0.Serializer(var_0, serializer_kwargs=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
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
    var_2 = module_0.Serializer(var_0, signer_kwargs=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous'
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 17/18 statements.


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
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret1'
    var_6 = 'secret2'
    var_7 = [var_5, var_6]
    var_8 = 'custom_salt'
    var_9 = 'indent'
    var_10 = 2
    var_11 = {var_9: var_10}
    var_12 = 'sep'
    var_13 = '|'
    var_14 = {var_12: var_13}
    var_15 = {var_12: var_13}
    var_16 = [var_15]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/16 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_custom_fallback_signers. Retrieved 1/3 statements.


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
    var_1 = lambda x: x
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer
    var_3 = var_1.signer_kwargs
    var_4 = bool(var_1.signer_kwargs == {})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'secret'

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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'

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
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = module_0.Serializer(var_3, serializer_kwargs=var_2)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == var_2)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == var_2)
    assert var_6 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serializer_with_custom_signer. Retrieved 1/2 statements.
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
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'sep': '|'})
    assert var_6 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'sep'
    var_4 = '|'
    var_5 = {var_3: var_4}
    var_6 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #34
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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'secret2'



