####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
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
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
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
    var_16 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
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
    var_15 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer_str. Retrieved 1/6 statements.
# Partially parsed test_serializer_constructor_with_serializer_bytes. Retrieved 1/6 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/5 statements.


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
    var_0 = 'secret'
    var_1 = b'custom_salt'
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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'digest_method': 'sha256'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'salt': 'fallback'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #3
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = '{}'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'{}'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is False

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dumps_returns_bytes_for_bytes_serializer. Retrieved 6/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'test'
    var_5 = var_3.dumps(var_4)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 4/15 statements.
# Partially parsed test_load_payload_with_custom_serializer_override. Retrieved 2/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer_non_utf8. Retrieved 5/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'hello'
    var_3 = var_1.dump_payload(var_2)
    var_4 = var_1.load_payload(var_3)
    assert var_4 == 'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = True
    var_3 = var_1.dump_payload(var_2)
    var_4 = None
    var_5 = var_1.load_payload(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_with_custom_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer_class. Retrieved 1/5 statements.
# Partially parsed test_constructor_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_fallback_signers_class. Retrieved 1/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'mysecret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'mysecret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'mysecretbytes'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'mysecretbytes'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'mysalt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'mysalt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'

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
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_iter_unsigners_returns_default_signer_first. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 1)
    assert var_5 is True
    var_6 = 0
    var_7 = var_3[var_6]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    var_9 = bool(var_8 > 1)
    assert var_9 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 >= 2)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = '__iter__'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = '__next__'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_overridden_serializer. Retrieved 6/13 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 4/14 statements.
# Partially parsed test_load_payload_with_text_overridden_serializer. Retrieved 6/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_custom_signer. Retrieved 1/5 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.serializer
    var_14 = bool(var_12.serializer == var_10)
    assert var_14 is True
    var_15 = var_12.is_text_serializer
    assert var_15 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomBytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is False

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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomBytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = b'custom_salt'
    var_15 = module_1.Serializer(var_13, var_14, var_12)
    var_16 = var_15.salt
    assert var_16 == b'custom_salt'
    var_17 = var_15.serializer
    var_18 = bool(var_15.serializer == var_12)
    assert var_18 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomBytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_positional. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_keyword. Retrieved 1/2 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
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
    var_1 = b'itsdangerous'

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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
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



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fallback_signers_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 16/19 statements.


import builtins as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'{"key": "value"}'
    var_6 = lambda self, obj: var_5
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = lambda self, s: var_9
    var_11 = {var_3: var_6, var_4: var_10}
    var_12 = [var_1, var_2, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = {var_7: var_8}
    var_17 = {}
    var_18 = module_1.dumps(var_16, **var_17)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.


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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_iter_unsigners_fallback_signers_not_dict_or_tuple. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 5/17 statements.


def test_case_0():
    var_0 = b'test-secret'
    var_1 = 'digest_method'
    var_2 = b'test-salt'
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serializer_init_with_serializer_does_not_use_default. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = [var_1]
    var_3 = module_0.Serializer(var_0, fallback_signers=var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    var_8 = bool(var_5[0] is not None)
    assert var_8 is True
    var_9 = var_5[1]
    var_10 = bool(var_5[1] is not None)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_iter_unsigners_fallback_tuple_handling. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'extra'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test_salt'
    var_5 = 1



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_init_predicate_line28_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_secret'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_exception_raised_when_is_text_false_and_loads_fails. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid_bytes'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_init_with_custom_str_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer_kwargs
    var_3 = bool(var_1.signer_kwargs == {})
    assert var_3 is True

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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer_kwargs
    var_3 = bool(var_1.serializer_kwargs == {})
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_key
    assert var_5 == b'key3'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'key'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serializer_constructor_default_serializer. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'digest_method'
    var_5 = 'sha256'
    var_6 = {var_4: var_5}
    var_7 = module_0.Serializer(var_0, serializer_kwargs=var_3, signer_kwargs=var_6)
    var_8 = var_7.serializer_kwargs
    var_9 = bool(var_7.serializer_kwargs == {'indent': 2})
    assert var_9 is True
    var_10 = var_7.signer_kwargs
    var_11 = bool(var_7.signer_kwargs == {'digest_method': 'sha256'})
    assert var_11 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 10/21 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #29
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_string'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'"test"'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_serializer_and_is_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_all_parameters. Retrieved 10/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'my-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'my-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = bool(var_1.serializer is var_1.default_serializer)
    assert var_3 is True

def test_case_0():
    var_0 = 'my-secret-key'

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer_kwargs
    var_3 = bool(var_1.signer_kwargs == {})
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer_kwargs
    var_3 = bool(var_1.serializer_kwargs == {})
    assert var_3 is True

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'my-salt'
    var_2 = 'sort_keys'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}
    var_9 = [var_8]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 11/17 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'none'
    var_9 = {var_7: var_8}
    var_10 = 'digest_method'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_results_in_false_predicate. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'test'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret-key'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dumps_returns_expected_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_string'
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_2.dumps(var_8)
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = var_2.dumps(var_12)
    var_14 = None
    var_15 = var_2.dumps(var_14)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.dumps(var_3)
    assert var_4 == '123'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    assert var_4 == 'null'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = True
    var_4 = var_2.dumps(var_3)
    assert var_4 == 'true'
    var_5 = False
    var_6 = var_2.dumps(var_5)
    assert var_6 == 'false'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 'two'
    var_5 = 3.0
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)
    assert var_7 == '[1, "two", 3.0]'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'name'
    var_4 = 'value'
    var_5 = 'test'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.dumps(var_7)
    assert var_8 == '{"name": "test", "value": 42}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'level1'
    var_4 = 'level2'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}
    var_11 = var_2.dumps(var_10)
    assert var_11 == '{"level1": {"level2": [1, 2, 3]}}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = []
    var_4 = var_2.dumps(var_3)
    assert var_4 == '[]'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = {}
    var_4 = var_2.dumps(var_3)



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #41
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{invalid json}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_signer_class. Retrieved 1/4 statements.


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
    var_0 = b'bytes-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    assert var_4 == 'test'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer_text. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = 'key_derivation'
    var_3 = 'none'
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = module_0.Serializer(var_0, fallback_signers=var_5)
    var_7 = var_6.fallback_signers
    var_8 = bool(var_6.fallback_signers == [{}, {'key_derivation': 'none'}])
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 5/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 7/10 statements.
# Partially parsed test_serializer_constructor_with_all_arguments. Retrieved 30/33 statements.


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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'abc'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer == var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'abc'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer == var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is False

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = 'sign'
    var_2 = 'unsign'
    var_3 = lambda self, x: x
    var_4 = lambda self, x: x
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'secret'

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import builtins as module_0

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'abc'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'CustomSigner'
    var_13 = 'sign'
    var_14 = 'unsign'
    var_15 = lambda self, x: x
    var_16 = lambda self, x: x
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'key1'
    var_19 = b'key2'
    var_20 = [var_18, var_19]
    var_21 = b'salt'
    var_22 = 'indent'
    var_23 = 2
    var_24 = {var_22: var_23}
    var_25 = 'key_derivation'
    var_26 = 'hmac'
    var_27 = {var_25: var_26}
    var_28 = 'algorithm'
    var_29 = 'sha512'
    var_30 = {var_28: var_29}
    var_31 = [var_30]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dumps_text_serializer_returns_str. Retrieved 5/7 statements.
# Partially parsed test_dumps_bytes_serializer_returns_bytes. Retrieved 5/11 statements.
# Partially parsed test_dumps_default_salt. Retrieved 4/5 statements.
# Partially parsed test_dumps_custom_salt. Retrieved 5/6 statements.
# Partially parsed test_dumps_with_serializer_kwargs. Retrieved 8/14 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = 'custom_salt'
    var_4 = var_1.dumps(var_2, var_3)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_loads_returns_any_for_integer_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_returns_any_for_list_payload. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b''
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None or var_4 is None)
    assert var_5 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'test-secret'
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'ensure_ascii': False})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = b'secret_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'ensure_ascii': False})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True



# Parsed testcases at query #56
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'bytes'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

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
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

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



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_binary_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_6 = bool(var_1.is_text_serializer)
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #59
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    assert var_4 == b'null'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    assert var_4 == b'42'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.dumps(var_3)
    assert var_4 == b'"hello"'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)
    assert var_7 == b'[1, 2, 3]'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    assert var_6 == b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = True
    var_4 = var_2.dumps(var_3)
    assert var_4 == b'true'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)
    assert var_4 == b'3.14'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_iter_unsigners_returns_default_signer_first. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_yields_fallback_signers. Retrieved 1/7 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = len(var_4)
    assert var_7 == 1

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'default_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'custom_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = [var_2]
    var_5 = module_0.Serializer(var_3, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[1].key_derivation
    assert var_9 == 'hmac'

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_8.secret_keys
    var_10 = len(var_9)
    assert var_10 == 2



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 8/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/15 statements.


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
    var_6 = bool(var_1.is_text_serializer)
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'pepper'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'none'
    var_12 = {var_10: var_11}
    var_13 = [var_12]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'"test"'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_when_is_text_false_and_loads_fails. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #65
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #66
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, obj: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'bytes'
    var_5 = lambda self, obj: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is False

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = {}
    var_7 = {}
    var_8 = module_1.dumps(var_6, **var_7)
    var_9 = var_1.is_text_serializer
    var_10 = var_1.signer
    var_11 = var_1.signer_kwargs
    var_12 = bool(var_1.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_1.fallback_signers
    var_14 = bool(var_1.fallback_signers == [])
    assert var_14 is True
    var_15 = var_1.serializer_kwargs
    var_16 = bool(var_1.serializer_kwargs == {})
    assert var_16 is True

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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'test'])
    assert var_3 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/2 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'none'
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



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/24 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #71
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = var_2.loads(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_string'
    var_4 = var_2.loads(var_3)
    assert var_4 == 'test_string'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    assert var_4 == 123

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'bytes_data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_constructor_with_serializer. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_signer. Retrieved 1/2 statements.


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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

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
    var_0 = b'secret'
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
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True



# Parsed testcases at query #73
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #74
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_string'
    var_4 = var_2.dumps(var_3)
    assert var_4 == 'test_string'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.dumps(var_3)
    assert var_4 == 123

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"valid": true}'



# Parsed testcases at query #76
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.serializer
    var_14 = bool(var_12.serializer == var_10)
    assert var_14 is True
    var_15 = var_12.is_text_serializer
    assert var_15 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BinarySerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.is_text_serializer
    assert var_15 is False

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = [var_2]
    var_5 = module_0.Serializer(var_3, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [var_2])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_bytes. Retrieved 13/16 statements.
# Partially parsed test_dumps_returns_string_when_serializer_is_text. Retrieved 13/16 statements.
# Partially parsed test_dumps_includes_signature. Retrieved 4/5 statements.
# Partially parsed test_dumps_uses_serializer_kwargs. Retrieved 13/15 statements.


import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'BytesSerializer'
    var_5 = ()
    var_6 = 'dumps'
    var_7 = b'data'
    var_8 = lambda self, obj: var_7
    var_9 = {var_6: var_8}
    var_10 = [var_4, var_5, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = 'test'
    var_14 = var_3.dumps(var_13)

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'TextSerializer'
    var_5 = ()
    var_6 = 'dumps'
    var_7 = 'data'
    var_8 = lambda self, obj: var_7
    var_9 = {var_6: var_8}
    var_10 = [var_4, var_5, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = 'test'
    var_14 = var_3.dumps(var_13)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dumps(var_2)
    var_4 = 'different'
    var_5 = var_1.dumps(var_2, var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = 'MockJSON'
    var_6 = ()
    var_7 = 'dumps'
    var_8 = lambda self, obj, **kwargs: str(kwargs)
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = 'test'
    var_14 = var_4.dumps(var_13)
    var_15 = 'sort_keys'
    var_16 = bool('sort_keys' in var_14)
    assert var_16 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret-key'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.serializer
    var_14 = bool(var_12.serializer is var_10)
    assert var_14 is True
    var_15 = var_12.is_text_serializer
    assert var_15 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
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
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_loads_returns_any_given_serialized_input. Retrieved 1/3 statements.
# Partially parsed test_loads_accepts_serialized_payload. Retrieved 1/4 statements.
# Partially parsed test_loads_returns_none_for_empty_serialized. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)



# Parsed testcases at query #80
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None
    var_4 = '_salt_processed'
    var_5 = hasattr(var_2, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #81
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/9 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_load_payload_uses_provided_serializer. Retrieved 3/4 statements.
# Partially parsed test_load_payload_handles_text_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_handles_bytes_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_uses_overridden_serializer_is_text. Retrieved 3/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key":"value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'data'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key":"value"}'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/11 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_serializer_constructor_with_json_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_positional_bytes_serializer. Retrieved 2/9 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'ensure_ascii': False})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'salt'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_dumps_with_default_serializer. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_with_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_return_value_is_signed. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'data'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom_salt'
    var_6 = var_1.dumps(var_4, var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.
# Partially parsed test_serializer_constructor_with_signer_kwargs. Retrieved 2/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 2/6 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = b'custom_salt'
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'TextSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.is_text_serializer
    assert var_15 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'bytes'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.is_text_serializer
    assert var_15 is False



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
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

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_text. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/7 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
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
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
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
    var_16 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
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
    var_15 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_serializer_uses_provided_serializer_when_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_default_fallback_signers. Retrieved 2/4 statements.


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
    var_6 = var_1.serializer
    var_7 = module_0.is_text_serializer(var_6)
    var_8 = var_1.is_text_serializer
    var_9 = bool(var_1.is_text_serializer == var_7)
    assert var_9 is True
    var_10 = var_1.signer
    var_11 = var_1.signer_kwargs
    var_12 = bool(var_1.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_1.fallback_signers
    var_14 = bool(var_1.fallback_signers == [])
    assert var_14 is True
    var_15 = var_1.serializer_kwargs
    var_16 = bool(var_1.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.serializer
    var_14 = bool(var_12.serializer is var_10)
    assert var_14 is True
    var_15 = module_1.is_text_serializer(var_10)
    var_16 = var_12.is_text_serializer
    var_17 = bool(var_12.is_text_serializer == var_15)
    assert var_17 is True

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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_non_text_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True



# Parsed testcases at query #95
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #96
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = {}
    var_2 = {}
    var_3 = []
    var_4 = b'{"key": "value"}'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_loads_accepts_bytes. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_string. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'serialized content'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'serialized content'
    var_4 = var_2.loads(var_3)

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
    var_3 = b'42'
    var_4 = var_2.loads(var_3)
    assert var_4 == 42



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #100
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    var_4 = bool(var_2.salt is not None)
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom_salt'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 5/8 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_explicit_serializer. Retrieved 6/7 statements.
# Partially parsed test_load_payload_with_text_serializer_explicit. Retrieved 6/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 13/22 statements.


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
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'ensure_ascii': False})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = [var_4, var_5]
    var_7 = b'custom_salt'
    var_8 = 'ensure_ascii'
    var_9 = False
    var_10 = {var_8: var_9}
    var_11 = 'hmac'
    var_12 = {var_0: var_11}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

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

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.object(*var_3, **var_4)
    var_6 = var_2.dumps(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_serializer_returning_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_all_parameters. Retrieved 12/25 statements.


import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.default_serializer
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = isinstance(var_5, var_9)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = var_1.is_text_serializer
    assert var_12 is True
    var_13 = var_1.signer
    var_14 = var_1.signer_kwargs
    var_15 = bool(var_1.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_1.fallback_signers
    var_17 = bool(var_1.fallback_signers == [])
    assert var_17 is True
    var_18 = var_1.serializer_kwargs
    var_19 = bool(var_1.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'my-secret-key'

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'my-secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'k1'
    var_1 = b'k2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_6 = bool(var_1.is_text_serializer)
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = '{}'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = bool(var_14.is_text_serializer)
    assert var_17 is True

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_none_serializer_and_is_text_serializer. Retrieved 6/9 statements.
# Partially parsed test_load_payload_with_explicit_serializer_text. Retrieved 6/9 statements.
# Partially parsed test_load_payload_with_explicit_serializer_bytes. Retrieved 11/13 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 11/20 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'bytes'
    var_6 = lambda self, obj: var_5
    var_7 = lambda self, payload: payload
    var_8 = {var_3: var_6, var_4: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.Serializer(var_0, serializer=var_12)
    var_14 = var_13.load_payload(var_5)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'

import builtins as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes'
    var_2 = 'BytesSerializer'
    var_3 = ()
    var_4 = 'dumps'
    var_5 = 'loads'
    var_6 = lambda self, obj: var_1
    var_7 = lambda self, payload: payload
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = [var_2, var_3, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()

import builtins as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'FaultySerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'data'
    var_6 = lambda self, obj: var_5
    var_7 = ()
    var_8 = 'load error'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Exception(*var_9, **var_10)
    var_12 = b'data'
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_tuple. Retrieved 4/9 statements.


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
    var_0 = 'secret'
    var_1 = b'custom_salt'
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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_1 = 'none'
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
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 5/12 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 2/8 statements.
# Partially parsed test_iter_unsigners_with_multiple_secret_keys. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_2.make_signer(var_1)
    var_7 = var_4[0]
    var_8 = bool(var_4[0] == var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'test-salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8[1].secret_key
    assert var_10 == b'test-secret'
    var_11 = var_8[1].salt
    assert var_11 == b'test-salt'

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'test-salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'test-salt'

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = b'test-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'default-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'custom-salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'custom-salt'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dumps_with_text_serializer_returns_string. Retrieved 5/10 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 5/10 statements.
# Partially parsed test_dumps_with_default_serializer_returns_string. Retrieved 6/9 statements.
# Partially parsed test_dumps_with_salt. Retrieved 7/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'value2'
    var_7 = {var_2: var_6}
    var_8 = var_1.dumps(var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
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
    var_1 = b'custom-salt'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'Custom'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'secret'
    var_13 = var_11()
    var_14 = module_1.Serializer(var_12, serializer=var_13)
    var_15 = var_14.serializer
    var_16 = var_11()
    var_17 = var_15 is var_16
    var_18 = var_14.serializer
    var_19 = var_11()
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = isinstance(var_18, var_22)
    var_24 = bool(var_17 or var_23)
    assert var_24 is True
    var_25 = var_14.is_text_serializer
    assert var_25 is False

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/23 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous'
    var_4 = var_1.serializer
    var_5 = bool(var_1.serializer == var_1.default_serializer)
    assert var_5 is True
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
    var_2 = var_1.secret_key
    assert var_2 == b'secret'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'key1'
    var_5 = b'key2'
    var_6 = [var_4, var_5]
    var_7 = b'custom_salt'
    var_8 = 'sort_keys'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = 'digest_method'
    var_12 = 'sha256'
    var_13 = {var_11: var_12}



# Parsed testcases at query #16
#--------------------------




import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'CustomSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = module_1.Serializer(var_0, serializer=var_8)
    var_10 = b'test'
    var_11 = var_9.load_payload(var_10)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 7/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

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
    var_1 = b'custom_salt'
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
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer_kwargs
    var_3 = bool(var_1.signer_kwargs == {})
    assert var_3 is True

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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_2 = var_1.serializer_kwargs
    var_3 = bool(var_1.serializer_kwargs == {})
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_iter_unsigners_default_signer_yielded_first. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_fallback_tuple_signers_yielded. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_fallback_class_signers_yielded. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_fallback_with_multiple_secret_keys. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = 0
    var_7 = var_3[var_6]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners(var_1)
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'serializer_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'serializer_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_binary_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/10 statements.


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
    var_1 = b'test data'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'binary data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'anything'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'not json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_iter_unsigners_tuple_fallback. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_multiple_secret_keys. Retrieved 11/13 statements.
# Partially parsed test_iter_unsigners_with_custom_salt. Retrieved 9/10 statements.


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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback'
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

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 0
    var_2 = 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = var_5[var_9]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'override_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_text. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_overridden_default_serializer. Retrieved 1/9 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_loads_accepts_string. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = '{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test payload'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test bytes'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)
    assert var_4 == 42



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_and_invalid_payload_raises_bad_payload. Retrieved 8/17 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = ()
    var_5 = 'invalid'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.ValueError(*var_6, **var_7)
    var_9 = b'invalid'
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'gAN9cQAu'



# Parsed testcases at query #31
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/4 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 4/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b''
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dump_payload(var_2)
    var_4 = var_1.load_payload(var_3)
    assert var_4 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_iter_unsigners_predicate_line20_false. Retrieved 10/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'test_secret'
    var_1 = b'test_salt'
    var_2 = {}
    var_3 = [var_2]
    var_4 = module_0.Serializer(var_0, var_1, fallback_signers=var_3)
    var_5 = var_4.iter_unsigners(var_1)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 1
    var_9 = var_6[var_8]



# Parsed testcases at query #36
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous'
    var_6 = var_1.serializer
    var_7 = var_1.is_text_serializer
    assert var_7 is True
    var_8 = var_1.signer
    var_9 = var_1.signer_kwargs
    var_10 = bool(var_1.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_1.fallback_signers
    var_12 = bool(var_1.fallback_signers == [])
    assert var_12 is True
    var_13 = var_1.serializer_kwargs
    var_14 = bool(var_1.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'text'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer is var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'bytes'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer is var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is False

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'{}'
    var_6 = lambda self, x: var_5
    var_7 = {}
    var_8 = lambda self, x: var_7
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = [var_1, var_2, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.Serializer(var_0, serializer=var_13)
    var_15 = var_14.is_text_serializer
    assert var_15 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == var_2)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sort_keys'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, serializer_kwargs=var_2)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == var_2)
    assert var_6 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 14/22 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'none'
    var_12 = {var_10: var_11}
    var_13 = [var_12]



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dumps_returns_serialized_type. Retrieved 3/4 statements.
# Partially parsed test_dumps_handles_none. Retrieved 3/4 statements.
# Partially parsed test_dumps_handles_integer. Retrieved 3/4 statements.
# Partially parsed test_dumps_handles_list. Retrieved 6/7 statements.
# Partially parsed test_dumps_handles_dict. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)
    var_5 = var_2.dumps(var_3)
    var_6 = [var_4]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_1.type(*var_9, **var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
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
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_loads_accepts_serialized_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test payload'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'serialized data'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer_text. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #46
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 6/11 statements.
# Partially parsed test_load_payload_with_text_serializer_encoding. Retrieved 2/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.


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
    var_0 = b'my_secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my_secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

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
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_load_payload_is_text_false. Retrieved 3/6 statements.
# Partially parsed test_load_payload_is_text_false_with_explicit_serializer. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = b'"test"'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'loads'
    var_3 = 'dumps'
    var_4 = lambda self, x: x
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = {}
    var_13 = module_1.Serializer(var_11, serializer=var_10, serializer_kwargs=var_12)
    var_14 = b'test'
    var_15 = var_13.load_payload(var_14)
    assert var_15 == b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = b'"test"'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 5/15 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_serializer_constructor_default_values. Retrieved 2/3 statements.
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
    var_0 = b'bytes_secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_secret'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 3.14
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = True
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_as_text. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_as_bytes. Retrieved 1/8 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
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
    var_0 = b'secret_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_key'])
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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_key
    assert var_3 == b'key'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_dumps_with_default_serializer_returns_text. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 15/18 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 5/6 statements.
# Partially parsed test_dumps_empty_object. Retrieved 4/5 statements.
# Partially parsed test_dumps_none_value. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'BytesSerializer'
    var_5 = ()
    var_6 = 'dumps'
    var_7 = b'{"key":"value"}'
    var_8 = lambda self, obj: var_7
    var_9 = {var_6: var_8}
    var_10 = [var_4, var_5, var_9]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_3.dumps(var_15)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'data'
    var_3 = 'custom_salt'
    var_4 = var_1.dumps(var_2, var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_tuple. Retrieved 4/8 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_class. Retrieved 1/6 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/17 statements.
# Partially parsed test_serializer_constructor_with_text_serializer_detection. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer_detection. Retrieved 1/8 statements.


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
    var_6 = var_1.serializer
    var_7 = module_0.is_text_serializer(var_6)
    var_8 = var_1.is_text_serializer
    var_9 = bool(var_1.is_text_serializer == var_7)
    assert var_9 is True
    var_10 = var_1.signer
    var_11 = var_1.signer_kwargs
    var_12 = bool(var_1.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_1.fallback_signers
    var_14 = bool(var_1.fallback_signers == [])
    assert var_14 is True
    var_15 = var_1.serializer_kwargs
    var_16 = bool(var_1.serializer_kwargs == {})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'none'
    var_12 = {var_10: var_11}
    var_13 = [var_12]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #57
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_15 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
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



# Parsed testcases at query #60
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test string'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_all_params. Retrieved 12/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous'
    var_6 = var_1.serializer
    var_7 = var_1.is_text_serializer
    assert var_7 is True
    var_8 = var_1.signer
    var_9 = var_1.signer_kwargs
    var_10 = bool(var_1.signer_kwargs == {})
    assert var_10 is True
    var_11 = var_1.fallback_signers
    var_12 = bool(var_1.fallback_signers == [])
    assert var_12 is True
    var_13 = var_1.serializer_kwargs
    var_14 = bool(var_1.serializer_kwargs == {})
    assert var_14 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_serializer_keyword. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.


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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

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
    var_0 = b'bytes_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_key'])
    assert var_3 is True



# Parsed testcases at query #64
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    var_4 = bool(var_2.salt is not None)
    assert var_4 is True



# Parsed testcases at query #65
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True



# Parsed testcases at query #66
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'"hello"'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 6/7 statements.
# Partially parsed test_dumps_returns_string_when_serializer_is_text_serializer. Retrieved 5/10 statements.
# Partially parsed test_dumps_uses_salt_parameter. Retrieved 8/9 statements.
# Partially parsed test_dumps_decodes_to_utf8_when_is_text_serializer. Retrieved 5/9 statements.
# Partially parsed test_dumps_with_custom_serializer_kwargs. Retrieved 8/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = b'override_salt'
    var_7 = var_2.dumps(var_5, var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = {var_2: var_3}
    var_7 = var_1.dumps(var_6)
    var_8 = var_1.make_signer()
    var_9 = var_8.sign(var_5)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    assert var_5 == 'text'

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



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 13/14 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 6/10 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import builtins as module_0
import src.itsdangerous.serializer as module_1
import json as module_2

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'secret'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.serializer
    var_14 = bool(var_12.serializer == var_10)
    assert var_14 is True
    var_15 = {}
    var_16 = {}
    var_17 = module_2.dumps(var_15, **var_16)
    var_18 = var_12.is_text_serializer

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
    var_0 = 'CustomSigner'
    var_1 = '__init__'
    var_2 = None
    var_3 = lambda self, secret_keys, salt, **kwargs: var_2
    var_4 = {var_1: var_3}
    var_5 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_loads_returns_any_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None or var_4 is None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.loads(var_3)
    var_5 = b''
    var_6 = var_2.loads(var_5)
    var_7 = 123
    var_8 = var_2.loads(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = var_2.loads(var_12)
    var_14 = 'a'
    var_15 = {var_14: var_9}
    var_16 = var_2.loads(var_15)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.loads(var_3)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 5/12 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 2/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_and_multiple_secret_keys. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'digest_method'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'digest_method'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'default-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'custom-salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = []
    var_3 = module_0.Serializer(var_0, var_1, fallback_signers=var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_str. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/8 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_all_parameters. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'custom-salt'
    var_2 = 'indent'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = 'none'
    var_9 = {var_5: var_8}

def test_case_0():
    var_0 = 'test-secret'

def test_case_0():
    var_0 = 'test-secret'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_dumps_returns_serialized_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_load_payload_with_default_serializer. Retrieved 7/9 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 6/9 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 6/9 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 7/9 statements.
# Partially parsed test_load_payload_with_bytes_payload. Retrieved 5/7 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_1.dumps(var_4, **var_5)
    var_7 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'

import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_1.dumps(var_4, **var_5)
    var_7 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid data'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 123
    var_3 = {}
    var_4 = module_1.dumps(var_2, **var_3)
    var_5 = 'utf-8'



# Parsed testcases at query #76
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #77
#--------------------------




import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'explicit_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = module_1.want_bytes(var_1)
    var_4 = var_2.salt
    var_5 = bool(var_2.salt == var_3)
    assert var_5 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_fallback_signers. Retrieved 1/3 statements.


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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

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
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_is_false. Retrieved 14/16 statements.


import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = b'data'
    var_5 = lambda self, obj: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = module_1.Serializer(var_0, serializer=var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.dumps(var_14)



# Parsed testcases at query #80
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test payload'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #82
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = 'secret'
    var_2 = module_0.Serializer(var_1, fallback_signers=var_0)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers is not None)
    assert var_4 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = 'secret'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_all_params. Retrieved 14/30 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'none'
    var_12 = {var_10: var_11}
    var_13 = [var_12]



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_serializer_init_with_explicit_serializer. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_text. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/23 statements.


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
    var_0 = 'secret'
    var_1 = b'custom_salt'
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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = b'custom'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #87
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
    var_0 = b'bytes-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-secret'])
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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = '{}'
    var_5 = lambda self, o: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'{}'
    var_5 = lambda self, o: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer == var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is False

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer_str. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'bytes-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_load_payload_predicate_false_with_bytes_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = bool(var_1.is_text_serializer)
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = bool(var_1.is_text_serializer)
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
    var_8 = bool(var_3.is_text_serializer)
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous'
    var_7 = var_3.serializer
    var_8 = bool(var_3.is_text_serializer)
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
    var_7 = bool(var_2.is_text_serializer)
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
    var_7 = bool(var_2.is_text_serializer)
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
    var_9 = bool(var_4.is_text_serializer)
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
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = bool(var_4.is_text_serializer)
    assert var_9 is True
    var_10 = var_4.signer
    var_11 = var_4.signer_kwargs
    var_12 = bool(var_4.signer_kwargs == {})
    assert var_12 is True
    var_13 = var_4.fallback_signers
    var_14 = bool(var_4.fallback_signers == [])
    assert var_14 is True
    var_15 = var_4.serializer_kwargs
    var_16 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = bool(var_5.is_text_serializer)
    assert var_10 is True
    var_11 = var_5.signer
    var_12 = var_5.signer_kwargs
    var_13 = bool(var_5.signer_kwargs == {})
    assert var_13 is True
    var_14 = var_5.fallback_signers
    var_15 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_15 is True
    var_16 = var_5.serializer_kwargs
    var_17 = bool(var_5.serializer_kwargs == {})
    assert var_17 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_bytes. Retrieved 8/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_dumps_accepts_custom_object. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].salt
    var_6 = bool(var_3[0].salt == var_1.salt)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'custom_salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].salt
    assert var_6 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'fallback'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_0 = 'secret'
    var_1 = b'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #95
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #96
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_constructor_with_serializer_str. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_serializer_bytes. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_signer_class. Retrieved 1/4 statements.


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
    var_6 = bool(var_1.is_text_serializer)
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 25/30 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'bytes'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is False

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
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = [var_2]
    var_5 = module_0.Serializer(var_3, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [var_2])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import builtins as module_0

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'CustomSigner'
    var_14 = {}
    var_15 = 'key1'
    var_16 = b'key2'
    var_17 = [var_15, var_16]
    var_18 = 'custom_salt'
    var_19 = 'sort_keys'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = 'digest_method'
    var_23 = 'key_derivation'
    var_24 = 'none'
    var_25 = {var_23: var_24}
    var_26 = [var_25]



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_loads_with_integer_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_with_list_payload. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_payload'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.loads(var_3)
    assert var_4 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_default_serializer_override. Retrieved 1/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'

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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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
    var_1 = 'none'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_load_payload_serializer_raises_exception. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_iter_unsigners_elif_branch_with_tuple_fallback. Retrieved 7/21 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = b'test'
    var_5 = 0
    var_6 = 1



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_loads_returns_any. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_serialized_input. Retrieved 3/4 statements.
# Partially parsed test_loads_handles_none. Retrieved 4/5 statements.
# Partially parsed test_loads_handles_integer. Retrieved 4/5 statements.
# Partially parsed test_loads_handles_list. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    assert var_4 == 'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.loads(var_3)
    assert var_4 == 'HELLO'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = 'anything'
    var_5 = var_2.loads(var_4)
    assert var_5 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 2
    var_4 = 5
    var_5 = var_2.loads(var_4)
    assert var_5 == 10

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #104
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_serializer_constructor_serializer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_dumps_with_integer. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_list. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_none. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_float. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_empty_string. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_custom_object. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_string'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.dumps(var_6)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = True
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.object(*var_3, **var_4)
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 5/9 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_serializer_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/7 statements.


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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'str'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'secret'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer == var_11)
    assert var_15 is True
    var_16 = module_1.is_text_serializer(var_11)
    var_17 = var_13.is_text_serializer
    var_18 = bool(var_13.is_text_serializer == var_16)
    assert var_18 is True

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
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'itsdangerous.signer.Signer'
    var_4 = 'digest_method'
    var_5 = 'sha256'
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = [var_2, var_7]
    var_9 = 'secret'
    var_10 = module_0.Serializer(var_9, fallback_signers=var_8)
    var_11 = var_10.fallback_signers
    var_12 = bool(var_10.fallback_signers == var_8)
    assert var_12 is True



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_constructor_with_bytes_serializer. Retrieved 2/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_signer_kwargs. Retrieved 2/5 statements.
# Partially parsed test_constructor_with_fallback_signers. Retrieved 2/6 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = '{}'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'secret'
    var_14 = module_1.Serializer(var_13, serializer=var_12)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_12)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

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



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'none'}])
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #113
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



