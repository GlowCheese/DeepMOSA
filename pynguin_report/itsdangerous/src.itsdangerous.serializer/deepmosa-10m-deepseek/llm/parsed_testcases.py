####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_as_tuple. Retrieved 4/9 statements.


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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'key_derivation': 'hmac'}])
    assert var_7 is True

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_payload_uses_default_serializer_when_none. Retrieved 4/7 statements.
# Partially parsed test_load_payload_uses_provided_serializer. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/11 statements.
# Partially parsed test_load_payload_decodes_text_payload. Retrieved 2/10 statements.
# Partially parsed test_load_payload_passes_bytes_to_binary_serializer. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"test": 123}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"dummy": true}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'raw'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'data'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = b'some bytes'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'some bytes'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/11 statements.


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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
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

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'secret-key'

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
    var_0 = b'bytes-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 2/6 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 2/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = b'"test"'
    var_4 = var_2.load_payload(var_3)
    assert var_4 == 'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = {}
    var_3 = module_0._PDataSerializer(*var_1, **var_2)
    var_4 = module_0.Serializer(var_0, serializer=var_3)
    var_5 = b'"test"'
    var_6 = var_4.load_payload(var_5)
    assert var_6 == 'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'"test"'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'"test"'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = b'invalid'
    var_4 = var_2.load_payload(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_constructor_custom_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret'])
    assert var_3 is True

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous'

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
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

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
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key': 'value'})
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
    var_0 = 'key'
    var_1 = 'value'
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dumps_returns_bytes_with_default_serializer. Retrieved 6/7 statements.
# Partially parsed test_dumps_returns_string_with_text_serializer. Retrieved 5/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

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
    var_2 = 'data'
    var_3 = var_1.dumps(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'data'
    var_3 = 'custom_salt'
    var_4 = var_1.dumps(var_2, var_3)
    var_5 = 'other_salt'
    var_6 = var_1.dumps(var_2, var_5)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = 'b'
    var_6 = 'a'
    var_7 = 2
    var_8 = {var_5: var_7, var_6: var_2}
    var_9 = var_4.dumps(var_8)
    var_10 = b'"a"'
    var_11 = bool(b'"a"' in var_9)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 13/21 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'none'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_constructor_default_serializer. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
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
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'MockSerializer'
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
    var_12 = 'secret-key'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = module_1.is_text_serializer(var_11)
    var_15 = var_13.is_text_serializer
    var_16 = bool(var_13.is_text_serializer == var_14)
    assert var_16 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_all_params. Retrieved 12/13 statements.


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
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == [{'digest_method': 'sha256'}])
    assert var_7 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = 'sort_keys'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'key_derivation'
    var_6 = 'none'
    var_7 = {var_5: var_6}
    var_8 = 'digest_method'
    var_9 = 'sha512'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 13/14 statements.
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

import builtins as module_0
import src.itsdangerous.serializer as module_1
import json as module_2

def test_case_0():
    var_0 = 'secret'
    var_1 = 'CustomSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = lambda self, x: str(x)
    var_6 = lambda self, x: x
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = module_1.Serializer(var_0, serializer=var_11)
    var_13 = var_12.secret_keys
    var_14 = bool(var_12.secret_keys == [b'secret'])
    assert var_14 is True
    var_15 = var_12.salt
    assert var_15 == b'itsdangerous'
    var_16 = var_12.serializer
    var_17 = bool(var_12.serializer == var_11)
    assert var_17 is True
    var_18 = {}
    var_19 = {}
    var_20 = module_2.dumps(var_18, **var_19)
    var_21 = var_12.is_text_serializer
    var_22 = var_12.signer
    var_23 = var_12.signer_kwargs
    var_24 = bool(var_12.signer_kwargs == {})
    assert var_24 is True
    var_25 = var_12.fallback_signers
    var_26 = bool(var_12.fallback_signers == [])
    assert var_26 is True
    var_27 = var_12.serializer_kwargs
    var_28 = bool(var_12.serializer_kwargs == {})
    assert var_28 is True

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
    var_1 = 'str_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'str_salt'
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_non_text_serializer. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'test payload'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_iter_unsigners_fallback_as_tuple. Retrieved 2/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].salt
    assert var_5 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'custom-salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback-key'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'



# Parsed testcases at query #15
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
    var_3 = b'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = '{}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'{}'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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
    var_3 = '1'
    var_4 = var_2.loads(var_3)
    var_5 = '[1,2]'
    var_6 = var_2.loads(var_5)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 5/14 statements.
# Partially parsed test_load_payload_with_binary_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_serializer_override. Retrieved 4/11 statements.
# Partially parsed test_load_payload_preserves_exception. Retrieved 2/11 statements.


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
    var_1 = 'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'secret'
    var_1 = b'binary data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid payload'
    var_3 = var_1.load_payload(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'original'
    var_3 = var_1.dump_payload(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'anything'
    var_2 = 'broken'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_payload'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None or var_4 is None)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_iter_unsigners_default_signer_yielded_first. Retrieved 5/11 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 11/12 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 5/13 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 2/9 statements.
# Partially parsed test_iter_unsigners_with_multiple_secret_keys. Retrieved 3/8 statements.
# Partially parsed test_iter_unsigners_no_fallback_signers. Retrieved 8/9 statements.


def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = 0

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = [var_2]
    var_5 = module_0.Serializer(var_3, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 1
    var_10 = var_7[var_9]
    var_11 = var_7[1].key_derivation
    assert var_11 == 'none'

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'default_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'custom_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'custom_salt'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer_str. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_serializer_bytes. Retrieved 1/9 statements.
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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'separators'
    var_2 = ','
    var_3 = ':'
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = module_0.Serializer(var_0, serializer_kwargs=var_5)
    var_7 = var_6.serializer_kwargs
    var_8 = bool(var_6.serializer_kwargs == {'separators': (',', ':')})
    assert var_8 is True

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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_with_default_serializer_and_text_serializer. Retrieved 4/7 statements.
# Partially parsed test_load_payload_with_default_serializer_and_bytes_serializer. Retrieved 2/5 statements.
# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_explicit_serializer_override. Retrieved 4/7 statements.
# Partially parsed test_load_payload_with_explicit_serializer_and_text. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_explicit_serializer_and_bytes. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = 123

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

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

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

def test_case_0():
    var_0 = 'secret'
    var_1 = 42



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
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
    var_6 = bool(var_1.serializer is var_1.default_serializer)
    assert var_6 is True
    var_7 = bool(var_1.is_text_serializer)
    assert var_7 is True
    var_8 = var_1.signer
    var_9 = bool(var_1.signer is var_1.default_signer)
    assert var_9 is True
    var_10 = var_1.signer_kwargs
    var_11 = bool(var_1.signer_kwargs == {})
    assert var_11 is True
    var_12 = var_1.fallback_signers
    var_13 = bool(var_1.fallback_signers == [])
    assert var_13 is True
    var_14 = var_1.serializer_kwargs
    var_15 = bool(var_1.serializer_kwargs == {})
    assert var_15 is True

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

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_iter_unsigners_fallback_tuple. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = 1



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback_signers. Retrieved 9/22 statements.


def test_case_0():
    var_0 = b'test-secret'
    var_1 = b'test-salt'
    var_2 = 'digest_method'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}
    var_5 = 'sha512'
    var_6 = {var_2: var_5}
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------




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
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = var_2.dumps(var_3)
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = isinstance(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

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
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'a'
    var_4 = var_2.dumps(var_3)
    var_5 = 1
    var_6 = var_2.dumps(var_5)
    var_7 = [var_4]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = [var_6]
    var_11 = {}
    var_12 = module_1.type(*var_10, **var_11)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True



# Parsed testcases at query #28
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
    var_4 = 'text'
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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'



# Parsed testcases at query #29
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_false. Retrieved 17/19 statements.


import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'{}'
    var_6 = lambda self, obj, **kwargs: var_5
    var_7 = {}
    var_8 = lambda self, s: var_7
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = [var_1, var_2, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.Serializer(var_0, serializer=var_13)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_14.dumps(var_17)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/7 statements.
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dumps_returns_bytes_for_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_returns_str_for_text_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_uses_provided_salt. Retrieved 10/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'{"key":"value"}'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)
    assert var_8 == b'{"key":"value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = '{"key":"value"}'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)
    assert var_8 == '{"key":"value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'payload'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'custom_salt'
    var_9 = var_3.dumps(var_7, var_8)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback_signers. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = 'secret'



# Parsed testcases at query #34
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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_text_serializer_false. Retrieved 8/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = {}
    var_3 = module_0.Serializer(var_0, serializer=var_1, serializer_kwargs=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_3 = iter(var_2)
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2'])
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #38
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
    var_4 = 'text'
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
    var_15 = var_11()
    var_16 = var_14.serializer
    var_17 = bool(var_14.serializer is var_15)
    assert var_17 is True
    var_18 = var_14.is_text_serializer
    assert var_18 is True

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
    var_12 = 'secret'
    var_13 = var_11()
    var_14 = module_1.Serializer(var_12, serializer=var_13)
    var_15 = var_14.is_text_serializer
    assert var_15 is False

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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_fallback_signers_tuple. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_fallback_signers_class. Retrieved 1/5 statements.


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

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

def test_case_0():
    var_0 = 'secret'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #41
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 6/10 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 30/34 statements.


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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'dumped'
    var_5 = lambda self, x: var_4
    var_6 = 'loaded'
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
    var_17 = module_1.is_text_serializer(var_12)
    var_18 = var_14.is_text_serializer
    var_19 = bool(var_14.is_text_serializer == var_17)
    assert var_19 is True

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
    var_1 = '__init__'
    var_2 = None
    var_3 = lambda self, keys, salt, **kwargs: var_2
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'dumped'
    var_5 = lambda self, x: var_4
    var_6 = 'loaded'
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'CustomSigner'
    var_14 = '__init__'
    var_15 = None
    var_16 = lambda self, keys, salt, **kwargs: var_15
    var_17 = {var_14: var_16}
    var_18 = 'key_derivation'
    var_19 = 'none'
    var_20 = {var_18: var_19}
    var_21 = [var_20]
    var_22 = 'key1'
    var_23 = b'key2'
    var_24 = [var_22, var_23]
    var_25 = b'custom_salt'
    var_26 = 'sort_keys'
    var_27 = True
    var_28 = {var_26: var_27}
    var_29 = 'hmac'
    var_30 = {var_18: var_29}
    var_31 = module_1.is_text_serializer(var_12)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/3 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_payload_is_text_false_when_serializer_is_bytes_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/5 statements.


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
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
    var_7 = var_1.default_serializer
    var_8 = module_0.is_text_serializer(var_7)
    var_9 = var_1.is_text_serializer
    var_10 = bool(var_1.is_text_serializer == var_8)
    assert var_10 is True
    var_11 = var_1.signer
    var_12 = bool(var_1.signer == var_1.default_signer)
    assert var_12 is True
    var_13 = var_1.signer_kwargs
    var_14 = bool(var_1.signer_kwargs == {})
    assert var_14 is True
    var_15 = var_1.fallback_signers
    var_16 = bool(var_1.fallback_signers == [])
    assert var_16 is True
    var_17 = var_1.serializer_kwargs
    var_18 = bool(var_1.serializer_kwargs == {})
    assert var_18 is True

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
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
    var_7 = var_1.default_serializer
    var_8 = module_0.is_text_serializer(var_7)
    var_9 = var_1.is_text_serializer
    var_10 = bool(var_1.is_text_serializer == var_8)
    assert var_10 is True
    var_11 = var_1.signer
    var_12 = bool(var_1.signer == var_1.default_signer)
    assert var_12 is True
    var_13 = var_1.signer_kwargs
    var_14 = bool(var_1.signer_kwargs == {})
    assert var_14 is True
    var_15 = var_1.fallback_signers
    var_16 = bool(var_1.fallback_signers == [])
    assert var_16 is True
    var_17 = var_1.serializer_kwargs
    var_18 = bool(var_1.serializer_kwargs == {})
    assert var_18 is True

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
    var_8 = bool(var_3.serializer == var_3.default_serializer)
    assert var_8 is True
    var_9 = var_3.default_serializer
    var_10 = module_0.is_text_serializer(var_9)
    var_11 = var_3.is_text_serializer
    var_12 = bool(var_3.is_text_serializer == var_10)
    assert var_12 is True
    var_13 = var_3.signer
    var_14 = bool(var_3.signer == var_3.default_signer)
    assert var_14 is True
    var_15 = var_3.signer_kwargs
    var_16 = bool(var_3.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_3.fallback_signers
    var_18 = bool(var_3.fallback_signers == [])
    assert var_18 is True
    var_19 = var_3.serializer_kwargs
    var_20 = bool(var_3.serializer_kwargs == {})
    assert var_20 is True

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
    var_8 = bool(var_3.serializer == var_3.default_serializer)
    assert var_8 is True
    var_9 = var_3.default_serializer
    var_10 = module_0.is_text_serializer(var_9)
    var_11 = var_3.is_text_serializer
    var_12 = bool(var_3.is_text_serializer == var_10)
    assert var_12 is True
    var_13 = var_3.signer
    var_14 = bool(var_3.signer == var_3.default_signer)
    assert var_14 is True
    var_15 = var_3.signer_kwargs
    var_16 = bool(var_3.signer_kwargs == {})
    assert var_16 is True
    var_17 = var_3.fallback_signers
    var_18 = bool(var_3.fallback_signers == [])
    assert var_18 is True
    var_19 = var_3.serializer_kwargs
    var_20 = bool(var_3.serializer_kwargs == {})
    assert var_20 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = bool(var_2.serializer == var_2.default_serializer)
    assert var_7 is True
    var_8 = var_2.default_serializer
    var_9 = module_0.is_text_serializer(var_8)
    var_10 = var_2.is_text_serializer
    var_11 = bool(var_2.is_text_serializer == var_9)
    assert var_11 is True
    var_12 = var_2.signer
    var_13 = bool(var_2.signer == var_2.default_signer)
    assert var_13 is True
    var_14 = var_2.signer_kwargs
    var_15 = bool(var_2.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_2.fallback_signers
    var_17 = bool(var_2.fallback_signers == [])
    assert var_17 is True
    var_18 = var_2.serializer_kwargs
    var_19 = bool(var_2.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.serializer
    var_7 = bool(var_2.serializer == var_2.default_serializer)
    assert var_7 is True
    var_8 = var_2.default_serializer
    var_9 = module_0.is_text_serializer(var_8)
    var_10 = var_2.is_text_serializer
    var_11 = bool(var_2.is_text_serializer == var_9)
    assert var_11 is True
    var_12 = var_2.signer
    var_13 = bool(var_2.signer == var_2.default_signer)
    assert var_13 is True
    var_14 = var_2.signer_kwargs
    var_15 = bool(var_2.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_2.fallback_signers
    var_17 = bool(var_2.fallback_signers == [])
    assert var_17 is True
    var_18 = var_2.serializer_kwargs
    var_19 = bool(var_2.serializer_kwargs == {})
    assert var_19 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer
    var_7 = bool(var_2.serializer == var_2.default_serializer)
    assert var_7 is True
    var_8 = var_2.default_serializer
    var_9 = module_0.is_text_serializer(var_8)
    var_10 = var_2.is_text_serializer
    var_11 = bool(var_2.is_text_serializer == var_9)
    assert var_11 is True
    var_12 = var_2.signer
    var_13 = bool(var_2.signer == var_2.default_signer)
    assert var_13 is True
    var_14 = var_2.signer_kwargs
    var_15 = bool(var_2.signer_kwargs == {})
    assert var_15 is True
    var_16 = var_2.fallback_signers
    var_17 = bool(var_2.fallback_signers == [])
    assert var_17 is True
    var_18 = var_2.serializer_kwargs
    var_19 = bool(var_2.serializer_kwargs == {})
    assert var_19 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

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
    var_11 = 'my-secret-key'
    var_12 = module_1.Serializer(var_11, serializer=var_10)
    var_13 = var_12.secret_keys
    var_14 = bool(var_12.secret_keys == [b'my-secret-key'])
    assert var_14 is True
    var_15 = var_12.salt
    assert var_15 == b'itsdangerous'
    var_16 = var_12.serializer
    var_17 = bool(var_12.serializer == var_10)
    assert var_17 is True
    var_18 = module_1.is_text_serializer(var_10)
    var_19 = var_12.is_text_serializer
    var_20 = bool(var_12.is_text_serializer == var_18)
    assert var_20 is True
    var_21 = var_12.signer
    var_22 = bool(var_12.signer == var_12.default_signer)
    assert var_22 is True
    var_23 = var_12.signer_kwargs
    var_24 = bool(var_12.signer_kwargs == {})
    assert var_24 is True
    var_25 = var_12.fallback_signers
    var_26 = bool(var_12.fallback_signers == [])
    assert var_26 is True
    var_27 = var_12.serializer_kwargs
    var_28 = bool(var_12.serializer_kwargs == {})
    assert var_28 is True

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'my-secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = bool(var_4.serializer == var_4.default_serializer)
    assert var_9 is True
    var_10 = var_4.default_serializer
    var_11 = module_0.is_text_serializer(var_10)
    var_12 = var_4.is_text_serializer
    var_13 = bool(var_4.is_text_serializer == var_11)
    assert var_13 is True
    var_14 = var_4.signer
    var_15 = bool(var_4.signer == var_4.default_signer)
    assert var_15 is True
    var_16 = var_4.signer_kwargs
    var_17 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_17 is True
    var_18 = var_4.fallback_signers
    var_19 = bool(var_4.fallback_signers == [])
    assert var_19 is True
    var_20 = var_4.serializer_kwargs
    var_21 = bool(var_4.serializer_kwargs == {})
    assert var_21 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'my-secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'my-secret-key'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'itsdangerous'
    var_9 = var_5.serializer
    var_10 = bool(var_5.serializer == var_5.default_serializer)
    assert var_10 is True
    var_11 = var_5.default_serializer
    var_12 = module_0.is_text_serializer(var_11)
    var_13 = var_5.is_text_serializer
    var_14 = bool(var_5.is_text_serializer == var_12)
    assert var_14 is True
    var_15 = var_5.signer
    var_16 = bool(var_5.signer == var_5.default_signer)
    assert var_16 is True
    var_17 = var_5.signer_kwargs
    var_18 = bool(var_5.signer_kwargs == {})
    assert var_18 is True
    var_19 = var_5.fallback_signers
    var_20 = bool(var_5.fallback_signers == var_3)
    assert var_20 is True
    var_21 = var_5.serializer_kwargs
    var_22 = bool(var_5.serializer_kwargs == {})
    assert var_22 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'my-secret-key'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous'
    var_8 = var_4.serializer
    var_9 = bool(var_4.serializer == var_4.default_serializer)
    assert var_9 is True
    var_10 = var_4.default_serializer
    var_11 = module_0.is_text_serializer(var_10)
    var_12 = var_4.is_text_serializer
    var_13 = bool(var_4.is_text_serializer == var_11)
    assert var_13 is True
    var_14 = var_4.signer
    var_15 = bool(var_4.signer == var_4.default_signer)
    assert var_15 is True
    var_16 = var_4.signer_kwargs
    var_17 = bool(var_4.signer_kwargs == {})
    assert var_17 is True
    var_18 = var_4.fallback_signers
    var_19 = bool(var_4.fallback_signers == [])
    assert var_19 is True
    var_20 = var_4.serializer_kwargs
    var_21 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_21 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/22 statements.


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
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
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



# Parsed testcases at query #47
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
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.type(*var_7, **var_8)
    var_10 = isinstance(var_6, var_9)
    var_11 = bool(var_10)
    assert var_11 is True

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
    var_3 = 1
    var_4 = 'a'
    var_5 = (var_3, var_4)
    var_6 = var_2.dumps(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

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
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers == [])
    assert var_3 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_loads_returns_any. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)
    assert var_4 == 'test'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 13/24 statements.


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

def test_case_0():
    var_0 = 'secret-key'

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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'none'
    var_9 = {var_7: var_8}
    var_10 = 'hmac'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/2 statements.
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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_constructor_with_serializer_bytes. Retrieved 1/2 statements.
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

def test_case_0():
    var_0 = 'secret'

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



# Parsed testcases at query #54
#--------------------------




import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)
    var_5 = var_2.dumps(var_3)
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = isinstance(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_constructor_with_serializer_bytes. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_fallback_signers_tuple. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_fallback_signers_class. Retrieved 1/5 statements.
# Partially parsed test_constructor_with_serializer_as_positional. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_serializer_as_keyword. Retrieved 1/2 statements.


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
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True

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
    var_0 = b'secret_bytes'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_bytes'])
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
    var_1 = {}
    var_2 = module_0.Serializer(var_0, signer_kwargs=var_1)
    var_3 = var_2.signer_kwargs
    var_4 = bool(var_2.signer_kwargs == {})
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = module_0.Serializer(var_0, serializer_kwargs=var_1)
    var_3 = var_2.serializer_kwargs
    var_4 = bool(var_2.serializer_kwargs == {})
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'salt'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 10/12 statements.


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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'string_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'string_salt'

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
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'ensure_ascii': False})
    assert var_6 is True

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_serializer_constructor_with_json_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer_kwargs. Retrieved 2/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_dict. Retrieved 2/6 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_tuple. Retrieved 2/7 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_class. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_multiple_parameters. Retrieved 11/15 statements.
# Partially parsed test_serializer_constructor_is_text_serializer_false. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'digest_method'

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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = bool(var_1.is_text_serializer)
    assert var_2 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer_returns_correct_data. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer_returns_correct_data. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_override_serializer_uses_override. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_bytes_serializer_does_not_decode. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_serializer_that_returns_text_uses_decode. Retrieved 2/11 statements.


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
    var_1 = 42

def test_case_0():
    var_0 = 'secret'
    var_1 = 100

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some bytes'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid data'
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
    var_1 = b'binary data'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_false. Retrieved 8/11 statements.


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



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_with_fallback_and_multiple_keys. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].salt
    assert var_5 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'custom-salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback-secret'
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
    var_2 = 'fallback-secret'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

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

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = var_3[0].secret_keys
    var_5 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners(var_1)
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 is None



# Parsed testcases at query #62
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_dumps_returns_expected_serialized_type. Retrieved 4/5 statements.
# Partially parsed test_dumps_accepts_any_object. Retrieved 4/5 statements.
# Partially parsed test_dumps_returns_serialized_type. Retrieved 4/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = 'test'
    var_5 = var_2.dumps(var_4)
    assert var_5 == b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = 123
    var_5 = var_2.dumps(var_4)
    assert var_5 == b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = None
    var_5 = var_2.dumps(var_4)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_serializer_constructor_with_explicit_serializer. Retrieved 1/2 statements.
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



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'salt'
    var_5 = var_1.serializer
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
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



# Parsed testcases at query #66
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'salt'



# Parsed testcases at query #67
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



# Parsed testcases at query #68
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_returns_false_for_is_text. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.
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

def test_case_0():
    var_0 = 'secret'

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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 2/10 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/9 statements.
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
    var_1 = b'salt'

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



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 2/9 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/3 statements.
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
    var_1 = b'salt'

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



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_iter_unsigners_fallback_tuple. Retrieved 6/16 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = b'test'
    var_5 = 1



# Parsed testcases at query #74
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1, serializer_kwargs=var_1, signer=var_1, signer_kwargs=var_1, fallback_signers=var_1)
    var_3 = b'\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00}\x94.'
    var_4 = var_2.load_payload(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 13/22 statements.


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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'none'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_loads_accepts_serialized_type. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_complex_type. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None or var_4 is None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)

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
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.loads(var_6)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 5/13 statements.
# Partially parsed test_load_payload_with_explicit_serializer_override. Retrieved 3/5 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 5/14 statements.
# Partially parsed test_load_payload_uses_is_text_serializer_flag. Retrieved 3/12 statements.


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
    var_7 = var_1.load_payload(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'data'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"custom": "payload"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'text'
    var_2 = 'hello'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 6/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'test'
    var_5 = var_3.dumps(var_4)



# Parsed testcases at query #80
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #81
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_object'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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
    var_0 = 'key'

def test_case_0():
    var_0 = 'key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
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
    var_4 = 'key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

def test_case_0():
    var_0 = 'key'

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
    var_0 = b'bytes-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 2/8 statements.


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
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None

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

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'

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

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
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
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
    var_7 = var_1.is_text_serializer
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
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'digest_method': 'sha256'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_secret'



# Parsed testcases at query #85
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
    var_0 = 'my_secret_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my_secret_key'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my_secret_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my_secret_key'])
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



# Parsed testcases at query #86
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.is_text_serializer
    assert var_5 is True
    var_6 = var_1.signer
    var_7 = var_1.signer_kwargs
    var_8 = bool(var_1.signer_kwargs == {})
    assert var_8 is True
    var_9 = var_1.fallback_signers
    var_10 = bool(var_1.fallback_signers == [])
    assert var_10 is True
    var_11 = var_1.serializer_kwargs
    var_12 = bool(var_1.serializer_kwargs == {})
    assert var_12 is True

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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

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



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_dumps_returns_string_when_serialized_type_is_string. Retrieved 3/4 statements.
# Partially parsed test_dumps_returns_bytes_when_serialized_type_is_bytes. Retrieved 3/4 statements.
# Partially parsed test_dumps_accepts_custom_object. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_data'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)

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
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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



# Parsed testcases at query #89
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



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 12/20 statements.


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

def test_case_0():
    var_0 = 'secret-key'

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

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'sort_keys'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'digest_method'
    var_6 = 'sha256'
    var_7 = {var_5: var_6}
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #91
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_serializer_init_with_explicit_serializer. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #93
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_load_payload_with_default_serializer_and_text_payload. Retrieved 4/6 statements.
# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 3/12 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 6/15 statements.
# Partially parsed test_load_payload_with_serializer_override. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_serializer_override_raises_bad_payload. Retrieved 4/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = '{"key": "value"}'
    var_3 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = "{'key': 'value'}"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = repr(var_3)
    var_5 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b''
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some data'
    var_3 = var_1.load_payload(var_2, var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Could not load the payload'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 6/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'data'
    var_5 = var_3.dumps(var_4)



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_constructor_custom_text_serializer. Retrieved 3/6 statements.
# Partially parsed test_constructor_custom_bytes_serializer. Retrieved 3/6 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer

def test_case_0():
    var_0 = lambda self, obj: 'text'
    var_1 = lambda self, s: {}
    var_2 = 'secret'

def test_case_0():
    var_0 = lambda self, obj: b'bytes'
    var_1 = lambda self, s: {}
    var_2 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'mysecret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'mysecret'])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'oldkey'
    var_1 = 'newkey'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'oldkey', b'newkey'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'newkey'

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
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #97
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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

def test_case_0():
    var_0 = 'secret-key'

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



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_iter_unsigners_includes_fallback_tuple_signers. Retrieved 2/10 statements.
# Partially parsed test_iter_unsigners_includes_fallback_class_signers. Retrieved 1/8 statements.
# Partially parsed test_iter_unsigners_uses_fallback_signer_class_with_default_kwargs. Retrieved 4/10 statements.


import src.itsdangerous.signer as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = module_1.Serializer(var_0)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = var_2.make_signer()
    var_6 = var_4[0].secret_key
    var_7 = bool(var_4[0].secret_key == var_5.secret_key)
    assert var_7 is True

import src.itsdangerous.signer as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'custom-salt'
    var_3 = module_1.Serializer(var_0, var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key'
    var_2 = b'fallback-key'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    var_9 = bool(var_8 > 1)
    assert var_9 is True

def test_case_0():
    var_0 = b'secret-key'
    var_1 = {}

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_2, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 3

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = 'hmac'
    var_5 = {var_1: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_0, signer_kwargs=var_3, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = var_9[1].key_derivation
    assert var_10 == 'hmac'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'test-salt'
    var_2 = {}
    var_3 = [var_2]
    var_4 = module_0.Serializer(var_0, var_1, fallback_signers=var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}



# Parsed testcases at query #100
#--------------------------

# Partially parsed test__pdata_serializer_loads_accepts_bytes. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test_payload'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'string_payload'
    var_4 = var_2.loads(var_3)
    assert var_4 == 'string_payload'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 5/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = 'test_data'
    var_4 = var_2.dumps(var_3)



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_iter_unsigners_fallback_signer_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_fallback_signer_class. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_fallback_tuple_with_multiple_secret_keys. Retrieved 6/12 statements.
# Partially parsed test_iter_unsigners_fallback_class_with_multiple_secret_keys. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].secret_key
    assert var_5 == b'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'override'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'override'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = 'fallback'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = 'fallback'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'instance'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = None
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'instance'



# Parsed testcases at query #103
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



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serializer_constructor_serializer_json. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_serializer_custom_bytes. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_serializer_custom_str. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_signer_custom. Retrieved 1/4 statements.


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
    var_6 = bool(not var_1.is_text_serializer)
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
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

def test_case_0():
    var_0 = 'secret'

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



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/3 statements.
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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_false_when_serializer_is_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_serializer_constructor_default_values. Retrieved 3/5 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
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
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
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
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
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



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


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
    var_1 = b'custom-salt'
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

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/7 statements.


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
    var_4 = 'data'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'secret-key'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer is var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'secret-key'
    var_13 = module_1.Serializer(var_12, serializer=var_11)
    var_14 = var_13.serializer
    var_15 = bool(var_13.serializer is var_11)
    assert var_15 is True
    var_16 = var_13.is_text_serializer
    assert var_16 is False

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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_fallback_signers. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 12/22 statements.


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

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'secret'

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
    var_1 = 'salt'
    var_2 = 'sort_keys'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'digest_method'
    var_6 = 'sha256'
    var_7 = {var_5: var_6}
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = b'custom'
    var_5 = module_0.Serializer(var_3, var_4)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'custom'

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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.default_serializer

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.default_signer

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.default_fallback_signers
    var_3 = bool(var_1.default_fallback_signers == [])
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_serializer_returning_bytes. Retrieved 1/8 statements.


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
    var_0 = 'secret'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/4 statements.


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
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
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
    var_0 = 'secret-key'
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
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 5/11 statements.
# Partially parsed test_iter_unsigners_with_signer_class_fallback. Retrieved 2/7 statements.
# Partially parsed test_iter_unsigners_yields_for_each_secret_key_with_fallback. Retrieved 4/9 statements.
# Partially parsed test_iter_unsigners_fallback_with_dict_uses_default_signer. Retrieved 8/11 statements.


import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 >= 1)
    assert var_6 is True
    var_7 = module_1.want_bytes(var_1)
    var_8 = var_4[0].salt
    var_9 = bool(var_4[0].salt == var_7)
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'key'
    var_3 = 'fallback_secret'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    var_10 = bool(var_9 >= 2)
    assert var_10 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'key'
    var_3 = 'fallback_secret'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'

import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'default_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'custom_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = module_1.want_bytes(var_3)
    var_7 = var_5[0].salt
    var_8 = bool(var_5[0].salt == var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = []
    var_3 = module_0.Serializer(var_0, var_1, fallback_signers=var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)
    var_7 = module_1.want_bytes(var_0)
    var_8 = module_1.want_bytes(var_1)
    var_9 = [var_7, var_8]
    var_10 = var_6[0].secret_keys
    var_11 = bool(var_6[0].secret_keys == var_9)
    assert var_11 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'key'
    var_3 = 'fallback_secret'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.want_bytes(var_3)
    var_7 = [var_6]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_override_serializer_parameter. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_text_serializer_and_unicode_payload. Retrieved 3/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test_payload'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test_bytes'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'unicode'
    var_2 = 'utf-8'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_iter_unsigners_predicate_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = b'test_key'
    var_1 = b'test_salt'
    var_2 = 'digest_method'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}
    var_5 = bool(var_0)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signer_class. Retrieved 2/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = {}
    var_2 = 'algorithm'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = module_0.Serializer(var_0, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    var_10 = bool(var_9 == 1 + 1 + 1)
    assert var_10 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].secret_keys
    var_6 = bool(var_3[0].secret_keys == [b'secret-key'])
    assert var_6 is True

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
    var_9 = var_7[1].key_derivation
    assert var_9 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 1

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'custom-salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)
    var_5 = var_4[0].salt
    assert var_5 == b'custom-salt'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'default-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = None
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = var_5[0].salt
    assert var_6 == b'default-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/7 statements.
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_text_serializer_raises_bad_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_iter_unsigners_with_default_signer_and_no_fallback. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_custom_salt. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 14/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 7/17 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 4/13 statements.
# Partially parsed test_iter_unsigners_with_multiple_secret_keys. Retrieved 10/11 statements.
# Partially parsed test_iter_unsigners_with_fallback_and_multiple_keys. Retrieved 7/18 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.iter_unsigners(var_1)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = b'custom_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback_secret'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = None
    var_7 = var_5.iter_unsigners(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = 1
    var_13 = var_8[var_12]

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback_secret'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 0
    var_6 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = 0
    var_3 = 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = None
    var_5 = var_3.iter_unsigners(var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = 0
    var_5 = 1
    var_6 = 2



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 7/11 statements.


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
    var_4 = 'dumped'
    var_5 = lambda self, x: var_4
    var_6 = 'loaded'
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
    var_0 = 'BinarySerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'dumped'
    var_5 = lambda self, x: var_4
    var_6 = b'loaded'
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
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_init_with_explicit_serializer_passes_through_to_attribute. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_iter_unsigners_fallback_is_tuple. Retrieved 4/16 statements.


def test_case_0():
    var_0 = b'test-secret'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 9/16 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.dumps(var_6, **var_7)
    var_9 = 'utf-8'



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fallback_signers_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer_not_none. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer_str. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/8 statements.
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_is_text_serializer_true. Retrieved 1/8 statements.
# Partially parsed test_constructor_is_text_serializer_false. Retrieved 1/8 statements.
# Partially parsed test_constructor_default_fallback_signers. Retrieved 2/4 statements.


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
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = b'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True

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

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_init_with_serializer_evaluates_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dumps_returns_bytes_with_default_json_serializer. Retrieved 6/7 statements.
# Partially parsed test_dumps_returns_string_with_text_serializer. Retrieved 5/13 statements.
# Partially parsed test_dumps_uses_serializer_kwargs. Retrieved 6/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)

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
    var_2 = 'data'
    var_3 = var_1.dumps(var_2)
    var_4 = b'.'
    var_5 = bool(b'.' in var_3)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'default_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)
    var_5 = b'custom_salt'
    var_6 = var_2.dumps(var_3, var_5)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_6 = var_3.secret_key
    assert var_6 == b'key2'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_fallback_dict. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_fallback_tuple. Retrieved 8/20 statements.
# Partially parsed test_iter_unsigners_fallback_signer_class. Retrieved 5/16 statements.


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
    var_1 = 'key'
    var_2 = 'fallback_secret'
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
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'
    var_3 = 'key'
    var_4 = 'fallback_secret'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 1

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = 'secret-key'
    var_3 = 0
    var_4 = 1

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'fallback_secret'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_9[0].secret_keys
    var_12 = bool(var_9[0].secret_keys == [b'old-secret', b'new-secret'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == [b'old-secret'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == [b'new-secret'])
    assert var_16 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'default-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'explicit-salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].salt
    assert var_7 == b'explicit-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback_secret'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = 'explicit-salt'
    var_7 = var_5.iter_unsigners(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8[0].salt
    assert var_10 == b'explicit-salt'
    var_11 = var_8[1].salt
    assert var_11 == b'explicit-salt'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 12/14 statements.


import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'BytesSerializer'
    var_5 = ()
    var_6 = 'loads'
    var_7 = lambda self, x: x
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = {}
    var_11 = module_1.type(*var_9, **var_10)
    var_12 = b'test'
    var_13 = var_3.load_payload(var_12)
    assert var_13 == b'test'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dumps_returns_serialized_type_for_complex_object. Retrieved 1/5 statements.


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
    var_3 = True
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'null'



# Parsed testcases at query #33
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_object'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #35
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'data'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or var_4 is not None)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 123
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or var_4 is not None)
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
    var_8 = bool(var_7 is None or var_7 is not None)
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
    var_7 = bool(var_6 is None or var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 11/21 statements.


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

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old_key'
    var_2 = b'new_key'
    var_3 = [var_1, var_2]
    var_4 = b'custom_salt'
    var_5 = 'ensure_ascii'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_returns_bytes. Retrieved 1/8 statements.


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

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 3/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes_salt'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)

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



# Parsed testcases at query #39
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    var_4 = bool(var_2.salt is not None)
    assert var_4 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serializer_constructor_serializer_text. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_signer. Retrieved 1/4 statements.


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
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_payload_with_text_serializer_and_valid_payload_returns_deserialized_data. Retrieved 6/9 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = 'utf-8'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/3 statements.
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_integer_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_list_serializer. Retrieved 5/8 statements.
# Partially parsed test_dumps_with_none_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_dict_serializer. Retrieved 4/7 statements.
# Partially parsed test_dumps_with_float_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_bool_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 2/4 statements.
# Partially parsed test_dumps_with_tuple_serializer. Retrieved 4/7 statements.
# Partially parsed test_dumps_with_set_serializer. Retrieved 5/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_string'
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    assert var_2 == 'test_string'

import json as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    assert var_2 == 42

import json as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import json as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    assert var_2 is None

import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = bool(var_4 == {'key': 1})
    assert var_5 is True

import json as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    var_3 = bool(var_2 == 3.14)
    assert var_3 is True

import json as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    assert var_2 is True

import json as module_0

def test_case_0():
    var_0 = b'bytes_data'
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    assert var_2 == b'bytes_data'

import json as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = bool(var_4 == (1, 'test'))
    assert var_5 is True

import json as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {}
    var_5 = module_0.dumps(var_3, **var_4)
    var_6 = bool(var_5 == {1, 2, 3})
    assert var_6 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 13/14 statements.


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
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'none'})
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
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'none'
    var_9 = {var_7: var_8}
    var_10 = 'hmac'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_load_payload_with_text_serializer_decodes_bytes. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_bytes_serializer_does_not_decode. Retrieved 2/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_is_text_true_uses_decode. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_is_text_false_no_decode. Retrieved 2/10 statements.


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

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'loads'
    var_3 = 'dumps'
    var_4 = ' loaded'
    var_5 = lambda x: x + var_4
    var_6 = staticmethod(var_5)
    var_7 = lambda x: x
    var_8 = staticmethod(var_7)
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = 'secret'
    var_15 = module_1.Serializer(var_14, serializer=var_13)
    var_16 = 'test'
    var_17 = var_15.dump_payload(var_16)
    var_18 = var_15.load_payload(var_17, var_13)
    assert var_18 == 'test loaded'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'encoded'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'raw'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid'
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 123
    var_3 = var_1.dump_payload(var_2)
    var_4 = None
    var_5 = var_1.load_payload(var_3, var_4)
    assert var_5 == 123

def test_case_0():
    var_0 = 'secret'
    var_1 = b'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'world'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_fallback_signers_not_none. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #47
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/11 statements.


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



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'test-secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = var_1.is_text_serializer
    assert var_6 is True
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
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
    var_0 = 'test'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
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
    var_0 = 'test'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'key'

def test_case_0():
    var_0 = 'key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

def test_case_0():
    var_0 = 'key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
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
    var_4 = 'key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #51
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'sort_keys': True})
    assert var_6 is True

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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/21 statements.


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

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = 'custom_salt'
    var_6 = 'sort_keys'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = 'digest_method'
    var_10 = 'sha256'
    var_11 = {var_9: var_10}



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_is_false. Retrieved 8/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_dumps_signs_payload_with_signer. Retrieved 8/10 statements.
# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 15/16 statements.
# Partially parsed test_dumps_returns_string_when_text_serializer. Retrieved 15/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'BytesSerializer'
    var_3 = ()
    var_4 = 'dumps'
    var_5 = 'loads'
    var_6 = b'serialized'
    var_7 = lambda self, obj: var_6
    var_8 = lambda self, data: data
    var_9 = {var_4: var_7, var_5: var_8}
    var_10 = [var_2, var_3, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.Serializer(var_0, var_1, var_13)
    var_15 = 'data'
    var_16 = var_14.dumps(var_15)

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'TextSerializer'
    var_3 = ()
    var_4 = 'dumps'
    var_5 = 'loads'
    var_6 = 'serialized'
    var_7 = lambda self, obj: var_6
    var_8 = lambda self, data: data
    var_9 = {var_4: var_7, var_5: var_8}
    var_10 = [var_2, var_3, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.Serializer(var_0, var_1, var_13)
    var_15 = 'data'
    var_16 = var_14.dumps(var_15)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'default'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)
    var_5 = 'custom'
    var_6 = var_2.dumps(var_3, var_5)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #55
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = b'test'
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is None or True)
    assert var_5 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_serializer_keyword. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_returning_bytes. Retrieved 1/9 statements.


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
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_custom_default_serializer. Retrieved 2/12 statements.
# Partially parsed test_serializer_constructor_with_custom_default_fallback_signers. Retrieved 5/9 statements.


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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = var_1.is_text_serializer
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = module_0.Serializer(var_3)
    var_5 = var_4.fallback_signers
    var_6 = bool(var_4.fallback_signers == [{'key_derivation': 'none'}])
    assert var_6 is True



# Parsed testcases at query #58
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.loads(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_positional_serializer. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_keyword_serializer. Retrieved 1/2 statements.


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
    var_7 = var_1.signer
    var_8 = var_1.signer_kwargs
    var_9 = bool(var_1.signer_kwargs == {})
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

def test_case_0():
    var_0 = 'secret-key'

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

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 2/3 statements.


def test_case_0():
    var_0 = None
    var_1 = 42



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_signer_class. Retrieved 1/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].secret_keys
    var_7 = bool(var_4[0].secret_keys == [b'secret-key'])
    assert var_7 is True

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
    assert var_8 == 2
    var_9 = var_7[0].secret_keys
    var_10 = bool(var_7[0].secret_keys == [b'secret-key'])
    assert var_10 is True
    var_11 = var_7[1].secret_keys
    var_12 = bool(var_7[1].secret_keys == [b'secret-key'])
    assert var_12 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = module_0.Serializer(var_2, fallback_signers=var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0].secret_keys
    var_9 = bool(var_6[0].secret_keys == [b'old-secret', b'new-secret'])
    assert var_9 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = []
    var_3 = module_0.Serializer(var_0, var_1, fallback_signers=var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].salt
    assert var_7 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = []
    var_3 = module_0.Serializer(var_0, var_1, fallback_signers=var_2)
    var_4 = var_3.iter_unsigners(var_1)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].salt
    assert var_7 == b'itsdangerous'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = b'explicit-salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].salt
    assert var_7 == b'explicit-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
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
    var_11 = var_9[0].secret_keys
    var_12 = bool(var_9[0].secret_keys == [b'key1', b'key2'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == [b'key1'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == [b'key2'])
    assert var_16 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_raises_bad_payload_on_decode_error. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'\xff\xfe'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/15 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/2 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, signer=var_1)
    var_3 = var_2.signer

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
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

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_dumps_returns_bytes_for_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_returns_str_for_text_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_uses_salt. Retrieved 9/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'{"key":"value"}'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = '{"key":"value"}'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'data'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = b'override_salt'
    var_8 = var_2.dumps(var_6, var_7)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/9 statements.
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
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

def test_case_0():
    var_0 = 'secret-key'

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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #66
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
    var_0 = 'secret'
    var_1 = 'CustomSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = '{}'
    var_6 = lambda self, x: var_5
    var_7 = {}
    var_8 = lambda self, x: var_7
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = [var_1, var_2, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.Serializer(var_0, serializer=var_13)
    var_15 = var_14.serializer
    var_16 = bool(var_14.serializer is var_13)
    assert var_16 is True
    var_17 = var_14.is_text_serializer
    assert var_17 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'CustomSigner'
    var_2 = {}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == var_3)
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
    var_7 = bool(var_5.fallback_signers == var_4)
    assert var_7 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == var_3)
    assert var_6 is True



# Parsed testcases at query #67
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_data'
    var_4 = var_2.dumps(var_3)
    var_5 = var_2.loads(var_4)
    assert var_5 == 'test_data'



# Parsed testcases at query #68
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)
    var_7 = var_6.serializer
    var_8 = bool(var_6.serializer is var_4)
    assert var_8 is True



# Parsed testcases at query #69
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/11 statements.


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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 13/14 statements.


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
    var_3 = b'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'none'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_with_fallback_and_multiple_secret_keys. Retrieved 3/8 statements.
# Partially parsed test_iter_unsigners_yields_generator. Retrieved 3/5 statements.


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
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'custom_salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = module_1.want_bytes(var_2)
    var_7 = var_4[0].salt
    var_8 = bool(var_4[0].salt == var_6)
    assert var_8 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'fallback_key'
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
    var_2 = 'fallback_key'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.iter_unsigners()
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = module_1.want_bytes(var_0)
    var_8 = module_1.want_bytes(var_1)
    var_9 = [var_7, var_8]
    var_10 = var_5[0].secret_keys
    var_11 = bool(var_5[0].secret_keys == var_9)
    assert var_11 is True

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_binary_serializer. Retrieved 1/8 statements.
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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 is None
    var_6 = var_2.serializer

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer

def test_case_0():
    var_0 = 'secret-key'

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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
    assert var_6 is True

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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_loads_accepts_serialized_input. Retrieved 3/4 statements.


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
    var_3 = b'data'
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
    assert var_4 is None



# Parsed testcases at query #75
#--------------------------




import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'dumped'
    var_5 = lambda x, **kw: var_4
    var_6 = 'loaded'
    var_7 = lambda x: var_6
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



# Parsed testcases at query #76
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test_object'
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/5 statements.
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

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'digest_method': 'sha256'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
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
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = var_2.serializer
    var_4 = var_2.is_text_serializer
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, signer=var_1)
    var_3 = var_2.signer

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous'
    var_5 = var_1.serializer
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
    var_7 = {}
    var_8 = {}
    var_9 = module_1.dumps(var_7, **var_8)
    var_10 = var_1.is_text_serializer
    var_11 = var_1.signer
    var_12 = bool(var_1.signer == var_1.default_signer)
    assert var_12 is True
    var_13 = var_1.signer_kwargs
    var_14 = bool(var_1.signer_kwargs == {})
    assert var_14 is True
    var_15 = var_1.fallback_signers
    var_16 = bool(var_1.fallback_signers == [])
    assert var_16 is True
    var_17 = var_1.serializer_kwargs
    var_18 = bool(var_1.serializer_kwargs == {})
    assert var_18 is True

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



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_dumps_with_default_json_serializer. Retrieved 7/9 statements.
# Partially parsed test_dumps_with_text_serializer_returns_str. Retrieved 3/5 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 3/13 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 5/6 statements.
# Partially parsed test_dumps_with_serializer_kwargs. Retrieved 11/12 statements.
# Partially parsed test_dumps_with_bytes_secret_key. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_multiple_secret_keys. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = '.'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.dumps(var_1, **var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'data1'
    var_3 = var_1.dumps(var_2)
    var_4 = 'data2'
    var_5 = var_1.dumps(var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = 'b'
    var_6 = 'a'
    var_7 = 2
    var_8 = {var_5: var_2, var_6: var_7}
    var_9 = var_4.dumps(var_8)
    var_10 = '.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = 'test'
    var_5 = var_3.dumps(var_4)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes_salt'

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
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'key': 'value'})
    assert var_6 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
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



# Parsed testcases at query #81
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None



# Parsed testcases at query #82
#--------------------------




import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = 'CustomSerializer'
    var_4 = ()
    var_5 = 'loads'
    var_6 = lambda self, x: x
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = var_1.load_payload(var_2, var_11)
    var_13 = bool(var_12 == var_2)
    assert var_13 is True



# Parsed testcases at query #83
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.fallback_signers
    var_3 = bool(var_1.fallback_signers is not None)
    assert var_3 is True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_serializer_constructor_serializer_positional. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_serializer_keyword. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.


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

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'salt'

def test_case_0():
    var_0 = 'secret-key'

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



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #86
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 42
    var_4 = var_2.dumps(var_3)
    assert var_4 == 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.dumps(var_3)
    assert var_4 == 'hello'

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
    var_8 = bool(var_7 == [1, 2, 3])
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
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

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
    var_3 = 3.14
    var_4 = var_2.dumps(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True



# Parsed testcases at query #87
#--------------------------




import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'Dummy'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'data'
    var_6 = lambda self, obj: var_5
    var_7 = lambda self, s: s
    var_8 = {var_3: var_6, var_4: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.Serializer(var_0, serializer=var_12)
    var_14 = var_13.is_text_serializer
    assert var_14 is False



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers. Retrieved 13/15 statements.
# Partially parsed test_iter_unsigners_fallback_as_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_fallback_as_class. Retrieved 3/12 statements.


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
    var_2 = 'fallback_secret'
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
    var_7 = var_5[0].salt
    assert var_7 == b'override_salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'fallback_secret'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = module_0.Serializer(var_2, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_9[0].secret_keys
    var_12 = bool(var_9[0].secret_keys == [b'old_secret', b'new_secret'])
    assert var_12 is True
    var_13 = var_9[1].secret_keys
    var_14 = bool(var_9[1].secret_keys == [b'old_secret'])
    assert var_14 is True
    var_15 = var_9[2].secret_keys
    var_16 = bool(var_9[2].secret_keys == [b'new_secret'])
    assert var_16 is True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'tuple_fallback'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 'secret'
    var_1 = 0
    var_2 = 1



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 7/11 statements.


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
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: eval(x)
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
    var_15 = var_12.is_text_serializer
    assert var_15 is True

import builtins as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = 'data'
    var_7 = lambda self, x: {var_6: x}
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



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_keyword. Retrieved 1/7 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_text_serializer. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_signer. Retrieved 1/4 statements.


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



# Parsed testcases at query #93
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0._PDataSerializer(*var_0, **var_1)
    var_3 = 'test'
    var_4 = var_2.loads(var_3)



# Parsed testcases at query #94
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

import builtins as module_0
import src.itsdangerous.serializer as module_1

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
    var_15 = var_12.is_text_serializer
    assert var_15 is False

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
    var_8 = var_7.signer
    var_9 = bool(var_7.signer == var_5)
    assert var_9 is True

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
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

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
    var_15 = var_14.is_text_serializer
    assert var_15 is False



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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
    var_0 = b'secret-bytes'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-bytes'])
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



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 2/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_positional_serializer. Retrieved 2/10 statements.


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
    var_0 = b'secret'
    var_1 = b'salt'

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
    var_1 = b'salt'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/15 statements.


import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = module_1.want_bytes(var_0)
    var_3 = [var_2]
    var_4 = var_1.secret_keys
    var_5 = bool(var_1.secret_keys == var_3)
    assert var_5 is True
    var_6 = b'itsdangerous'
    var_7 = module_1.want_bytes(var_6)
    var_8 = var_1.salt
    var_9 = bool(var_1.salt == var_7)
    assert var_9 is True
    var_10 = var_1.serializer
    var_11 = bool(var_1.is_text_serializer)
    assert var_11 is True
    var_12 = var_1.signer
    var_13 = var_1.signer_kwargs
    var_14 = bool(var_1.signer_kwargs == {})
    assert var_14 is True
    var_15 = var_1.fallback_signers
    var_16 = bool(var_1.fallback_signers == [])
    assert var_16 is True
    var_17 = var_1.serializer_kwargs
    var_18 = bool(var_1.serializer_kwargs == {})
    assert var_18 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [var_0])
    assert var_3 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = [want_bytes(s) for s in var_2]
    var_5 = var_3.secret_keys
    var_6 = bool(var_3.secret_keys == var_4)
    assert var_6 is True

import src.itsdangerous.serializer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = module_1.want_bytes(var_1)
    var_4 = var_2.salt
    var_5 = bool(var_2.salt == var_3)
    assert var_5 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == var_3)
    assert var_6 is True

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == var_3)
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_4)
    assert var_7 is True

import src.itsdangerous.encoding as module_0

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
    var_10 = [var_9]
    var_11 = module_0.want_bytes(var_0)
    var_12 = [var_11]
    var_13 = module_0.want_bytes(var_1)



# Parsed testcases at query #98
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



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_dumps_with_text_serializer_returns_string. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 5/11 statements.


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
    var_5 = 'salt1'
    var_6 = var_1.dumps(var_4, var_5)
    var_7 = {var_2: var_3}
    var_8 = 'salt2'
    var_9 = var_1.dumps(var_7, var_8)
    var_10 = bool(var_6 != var_9)
    assert var_10 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = var_1.dumps(var_8)
    var_10 = bool(var_5 != var_9)
    assert var_10 is True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_serializer_constructor_with_json_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
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
    var_0 = b'secret_bytes'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_bytes'])
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
    var_1 = b'bytes_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes_salt'

def test_case_0():
    var_0 = 'secret'

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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 12/20 statements.


import src.itsdangerous.serializer as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'FakeSerializer'
    var_5 = ()
    var_6 = 'loads'
    var_7 = ()
    var_8 = 'test'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Exception(*var_9, **var_10)
    var_12 = b'test'
    var_13 = var_3.load_payload(var_12)



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_iter_unsigners_predicate_line_15_true. Retrieved 6/17 statements.


def test_case_0():
    var_0 = b'test_secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_constructor_serializer_is_bytes. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_default_fallback_signers. Retrieved 2/4 statements.


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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.is_text_serializer
    assert var_2 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'test-secret'])
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
    var_0 = b'test-secret-bytes'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'test-secret-bytes'])
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
    var_0 = 'test-secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 is None

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 4})
    assert var_6 is True

def test_case_0():
    var_0 = 'test-secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)
    var_5 = var_4.signer_kwargs
    var_6 = bool(var_4.signer_kwargs == {'key_derivation': 'hmac'})
    assert var_6 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
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
    var_0 = 'test-secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #105
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



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_fallback_signers_not_none_initialization. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #107
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.fallback_signers
    var_4 = bool(var_2.fallback_signers == [])
    assert var_4 is True



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_serializer_returns_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/27 statements.


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
    var_6 = bool(var_1.serializer == var_1.default_serializer)
    assert var_6 is True
    var_7 = bool(var_1.is_text_serializer)
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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = var_4.serializer_kwargs
    var_6 = bool(var_4.serializer_kwargs == {'indent': 2})
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

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = b'custom'
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



# Parsed testcases at query #109
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
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)
    var_6 = var_5.fallback_signers
    var_7 = bool(var_5.fallback_signers == var_3)
    assert var_7 is True

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



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 8/21 statements.


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
    var_3 = b'mysalt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'



