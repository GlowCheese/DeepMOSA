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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #2
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
    var_4 = module_0.dumps(var_3)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'invalid json'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_iter_unsigners_default. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 2/10 statements.
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 2/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'digest_method'
    var_1 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/11 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = module_0.loads(var_4)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = module_0.loads(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = var_2.loads(var_6, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'salt2'
    var_7 = var_2.dumps(var_5, var_6)
    var_8 = var_2.loads(var_7, var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key1'
    var_1 = 'secret-key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.dumps(var_6)
    var_8 = var_3.loads(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'digest_method'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = [var_4]
    var_7 = module_0.Serializer(var_5, fallback_signers=var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = var_7.dumps(var_10)
    var_12 = var_7.loads(var_11)

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
    var_9 = var_4.loads(var_8)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    var_4 = var_1.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)
    var_4 = var_1.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'list'
    var_4 = 'nested'
    var_5 = 'value'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_11}
    var_13 = {var_2: var_5, var_3: var_9, var_4: var_12}
    var_14 = var_1.dumps(var_13)
    var_15 = var_1.loads(var_14)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = b'secret'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x)
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: var_2
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

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
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = '!'
    var_4 = {var_0: var_3}
    var_5 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x).encode()
    var_3 = lambda x: int(x.decode())
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serializer_with_str_secret_key. Retrieved 2/3 statements.
# Partially parsed test_serializer_with_bytes_secret_key. Retrieved 2/3 statements.
# Partially parsed test_serializer_with_list_secret_keys. Retrieved 4/5 statements.
# Partially parsed test_serializer_with_none_salt. Retrieved 2/3 statements.
# Partially parsed test_serializer_with_fallback_signers. Retrieved 5/10 statements.
# Partially parsed test_serializer_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'salt'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'digest_method'
    var_4 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_custom_fallback_signers. Retrieved 2/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'utf-8'
    var_3 = lambda x: bytes(x, var_2)
    var_4 = lambda x: x
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = b'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'digest_method'
    var_1 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = b'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = b'custom'
    var_3 = lambda x: var_2
    var_4 = {}
    var_5 = lambda x: var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = b'secret'
    var_8 = module_0.Serializer(var_7, serializer=var_6)

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some-payload'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 4/6 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = 'custom'
    var_3 = lambda x: var_2
    var_4 = lambda x: var_2
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, serializer=var_5)

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_text_serializer_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #4
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
    var_4 = module_0.dumps(var_3)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'invalid json'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dumps_with_text_serializer. Retrieved 6/9 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 9/10 statements.
# Partially parsed test_dumps_with_bytes_input. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_unicode_input. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_empty_input. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_none_input. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_complex_input. Retrieved 13/14 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = '{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = b'{"key": "value"}'

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
    var_0 = 'different-secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'bytes input'
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'unicode input'
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = ''
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'nested'
    var_3 = 'list'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = {var_2: var_6, var_3: var_10}
    var_12 = var_1.dumps(var_11)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dumps_serializes_object_correctly. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 5/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 5/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_class. Retrieved 4/13 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallbacks. Retrieved 5/19 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 10/11 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallbacks. Retrieved 5/13 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = 0
    var_4 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = 0
    var_4 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = b'secret-key'
    var_4 = b'test-salt'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'test-salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]
    var_4 = 'test-salt'



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, serializer=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_isinstance_dict_predicate. Retrieved 11/12 statements.


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
    var_9 = 1
    var_10 = var_7[var_9]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_raises_bad_payload. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'invalid-json'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_with_string_secret_key. Retrieved 5/6 statements.
# Partially parsed test_serializer_constructor_with_bytes_secret_key. Retrieved 5/6 statements.
# Partially parsed test_serializer_constructor_with_list_secret_key. Retrieved 7/8 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = var_1.secret_keys
    var_4 = len(var_3)
    assert var_4 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.secret_keys
    var_3 = var_1.secret_keys
    var_4 = len(var_3)
    assert var_4 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)
    var_4 = var_3.secret_keys
    var_5 = var_3.secret_keys
    var_6 = len(var_5)
    assert var_6 == 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_iter_unsigners_with_empty_secret_keys. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test_key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_deserializes_payload. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'{"key": "value"}'
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = '!'
    var_4 = {var_0: var_3}
    var_5 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'invalid_json'



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_iter_unsigners_default_signer. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 4/14 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/15 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 6/22 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 7/19 statements.


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
    var_6 = 0
    var_7 = var_4[var_6]

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3

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

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'
    var_4 = 0
    var_5 = 1
    var_6 = 2



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dumps_with_default_serializer. Retrieved 7/9 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 6/8 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_different_object_types. Retrieved 11/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = '{"key": "value"}'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'salt'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'
    var_6 = var_1.dumps(var_4, var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 123
    var_3 = var_1.dumps(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.dumps(var_7)
    var_9 = 'string'
    var_10 = var_1.dumps(var_9)

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_iter_unsigners_default. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_custom_salt. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_dict. Retrieved 15/17 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 3/12 statements.
# Partially parsed test_iter_unsigners_with_multiple_fallback_signers. Retrieved 11/23 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 9/10 statements.
# Partially parsed test_iter_unsigners_with_key_rotation_and_fallback. Retrieved 19/22 statements.


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
    var_6 = 0
    var_7 = var_4[var_6]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'signer_kwargs'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'secret-key'
    var_6 = [var_4]
    var_7 = module_0.Serializer(var_5, fallback_signers=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = 1
    var_14 = var_9[var_13]

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'signer_kwargs'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '.'
    var_6 = {var_1: var_5}
    var_7 = 'secret-key'
    var_8 = 0
    var_9 = 1
    var_10 = 2

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'signer_kwargs'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'old-key'
    var_6 = 'new-key'
    var_7 = [var_5, var_6]
    var_8 = [var_4]
    var_9 = module_0.Serializer(var_7, fallback_signers=var_8)
    var_10 = var_9.iter_unsigners()
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = 1
    var_16 = var_11[var_15]
    var_17 = 2
    var_18 = var_11[var_17]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 8/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 4/6 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 10/14 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, obj: str(obj)
    var_5 = lambda self, s: s
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'CustomSigner'
    var_4 = ()
    var_5 = {}
    var_6 = 'key2'
    var_7 = 'value2'
    var_8 = {var_6: var_7}
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #25
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-payload'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #31
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dumps_with_string_serializer. Retrieved 6/9 statements.
# Partially parsed test_dumps_with_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_different_secret_key. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_list_secret_key. Retrieved 10/12 statements.
# Partially parsed test_dumps_with_bytes_input. Retrieved 5/7 statements.
# Partially parsed test_dumps_with_none_salt. Retrieved 8/10 statements.
# Partially parsed test_dumps_with_custom_serializer. Retrieved 6/12 statements.
# Partially parsed test_dumps_with_serializer_kwargs. Retrieved 9/12 statements.
# Partially parsed test_dumps_with_empty_object. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = '{"key": "value"}.'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'ensure_ascii'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = b'{"key": "value"}.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = '{"key": "value"}.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'different-secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = '{"key": "value"}.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom-salt'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)
    var_9 = '{"key": "value"}.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'bytes input'
    var_3 = var_1.dumps(var_2)
    var_4 = '"bytes input".'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = '{"key": "value"}.'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'custom-{'

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '{\n  "key": "value"\n}.'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = var_1.dumps(var_2)
    var_4 = '{}.'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'secret-key-2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: str(x)
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '|'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'secret-key'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 6/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)
    var_5 = var_0.dumps(var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 1/3 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = lambda obj: str(obj)
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1, serializer=var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'utf-8'
    var_1 = lambda obj: bytes(str(obj), var_0)
    var_2 = 'secret-key'
    var_3 = module_0.Serializer(var_2, serializer=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'indent'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, serializer_kwargs=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sep'
    var_1 = ';'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'invalid-json'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret-key'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = {}
    var_2 = b'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 5/8 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 4/6 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = ()
    var_2 = {}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dumps_serializes_object. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some-payload'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 6/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '?'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'sep'
    var_1 = '?'
    var_2 = {var_0: var_1}
    var_3 = '!'
    var_4 = {var_0: var_3}
    var_5 = 'secret'

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



# Parsed testcases at query #46
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
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
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'hello'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid-json'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #51
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 2/5 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sep'
    var_2 = '|'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_fallback_signers. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret_key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret_key'

def test_case_0():
    var_0 = 'secret_key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret_key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret_key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret_key'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_serializer_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'dumps'
    var_1 = 'loads'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'secret'
    var_6 = module_0.Serializer(var_5, serializer=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'secret'
    var_4 = module_0.Serializer(var_3, serializer_kwargs=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'secret'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 6/9 statements.


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
    var_0 = 'secret1'
    var_1 = 'secret2'
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
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = lambda x: str(x).encode()
    var_4 = lambda x: int(x.decode())
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, serializer=var_5)
    var_7 = lambda x: str(x).encode()
    var_8 = lambda x: int(x.decode())
    var_9 = {var_1: var_7, var_2: var_8}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'CustomSigner'
    var_2 = ()
    var_3 = {}
    var_4 = ()
    var_5 = {}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'digest_method'
    var_3 = 'hmac'
    var_4 = 'sha256'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, signer_kwargs=var_5)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'dumps'
    var_2 = 'loads'
    var_3 = lambda x: str(x).encode()
    var_4 = lambda x: int(x)
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Serializer(var_0, serializer=var_5)
    var_7 = lambda x: str(x).encode()
    var_8 = lambda x: int(x)
    var_9 = {var_1: var_7, var_2: var_8}



# Parsed testcases at query #60
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom_serializer'
    var_2 = module_0.Serializer(var_0, serializer=var_1)



