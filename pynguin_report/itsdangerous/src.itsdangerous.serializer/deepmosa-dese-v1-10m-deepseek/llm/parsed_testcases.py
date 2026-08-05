####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'sort_keys'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'secret-key'
    var_4 = module_0.Serializer(var_3, serializer_kwargs=var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_iter_unsigners_returns_default_signer_first. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_class. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_fallback_with_multiple_secret_keys. Retrieved 3/8 statements.


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
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'override-salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)

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
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 13/28 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dumps_with_text_serializer_returns_string. Retrieved 7/8 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 6/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = {}
    var_2 = module_0.Serializer(var_0, serializer_kwargs=var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'data'
    var_3 = 'salt1'
    var_4 = var_1.dumps(var_2, var_3)
    var_5 = 'salt2'
    var_6 = var_1.dumps(var_2, var_5)

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_with_default_serializer_and_text_serializer. Retrieved 7/9 statements.
# Partially parsed test_load_payload_with_custom_serializer_text. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_serializer_bytes. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_text_serializer_and_unicode_payload. Retrieved 3/12 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_1.dumps(var_4)
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_data'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some_data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'unicode_data'
    var_2 = 'utf-8'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_uses_provided_serializer_instead_of_default. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
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
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'hmac'
    var_12 = {var_10: var_11}
    var_13 = [var_12]
    var_14 = module_0.Serializer(var_2, var_3, serializer_kwargs=var_6, signer_kwargs=var_9, fallback_signers=var_13)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_iter_unsigners_yields_default_signer_first. Retrieved 8/9 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers. Retrieved 4/13 statements.
# Partially parsed test_iter_unsigners_with_fallback_signers_tuple. Retrieved 5/11 statements.
# Partially parsed test_iter_unsigners_with_multiple_secret_keys_and_fallback. Retrieved 4/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 0
    var_3 = 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'key'
    var_3 = 'fallback-key'
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
    var_2 = 'key'
    var_3 = 'fallback-key'
    var_4 = {var_2: var_3}

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'default-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = None
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)

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



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'my_salt'
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = 'digest_method'
    var_9 = 'sha256'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_0, var_1, var_2, var_3, var_4, var_7, var_11)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 8/10 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_returns_str. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_returns_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'salt'
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
    var_12 = module_0.Serializer(var_0, var_1, serializer_kwargs=var_4, signer_kwargs=var_7, fallback_signers=var_11)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_payload_with_json_serializer_and_bytes_payload. Retrieved 6/10 statements.
# Partially parsed test_load_payload_with_text_serializer_and_string_payload. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_serializer_parameter. Retrieved 2/5 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_invalid_data. Retrieved 2/6 statements.
# Partially parsed test_load_payload_with_serializer_returning_bytes. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_text_serializer_and_unicode_payload. Retrieved 2/10 statements.
# Partially parsed test_load_payload_passes_exception_original_error. Retrieved 2/6 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'42'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'[1, 2, 3]'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'invalid json'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bad'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_init_with_all_parameters. Retrieved 13/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = module_0.Serializer(var_2, var_3)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'none'
    var_11 = {var_7: var_10}
    var_12 = [var_11]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serializer_constructor_with_all_arguments. Retrieved 13/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, signer=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer_kwargs=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, signer_kwargs=var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_str. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dumps_returns_serialized_type. Retrieved 4/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)
    var_3 = var_0.dumps(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer_str. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_serializer_bytes. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_payload_is_text_false. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_all_parameters. Retrieved 12/23 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_positional_serializer. Retrieved 2/11 statements.
# Partially parsed test_serializer_constructor_with_keyword_serializer. Retrieved 1/10 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.fallback_signers
    var_7 = len(var_6)
    assert var_7 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serializer_with_explicit_serializer_skips_default. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_text_serializer_kwarg. Retrieved 6/9 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'utf-8'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_loads_accepts_serialized_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'\x00\x01'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.loads(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.loads(var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/7 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)
    var_3 = var_0.loads(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 123
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_serializer_returning_bytes. Retrieved 1/8 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #36
#--------------------------




import src.itsdangerous.signer as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = [var_1]
    var_3 = module_1.Serializer(var_0, fallback_signers=var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 9/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_3.dumps(var_7)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_serializer_str. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.


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
    var_0 = 'secret'
    var_1 = 'pepper'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, signer=var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 3/6 statements.


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
    var_4 = 'string'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

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
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_loads_accepts_serialized_input. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_payload'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'serialized_data'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 12345
    var_2 = var_0.loads(var_1)
    assert var_2 == 12345

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.loads(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)
    assert var_2 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'hello'
    var_2 = var_0.loads(var_1)
    assert var_2 == 'hello'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'"test"'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/14 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret-key'

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
    var_9 = 'secret-key'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
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
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/3 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_serializer_init_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 10/14 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 3/6 statements.


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
    var_0 = 'secret'
    var_1 = b'custom_salt'
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
    var_4 = lambda x: str(x)
    var_5 = staticmethod(var_4)
    var_6 = lambda x: eval(x)
    var_7 = staticmethod(var_6)
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_dumps_accepts_any_object. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)
    assert var_4 == b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)
    assert var_2 == b'null'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = ''
    var_2 = var_0.dumps(var_1)
    assert var_2 == b'""'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = 'b'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = var_0.dumps(var_8)
    assert var_9 == b'{"a": [1, 2, {"b": true}]}'



# Parsed testcases at query #50
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #51
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_all_parameters. Retrieved 14/22 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
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



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_load_payload_predicate_false_with_custom_serializer. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test'



# Parsed testcases at query #57
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'explicit_salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_serializer_constructor_default_serializer. Retrieved 2/6 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text_serializer. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #60
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_load_payload_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_override_default_serializer. Retrieved 3/13 statements.


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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)



# Parsed testcases at query #63
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_constructor_serializer_bytes. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_dumps_returns_expected_type. Retrieved 5/7 statements.
# Partially parsed test_dumps_returns_same_type_for_different_inputs. Retrieved 5/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 123
    var_2 = var_0.dumps(var_1)
    var_3 = ''
    var_4 = var_0.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'hello'
    var_2 = var_0.dumps(var_1)
    var_3 = 42
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'"test"'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #68
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_payload'
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
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
    var_1 = None
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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = 'key_derivation'
    var_2 = 'none'
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
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_serializer_constructor_with_explicit_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_signer_class. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_not_text_serializer. Retrieved 7/8 statements.
# Partially parsed test_dumps_returns_str_when_text_serializer. Retrieved 4/6 statements.
# Partially parsed test_dumps_with_custom_salt. Retrieved 5/6 statements.
# Partially parsed test_dumps_with_none_salt. Retrieved 5/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1, serializer_kwargs=var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = 'test'
    var_3 = module_0.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'data'
    var_4 = var_2.dumps(var_3, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dump_payload(var_2)
    var_4 = var_1.make_signer()
    var_5 = var_4.sign(var_3)
    var_6 = var_1.dumps(var_2)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_dumps_returns_serialized_data. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_none. Retrieved 4/5 statements.
# Partially parsed test_dumps_with_integer. Retrieved 3/4 statements.
# Partially parsed test_dumps_with_list. Retrieved 6/7 statements.
# Partially parsed test_dumps_with_dict. Retrieved 5/6 statements.
# Partially parsed test_dumps_returns_bytes. Retrieved 4/5 statements.
# Partially parsed test_dumps_returns_string. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)
    assert var_2 == 'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'None'
    var_2 = None
    var_3 = var_0.dumps(var_2)
    assert var_3 == 'None'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)
    assert var_2 == '42'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test'
    var_2 = 'test'
    var_3 = var_0.dumps(var_2)
    assert var_3 == b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)
    assert var_2 == 'test'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 10/13 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret'

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
    var_9 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = b'secret1'
    var_1 = b'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'inner'
    var_2 = module_0.Serializer(var_1)
    var_3 = module_0.Serializer(var_0, serializer=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 4/11 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 5/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

def test_case_0():
    var_0 = 'secret'
    var_1 = 42

def test_case_0():
    var_0 = 'secret'
    var_1 = 123

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'original'
    var_3 = var_1.dump_payload(var_2)

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
    var_2 = 'héllo'
    var_3 = var_1.dump_payload(var_2)
    var_4 = var_1.load_payload(var_3)
    assert var_4 == 'héllo'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_init_with_custom_fallback_signers. Retrieved 1/6 statements.


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
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: eval(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'test-secret'

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x).encode()
    var_5 = lambda self, x: eval(x.decode())
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'test-secret'

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
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'test-secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_non_text_serializer. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/15 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'old_key'
    var_1 = b'new_key'
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



# Parsed testcases at query #83
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 3.14
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.loads(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
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
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 9/12 statements.


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
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 'secret'

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_loads_accepts_serialized_type. Retrieved 3/4 statements.
# Partially parsed test_loads_handles_complex_data. Retrieved 3/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'binary_data'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = ''
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 12345
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_and_signer_kwargs. Retrieved 4/5 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
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
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_serializer_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 8/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'mysecret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'mysecret'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, obj, **kwargs: str(obj)
    var_5 = lambda self, obj: obj
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = 'secret'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_is_false. Retrieved 7/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #91
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_load_payload_with_default_serializer_and_text_serializer. Retrieved 10/14 statements.
# Partially parsed test_load_payload_with_default_serializer_and_bytes_serializer. Retrieved 10/14 statements.
# Partially parsed test_load_payload_with_custom_serializer_override. Retrieved 15/21 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 10/20 statements.
# Partially parsed test_load_payload_with_override_serializer_text_false. Retrieved 16/22 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'TextSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, x: x
    var_6 = '{}'
    var_7 = lambda self, x: var_6
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = b'"test"'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, x: x
    var_6 = b'{}'
    var_7 = lambda self, x: var_6
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = b'test'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'TextSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, x: x
    var_6 = '{}'
    var_7 = lambda self, x: var_6
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = 'CustomSerializer'
    var_10 = ()
    var_11 = 'custom'
    var_12 = lambda self, x: var_11
    var_13 = {var_3: var_12}
    var_14 = b'"test"'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'FailingSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = ()
    var_6 = 'error'
    var_7 = '{}'
    var_8 = lambda self, x: var_7
    var_9 = b'"test"'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, x: x
    var_6 = b'{}'
    var_7 = lambda self, x: var_6
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = 'TextSerializer'
    var_10 = ()
    var_11 = lambda self, x: x
    var_12 = '{}'
    var_13 = lambda self, x: var_12
    var_14 = {var_3: var_11, var_4: var_13}
    var_15 = b'"test"'



# Parsed testcases at query #93
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = b'extra'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 3/10 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 15/24 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = {}
    var_3 = module_1.dumps(var_2)
    assert var_3 == '{}'

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
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 123
    var_2 = module_0.dumps(var_1)
    assert var_2 == '123'

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 123
    var_2 = module_0.dumps(var_1)
    assert var_2 == b'123'

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import json as module_0

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
    var_13 = 123
    var_14 = module_0.dumps(var_13)
    assert var_14 == '123'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_all_parameters. Retrieved 14/24 statements.


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
    var_1 = None
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

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = 'key_derivation'
    var_11 = 'hmac'
    var_12 = {var_10: var_11}
    var_13 = [var_12]



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer_class. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_arguments. Retrieved 12/23 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
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
    var_1 = 'custom-salt'
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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = 'pepper'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/8 statements.
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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_dumps_returns_serialized_type. Retrieved 4/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)
    var_3 = var_0.dumps(var_1)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_constructor_with_fallback_signers. Retrieved 1/3 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: int(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_false. Retrieved 5/15 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.


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
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/2 statements.


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
    var_1 = None
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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'test'
    var_5 = lambda self, obj: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 11/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = [var_3, var_4]
    var_6 = b'custom_salt'
    var_7 = 'indent'
    var_8 = 2
    var_9 = {var_7: var_8}
    var_10 = 'digest_method'



# Parsed testcases at query #110
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)



# Parsed testcases at query #111
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer_and_text. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_override_serializer_text. Retrieved 3/8 statements.
# Partially parsed test_load_payload_with_override_serializer_bytes. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

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
    var_2 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.load_payload(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dump_payload(var_2)
    var_4 = None
    var_5 = var_1.load_payload(var_3, var_4)
    assert var_5 == 'test'



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_serializer_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/14 statements.
# Partially parsed test_serializer_constructor_with_signer_and_signer_kwargs. Retrieved 4/7 statements.
# Partially parsed test_serializer_constructor_with_all_params. Retrieved 21/27 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_9 = 'secret'
    var_10 = b'custom_salt'
    var_11 = 'sort_keys'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = 'digest_method'
    var_15 = 'sha256'
    var_16 = {var_14: var_15}
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = [var_19]



# Parsed testcases at query #114
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'{"key": "value"}'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'[1, 2, 3]'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'"hello"'
    var_2 = var_0.loads(var_1)
    assert var_2 == 'hello'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'42'
    var_2 = var_0.loads(var_1)
    assert var_2 == 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'true'
    var_2 = var_0.loads(var_1)
    assert var_2 is True

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'null'
    var_2 = var_0.loads(var_1)
    assert var_2 is None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_init_with_custom_serializer_bytes. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = module_0.is_text_serializer(var_2)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_iter_unsigners_yields_signer_for_each_secret_key_in_fallback. Retrieved 3/8 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 4/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'algorithm'
    var_2 = 'sha512'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'algorithm'
    var_2 = 'sha512'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = 'custom-salt'
    var_7 = var_5.iter_unsigners(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'algorithm'
    var_2 = 'sha512'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_data'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 123
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 9/12 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 'secret'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer_returns_str. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_custom_serializer_returns_bytes. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_explicit_serializer_returns_str. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_explicit_serializer_returns_bytes. Retrieved 3/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_text_serializer_uses_utf8_decode. Retrieved 3/12 statements.
# Partially parsed test_load_payload_with_bytes_serializer_passes_bytes_directly. Retrieved 2/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'test'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'raw'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_when_is_text_is_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback_evaluates_isinstance_tuple_true. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = {}
    var_5 = None
    var_6 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 8/14 statements.


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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'invalid json'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_with_custom_text_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_custom_bytes_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_explicit_serializer_override. Retrieved 6/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer_and_text_input. Retrieved 4/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'data'
    var_2 = 'test'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'
    var_1 = 'data'
    var_2 = 'test'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fallback_signers_not_none. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/2 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loads_returns_any_type. Retrieved 4/5 statements.
# Partially parsed test_loads_accepts_serialized_input. Retrieved 3/4 statements.
# Partially parsed test_loads_handles_none_input. Retrieved 3/4 statements.
# Partially parsed test_loads_handles_complex_types. Retrieved 9/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = 'test'
    var_3 = var_0.loads(var_2)
    assert var_3 == 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'data'
    var_2 = var_0.loads(var_1)
    assert var_2 == 'data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)
    assert var_2 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = '{}'
    var_8 = var_0.loads(var_7)



# Parsed testcases at query #18
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dumps_with_str_serializer_returns_str. Retrieved 5/6 statements.
# Partially parsed test_dumps_with_bytes_serializer_returns_bytes. Retrieved 4/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = 'test'
    var_4 = var_2.dumps(var_3)

import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = 'test'
    var_3 = module_0.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 123
    var_3 = var_1.dumps(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.dumps(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_1.dumps(var_11)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = 'custom_salt'
    var_4 = var_1.dumps(var_2, var_3)
    var_5 = var_1.dumps(var_2, var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = 'salt1'
    var_4 = var_1.dumps(var_2, var_3)
    var_5 = 'salt2'
    var_6 = var_1.dumps(var_2, var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = ''
    var_3 = var_1.dumps(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = None
    var_3 = var_1.dumps(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serializer_uses_provided_serializer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_payload_with_explicit_serializer. Retrieved 3/4 statements.
# Partially parsed test_load_payload_with_text_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

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

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = '68656c6c6f'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'bad'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_is_text_serializer_is_false. Retrieved 7/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, serializer=var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/4 statements.


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
    var_1 = None
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
    var_2 = module_0.Serializer(var_0, serializer=var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: x
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret-key'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_arguments. Retrieved 11/25 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = 'custom-salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = 'digest_method'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/3 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = None
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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #30
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 9/23 statements.


def test_case_0():
    var_0 = b'test-secret'
    var_1 = b'test-salt'
    var_2 = 'CustomSigner'
    var_3 = {}
    var_4 = 'key_derivation'
    var_5 = 'none'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/8 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_loads_calls_loads_with_payload. Retrieved 2/8 statements.
# Partially parsed test_loads_returns_any_type. Retrieved 2/8 statements.
# Partially parsed test_loads_with_none_payload. Retrieved 2/8 statements.
# Partially parsed test_loads_with_complex_object. Retrieved 2/8 statements.
# Partially parsed test_loads_with_integer_payload. Retrieved 2/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'test_payload'

import json as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = module_0.loads(var_0)
    assert var_1 == 42

import json as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.loads(var_0)
    assert var_1 is None

import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)

import json as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.loads(var_0)
    assert var_1 == 10



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 15/16 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = None
    var_5 = 'indent'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 'digest_method'
    var_9 = 'sha256'
    var_10 = {var_8: var_9}
    var_11 = 'key_derivation'
    var_12 = 'hmac'
    var_13 = {var_11: var_12}
    var_14 = [var_13]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_constructor_with_serializer_str. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_serializer_bytes. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dumps_with_default_json_serializer_returns_bytes. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'test'
    var_3 = var_1.dumps(var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dumps_is_text_serializer_false. Retrieved 7/14 statements.


import json as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = b'test-salt'
    var_2 = 'digest_method'
    var_3 = 'sha256'
    var_4 = {var_2: var_3}
    var_5 = 'test data'
    var_6 = module_0.dumps(var_5)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_all_parameters. Retrieved 14/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
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
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'custom'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.Serializer(var_3)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/3 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_tuple. Retrieved 4/7 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_signer_class. Retrieved 1/3 statements.


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
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
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
    var_4 = 'text'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

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

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/2 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_payload_with_custom_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 4/13 statements.
# Partially parsed test_load_payload_with_explicit_serializer. Retrieved 4/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)

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
    var_2 = b'invalid_payload'
    var_3 = var_1.load_payload(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serializer_constructor_default. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_custom_serializer_text. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_fallback_signers_tuple. Retrieved 4/7 statements.
# Partially parsed test_serializer_constructor_fallback_signers_class. Retrieved 1/5 statements.


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
    var_1 = 'my_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'my_salt'
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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_iterations'
    var_2 = 100
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_iterations'
    var_2 = 50
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_iterations'
    var_2 = 50
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer_text. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_load_payload_predicate_false_when_serializer_not_text. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_iter_unsigners_handles_tuple_fallback. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = 1



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/2 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/8 statements.


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
    var_1 = None
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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_text. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/7 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_constructor_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_constructor_with_bytes_serializer. Retrieved 8/11 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_all_parameters. Retrieved 20/25 statements.


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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: eval(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret-key'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x).encode()
    var_5 = lambda self, x: eval(x.decode())
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret-key'

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: eval(x)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = [var_7, var_8]
    var_10 = 'custom-salt'
    var_11 = 'indent'
    var_12 = 2
    var_13 = {var_11: var_12}
    var_14 = 'key_derivation'
    var_15 = 'hmac'
    var_16 = {var_14: var_15}
    var_17 = 'none'
    var_18 = {var_14: var_17}
    var_19 = [var_18]



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_load_payload_is_text_false_with_bytes_serializer. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #54
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_default_serializer. Retrieved 2/5 statements.
# Partially parsed test_serializer_constructor_with_custom_default_signer. Retrieved 2/5 statements.
# Partially parsed test_serializer_constructor_with_custom_default_fallback_signers. Retrieved 5/8 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = 'secret'
    var_4 = module_0.Serializer(var_3)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_loads_accepts_string. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_payload'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'hello'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'data'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_dumps_returns_serialized_type. Retrieved 13/19 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = var_0.dumps
    var_2 = 'test'
    var_3 = 123
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = None
    var_12 = True



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.dumps(var_1)
    assert var_2 == 'test_string'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)
    assert var_2 == 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)
    assert var_2 is None



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_serializer_init_default_serializer. Retrieved 3/4 statements.
# Partially parsed test_serializer_init_with_custom_serializer_str. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.signer

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_serializer_constructor_with_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_class. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers_tuple. Retrieved 4/10 statements.
# Partially parsed test_serializer_constructor_inherits_default_fallback_signers. Retrieved 2/4 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'hmac'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'



# Parsed testcases at query #62
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 12/26 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = module_0.is_text_serializer(var_2)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'indent'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = [var_10]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_signer. Retrieved 1/4 statements.


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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_load_payload_predicate_false_when_is_text_false. Retrieved 4/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'not-valid-json'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_load_payload_when_serializer_is_bytes_serializer_and_is_text_is_false. Retrieved 5/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #67
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_iter_unsigners_yields_fallback_signers. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_yields_multiple_fallback_signers. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_yields_fallback_with_tuple_config. Retrieved 2/8 statements.
# Partially parsed test_iter_unsigners_yields_fallback_with_class_only. Retrieved 1/6 statements.
# Partially parsed test_iter_unsigners_yields_multiple_fallback_with_mixed_configs. Retrieved 3/9 statements.
# Partially parsed test_iter_unsigners_yields_fallback_for_each_secret_key. Retrieved 3/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'provided_salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)

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

def test_case_0():
    var_0 = 'secret'
    var_1 = {}

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = {}
    var_2 = {}

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_2, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 3



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_text_serializer. Retrieved 1/8 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #70
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_all_custom_parameters. Retrieved 31/37 statements.


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
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret-key'

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'bytes'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret-key'

def test_case_0():
    var_0 = 'CustomSigner'
    var_1 = '__init__'
    var_2 = 'sign'
    var_3 = 'unsign'
    var_4 = None
    var_5 = lambda self, keys, salt, **kwargs: var_4
    var_6 = lambda self, x: x
    var_7 = lambda self, x: x
    var_8 = {var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_9 = 'CustomSigner'
    var_10 = '__init__'
    var_11 = 'sign'
    var_12 = 'unsign'
    var_13 = None
    var_14 = lambda self, keys, salt, **kwargs: var_13
    var_15 = lambda self, x: x
    var_16 = lambda self, x: x
    var_17 = {var_10: var_14, var_11: var_15, var_12: var_16}
    var_18 = 'key1'
    var_19 = 'key2'
    var_20 = [var_18, var_19]
    var_21 = 'custom-salt'
    var_22 = 'sort_keys'
    var_23 = True
    var_24 = {var_22: var_23}
    var_25 = 'key_derivation'
    var_26 = 'none'
    var_27 = {var_25: var_26}
    var_28 = 'hmac'
    var_29 = {var_25: var_28}
    var_30 = [var_29]



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_serializer_init_with_serializer_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #73
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 5/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda self, x: str(x)
    var_5 = lambda self, x: x
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'secret'

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_loads_accepts_string_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_bytes_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_integer_payload. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_list_payload. Retrieved 6/7 statements.
# Partially parsed test_loads_accepts_dict_payload. Retrieved 5/6 statements.
# Partially parsed test_loads_accepts_empty_string. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_payload'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test_bytes'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.loads(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.loads(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)
    assert var_2 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = ''
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #77
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer_positional. Retrieved 2/8 statements.
# Partially parsed test_serializer_init_with_bytes_serializer_keyword. Retrieved 1/7 statements.
# Partially parsed test_serializer_init_with_custom_signer_class. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_multiple_secret_keys_and_fallback_signers. Retrieved 4/8 statements.


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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'salt'

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'digest_method'



# Parsed testcases at query #79
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)
    assert var_2 == 42

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)
    assert var_2 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = ''
    var_2 = var_0.dumps(var_1)
    assert var_2 == ''

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 0
    var_2 = var_0.dumps(var_1)
    assert var_2 == 0

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = False
    var_2 = var_0.dumps(var_1)
    assert var_2 is False

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = []
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = {}
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_positional. Retrieved 2/10 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_keyword. Retrieved 1/9 statements.
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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_0 = 'secret'
    var_1 = b'itsdangerous'

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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/8 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 9/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'test'
    var_5 = lambda self, x: var_4
    var_6 = lambda self, x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.serializer
    var_3 = module_0.is_text_serializer(var_2)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_serializer_constructor_default_serializer. Retrieved 2/3 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 10/14 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_4 = 'dumped'
    var_5 = lambda self, x: var_4
    var_6 = 'loaded'
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

def test_case_0():
    var_0 = 'TextSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = 'dumped'
    var_5 = lambda self, x: var_4
    var_6 = 'loaded'
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_iter_unsigners_yields_default_signer_first. Retrieved 7/8 statements.
# Partially parsed test_iter_unsigners_uses_fallback_as_tuple. Retrieved 4/10 statements.
# Partially parsed test_iter_unsigners_uses_fallback_as_class. Retrieved 1/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_3[var_5]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'algorithm'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret-key'

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
    assert var_9 == 3



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_serializer_constructor_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serializer_constructor_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/11 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key_bytes'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_str'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_serializer_constructor_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_custom_signer. Retrieved 1/2 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_2 = 'none'
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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'bytes_secret'
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



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_dumps_returns_bytes_with_bytes_serializer. Retrieved 7/10 statements.
# Partially parsed test_dumps_returns_str_with_text_serializer. Retrieved 7/10 statements.
# Partially parsed test_dumps_uses_custom_salt. Retrieved 14/18 statements.
# Partially parsed test_dumps_calls_dump_payload. Retrieved 12/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'data'
    var_5 = 'test'
    var_6 = var_3.dumps(var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'data'
    var_5 = 'test'
    var_6 = var_3.dumps(var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'data'
    var_5 = 'Signer'
    var_6 = ()
    var_7 = 'sign'
    var_8 = b'signed'
    var_9 = lambda self, payload: var_8
    var_10 = {var_7: var_9}
    var_11 = 'test'
    var_12 = 'custom-salt'
    var_13 = var_3.dumps(var_11, var_12)
    assert var_13 == b'signed'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = b'payload'
    var_5 = 'Signer'
    var_6 = ()
    var_7 = 'sign'
    var_8 = lambda self, payload: payload
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = var_3.dumps(var_10)
    assert var_11 == b'payload'



# Parsed testcases at query #89
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_loads_accepts_serialized_input. Retrieved 3/4 statements.
# Partially parsed test_loads_accepts_list_input. Retrieved 6/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.loads(var_1)
    assert var_2 is None

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.loads(var_4)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_explicit_text_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/11 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/10 statements.
# Partially parsed test_load_payload_with_override_bytes_serializer. Retrieved 3/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'"test"'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'base'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'\x00\x01'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_load_payload_is_text_false. Retrieved 4/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #93
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test'
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret'

def test_case_0():
    var_0 = 'BytesSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_when_serializer_returns_bytes_and_payload_decode_fails. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'\xff\xfe\x00\x00'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_serializer_keyword_argument. Retrieved 1/8 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_iter_unsigners_with_tuple_fallback. Retrieved 6/19 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 4/10 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_load_payload_with_bytes_serializer_evaluates_is_text_false. Retrieved 14/19 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = module_0.Serializer(var_0, serializer=var_2)
    var_4 = 'BytesSerializer'
    var_5 = ()
    var_6 = 'loads'
    var_7 = 'dumps'
    var_8 = lambda self, x: x
    var_9 = lambda self, x: x
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = b'test'
    var_12 = var_3.load_payload(var_11)
    var_13 = None
    assert var_13 == b'test'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 14/24 statements.


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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
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



# Parsed testcases at query #101
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 11/18 statements.


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
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = 'sort_keys'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'digest_method'
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_constructor_default_serializer. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serializer_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 3/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/5 statements.
# Partially parsed test_serializer_constructor_with_all_parameters. Retrieved 16/28 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = {}
    var_2 = module_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import json as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
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
    var_14 = {}
    var_15 = module_0.dumps(var_14)



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_dumps_returns_bytes_when_serializer_is_not_text. Retrieved 13/19 statements.


import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'dumps'
    var_4 = 'loads'
    var_5 = b'data'
    var_6 = lambda self, obj: var_5
    var_7 = lambda self, data: data
    var_8 = {var_3: var_6, var_4: var_7}
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_0.dumps(var_11)



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_dumps_returns_string_when_serialized_is_string. Retrieved 3/4 statements.
# Partially parsed test_dumps_returns_bytes_when_serialized_is_bytes. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 123
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'hello'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'data'
    var_2 = var_0.dumps(var_1)



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_binary_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'mysalt'
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
    var_4 = '{}'
    var_5 = lambda self, o: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

def test_case_0():
    var_0 = 'BinarySerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'{}'
    var_5 = lambda self, o: var_4
    var_6 = {}
    var_7 = lambda self, s: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 10/13 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 3/6 statements.


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
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_9 = 'secret'

def test_case_0():
    var_0 = 'CustomSerializer'
    var_1 = ()
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'{}'
    var_5 = lambda self, x: var_4
    var_6 = {}
    var_7 = lambda self, x: var_6
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = 'secret'

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_constructor_with_custom_text_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_custom_signer. Retrieved 1/4 statements.


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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_dumps_returns_expected_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test_string'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 42
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = None
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.dumps(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'data'
    var_2 = var_0.dumps(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 1
    var_2 = 'a'
    var_3 = 3.14
    var_4 = (var_1, var_2, var_3)
    var_5 = var_0.dumps(var_4)



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_loads_accepts_serialized_type. Retrieved 3/4 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test data'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'test payload'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'test string'
    var_2 = var_0.loads(var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b''
    var_2 = var_0.loads(var_1)



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_dumps_with_non_text_serializer_returns_bytes. Retrieved 5/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_load_payload_with_non_text_serializer_and_invalid_payload_raises_bad_payload. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'invalid json'



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/7 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_iter_unsigners_elif_branch_tuple_fallback. Retrieved 6/14 statements.


def test_case_0():
    var_0 = b'test-secret'
    var_1 = 'digest_method'
    var_2 = 'sha256'
    var_3 = {var_1: var_2}
    var_4 = b'test-salt'
    var_5 = 1



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serializer_init_with_custom_signer. Retrieved 1/4 statements.
# Partially parsed test_serializer_init_with_bytes_serializer. Retrieved 1/8 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'utf-8'
    var_2 = bytes(var_0, var_1)
    var_3 = module_0.Serializer(var_2)

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
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'indent'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'key_derivation'
    var_2 = 'hmac'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'my-secret-key'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)



# Parsed testcases at query #118
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_custom_signer_class. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
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
    var_1 = 'key_derivation'
    var_2 = 'none'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key_derivation'
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #119
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #120
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #121
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/2 statements.
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
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

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
    var_1 = 'sort_keys'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

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
    var_1 = 'none'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #122
#--------------------------

# Partially parsed test_serializer_constructor_serializer_custom. Retrieved 1/2 statements.
# Partially parsed test_serializer_constructor_is_text_serializer_false. Retrieved 1/8 statements.
# Partially parsed test_serializer_constructor_signer_custom. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

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
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



