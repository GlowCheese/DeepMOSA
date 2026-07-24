####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_derive_key_with_concat_derivation. Retrieved 6/8 statements.
# Partially parsed test_derive_key_with_django_concat_derivation. Retrieved 8/10 statements.
# Partially parsed test_derive_key_with_hmac_derivation. Retrieved 5/9 statements.
# Partially parsed test_derive_key_with_custom_salt. Retrieved 6/8 statements.
# Partially parsed test_derive_key_with_custom_secret_key. Retrieved 7/9 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = b'itsdangerous.Signer'
    var_5 = var_4 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'django-concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = b'itsdangerous.Signer'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'itsdangerous.Signer'
    var_4 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    assert var_3 == b'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'custom'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    var_5 = var_1 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'custom'
    var_4 = var_2.derive_key(var_3)
    var_5 = b'itsdangerous.Signer'
    var_6 = var_5 + var_3

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'invalid'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_3.sep
    assert var_7 == b'.'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.algorithm
    var_10 = bool(var_3.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'|'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'hmac'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 'my-secret-key'

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'invalid'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'hmac'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm
    var_10 = bool(var_2.algorithm is var_0)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '-'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_with_string_input. Retrieved 4/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value.sep.invalid_signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value_without_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "No b'.' found in value"

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'secret-key-2'
    var_3 = module_0.Signer(var_2)
    var_4 = 'value'
    var_5 = var_1.sign(var_4)
    var_6 = var_3.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'malformed!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_keys. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'hmac'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'other-value'
    var_4 = var_1.get_signature(var_2)
    var_5 = var_1.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'malformed!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = 'django-concat'
    var_4 = module_0.Signer(var_0, key_derivation=var_3)
    var_5 = b'test-value'
    var_6 = var_2.get_signature(var_5)
    var_7 = var_4.get_signature(var_5)
    var_8 = var_2.verify_signature(var_5, var_6)
    assert var_8 is True
    var_9 = var_4.verify_signature(var_5, var_7)
    assert var_9 is True
    var_10 = var_2.verify_signature(var_5, var_7)
    assert var_10 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_signer_constructor_with_string_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_key_list. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'|'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_3.sep
    assert var_7 == b'.'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid-base64-sig'
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid-base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'wrong-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = b'different-value'
    var_5 = var_1.verify_signature(var_4, var_3)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'other-key'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_1.verify_signature(var_4, var_5)
    assert var_6 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid-base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_list_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid-base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Signer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_4.sep
    assert var_8 == b'.'
    var_9 = var_4.key_derivation
    assert var_9 == 'django-concat'
    var_10 = var_4.algorithm
    var_11 = bool(var_4.algorithm is not None)
    assert var_11 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'|'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.sep
    assert var_6 == b'.'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 'my-secret-key'

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'other-value'
    var_4 = var_1.get_signature(var_2)
    var_5 = var_1.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = staticmethod(var_0)
    var_2 = 'secret'
    var_3 = module_0.Signer(var_2, digest_method=var_1)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = bool(var_3.digest_method == var_1)
    assert var_10 is True
    var_11 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm
    var_10 = bool(var_2.algorithm == var_0)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = b'world'
    var_4 = var_1.get_signature(var_3)
    var_5 = var_1.verify_signature(var_2, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = b'malformed$$$'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'hello'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'expired-key'
    var_1 = 'current-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'hello'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.algorithm
    var_10 = bool(var_3.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom_salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is var_0)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_derive_key_with_concat_derivation. Retrieved 6/8 statements.
# Partially parsed test_derive_key_with_django_concat_derivation. Retrieved 8/10 statements.
# Partially parsed test_derive_key_with_hmac_derivation. Retrieved 5/9 statements.
# Partially parsed test_derive_key_with_custom_salt. Retrieved 7/9 statements.
# Partially parsed test_derive_key_with_custom_secret_key. Retrieved 7/9 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = b'itsdangerous.Signer'
    var_5 = var_4 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'django-concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = b'itsdangerous.Signer'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'itsdangerous.Signer'
    var_4 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    assert var_3 == b'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'custom'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    var_5 = b'custom'
    var_6 = var_5 + var_0

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'custom'
    var_4 = var_2.derive_key(var_3)
    var_5 = b'itsdangerous.Signer'
    var_6 = var_5 + var_3

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'invalid'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'other-value'
    var_4 = var_1.get_signature(var_2)
    var_5 = var_1.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'malformed-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = b'invalid'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = var_1.get_signature(var_2)
    var_4 = b'world'
    var_5 = var_1.verify_signature(var_4, var_3)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'hello'
    var_3 = b'not-base64!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'hello'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'hello'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_list_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom_salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm
    var_10 = bool(var_2.algorithm is var_0)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = b'wrong-value'
    var_5 = var_1.verify_signature(var_4, var_3)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'malformed-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_keys. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/6 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/8 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom-salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm
    var_10 = bool(var_2.algorithm is var_0)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = staticmethod(var_0)
    var_2 = 'secret-key'
    var_3 = module_0.Signer(var_2, digest_method=var_1)
    var_4 = var_3.digest_method
    var_5 = bool(var_3.digest_method == var_1)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm == var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'cannot be used'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_verify_signature_with_string_signature. Retrieved 5/7 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = var_1.get_signature(var_2)
    var_4 = b'wrong-value'
    var_5 = var_1.verify_signature(var_4, var_3)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = b'malformed!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = module_0.Signer(var_1)
    var_8 = var_7.get_signature(var_4)
    var_9 = var_3.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = var_3.verify_signature(var_4, var_8)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 'ascii'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.algorithm
    var_10 = bool(var_3.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom_salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is var_0)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is not None)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid-base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = 'secret2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret1', b'secret2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom_salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm
    var_10 = bool(var_2.algorithm is var_0)
    assert var_10 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = 'invalid_base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_list_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.digest_method
    var_10 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'custom-salt'
    var_6 = var_2.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_2.digest_method
    var_8 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.digest_method
    var_9 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.key_derivation
    assert var_5 == 'concat'
    var_6 = var_2.digest_method
    var_7 = var_2.algorithm

def test_case_0():
    var_0 = 'my-secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'my-secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.algorithm
    var_6 = bool(var_2.algorithm == var_0)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'my-secret-key'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_2.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_2.digest_method
    var_8 = var_2.algorithm



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'malformed-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'other-secret-key'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_1.verify_signature(var_4, var_5)
    assert var_6 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_iterable_secret_key. Retrieved 5/6 statements.
# Partially parsed test_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/6 statements.
# Partially parsed test_signer_constructor_with_none_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.sep
    assert var_6 == b'.'
    var_7 = var_3.salt
    assert var_7 == b'itsdangerous.Signer'
    var_8 = var_3.key_derivation
    assert var_8 == 'django-concat'
    var_9 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'custom_salt'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'|'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'concat'
    var_8 = var_2.algorithm

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm
    var_9 = bool(var_2.algorithm is var_0)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.sep
    assert var_5 == b'.'
    var_6 = var_2.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_2.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_2.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '-'
    var_2 = module_0.Signer(var_0, sep=var_1)



