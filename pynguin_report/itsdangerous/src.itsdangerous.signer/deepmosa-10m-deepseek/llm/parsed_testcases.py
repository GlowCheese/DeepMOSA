####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_derive_key_default_secret_key. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_specific_secret. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_bytes_secret. Retrieved 4/5 statements.
# Partially parsed test_derive_key_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_django_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_hmac. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'other-secret'
    var_3 = var_1.derive_key(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'other-secret'
    var_3 = var_1.derive_key(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'django-concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = 'test-key'
    var_4 = var_2.derive_key(var_3)
    assert var_4 == b'test-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'unknown'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'novalue'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test.value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test.value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = 'test'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test'
    var_7 = module_0.Signer(var_0)
    var_8 = var_7.sign(var_4)
    var_9 = var_3.unsign(var_8)
    assert var_9 == b'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


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
    var_7 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

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
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'invalid_signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'secret2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 123
    var_5 = var_1.verify_signature(var_4, var_3)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 456
    var_5 = var_1.verify_signature(var_2, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'salt1'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'salt2'
    var_4 = module_0.Signer(var_0, var_3)
    var_5 = b'test_value'
    var_6 = var_2.get_signature(var_5)
    var_7 = var_4.verify_signature(var_5, var_6)
    assert var_7 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = 'invalid base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_sep_in_base64_alphabet_raises_error. Retrieved 1/5 statements.


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
    var_9 = var_1.algorithm.digest_method

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

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

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

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
    var_0 = []
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [])
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = b'!!!invalid-base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'invalid_signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.verify_signature(var_2, var_2)
    assert var_3 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'secret2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = 'wrong_secret'
    var_6 = module_0.Signer(var_5)
    var_7 = var_6.get_signature(var_4)
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'héllo'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'\xff\xfe'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_verify_signature_unicode_string_sig. Retrieved 7/9 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = b'invalidsignature'
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'secret-key-2'
    var_5 = module_0.Signer(var_4, var_1, var_2)
    var_6 = b'test_value'
    var_7 = var_3.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b''
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = b''
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = 'ascii'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = b'!!!invalidbase64!!!'
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = '.'
    var_5 = module_0.Signer(var_2, var_3, var_4)
    var_6 = b'test_value'
    var_7 = var_5.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'new-secret'
    var_5 = [var_0, var_4]
    var_6 = module_0.Signer(var_5, var_1, var_2)
    var_7 = b'test_value'
    var_8 = var_3.get_signature(var_7)
    var_9 = var_6.verify_signature(var_7, var_8)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'salt2'
    var_5 = module_0.Signer(var_0, var_4, var_2)
    var_6 = b'test_value'
    var_7 = var_3.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = ':'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_verify_signature_with_unicode_signature. Retrieved 5/7 statements.
# Partially parsed test_verify_signature_with_different_digest_method. Retrieved 2/7 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'invalid_signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b''
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'üñíçödé'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 'ascii'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'wrong-key'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_1.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/4 statements.


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
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
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
    var_1 = b'.'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'a'
    var_1 = b'secret'
    var_2 = module_0.Signer(var_1, sep=var_0)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_verify_signature_with_old_secret_key. Retrieved 8/9 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'invalid_signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test bytes'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test string'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = b'old-key'
    var_7 = var_3.verify_signature(var_4, var_5)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'different-key'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_1.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'not_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 4/6 statements.
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
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'key2'



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = 'invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_verify_signature_with_string_signature. Retrieved 5/7 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'invalid_signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'!!!not base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b''
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 'ascii'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = module_0.Signer(var_0)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_3.verify_signature(var_4, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'wrong-key'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_1.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = 'invalid base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test'
    var_4 = b'!!!invalid-base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False



