####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test-value'
    var_3 = b'invalid-signature'
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = module_0.Signer(var_0)
    var_5 = 'value'
    var_6 = var_4.get_signature(var_5)
    var_7 = var_3.verify_signature(var_5, var_6)
    var_8 = bool(var_7)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = ''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = 'value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_derive_key_default_secret_key. Retrieved 4/5 statements.
# Partially parsed test_derive_key_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_django_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_hmac. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_explicit_secret. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'django-concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    assert var_3 == b'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'other'
    var_1 = module_0.Signer(var_0)
    var_2 = b'custom'
    var_3 = var_1.derive_key(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'unknown'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = b'invalid-signature'
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'bytes-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'string-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = '.'
    var_5 = module_0.Signer(var_2, var_3, var_4)
    var_6 = b'test-value'
    var_7 = var_5.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = '.'
    var_5 = module_0.Signer(var_2, var_3, var_4)
    var_6 = b'test-value'
    var_7 = module_0.Signer(var_0, var_3, var_4)
    var_8 = var_7.get_signature(var_6)
    var_9 = var_5.verify_signature(var_6, var_8)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = b'!!!invalid-base64!!!'
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

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
    var_2 = '-'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_unicode_value. Retrieved 7/8 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test_value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.unsign(var_3)
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value.'
    var_4 = b'wrong_signature'
    var_5 = var_3 + var_4
    var_6 = var_2.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b''
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b''

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = b':'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = 'test'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'héllo'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = 'utf-8'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = 'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


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

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = b'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Signer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'key'

def test_case_0():
    var_0 = 'key'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 4/6 statements.
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
    var_0 = b'key1'
    var_1 = b'key2'
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
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

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
    var_3 = 'secret'
    var_4 = '-'
    var_5 = module_0.Signer(var_3, sep=var_4)
    var_6 = 'secret'
    var_7 = '_'
    var_8 = module_0.Signer(var_6, sep=var_7)
    var_9 = 'secret'
    var_10 = '='
    var_11 = module_0.Signer(var_9, sep=var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_all_parameters. Retrieved 6/12 statements.


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

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = b'|'
    var_5 = 'hmac'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'A'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Signer(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_algorithm. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_hmac_algorithm. Retrieved 4/5 statements.


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
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b''
    var_2 = module_0.Signer(var_0, sep=var_1)

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
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.HMACAlgorithm()
    var_2 = module_0.Signer(var_0, algorithm=var_1)
    var_3 = var_2.algorithm



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_invalid_sep. Retrieved 1/5 statements.


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

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

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

def test_case_0():
    var_0 = 'secret'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'invalidsig'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalidbase64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

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
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

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
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'django-concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = b'test'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.


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
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret_key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'none'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = 'secret'
    var_3 = module_0.Signer(var_2, digest_method=var_1)
    var_4 = var_3.digest_method
    var_5 = bool(var_3.digest_method == var_1)
    assert var_5 is True

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
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid!base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
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
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm

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
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

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



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test-value'
    var_3 = 'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
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
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

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
    var_1 = b'!'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'!'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = b'invalid-signature'
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
    var_6 = b'test-value'
    var_7 = var_3.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = '.'
    var_5 = module_0.Signer(var_2, var_3, var_4)
    var_6 = b'test-value'
    var_7 = var_5.get_signature(var_6)
    var_8 = var_5.verify_signature(var_6, var_7)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = b'!!!invalid-base64!!!'
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

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
    var_4 = 'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_derive_key_default_secret_key. Retrieved 3/4 statements.
# Partially parsed test_derive_key_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_django_concat. Retrieved 4/5 statements.
# Partially parsed test_derive_key_hmac. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_specific_secret. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_salt. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()

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
    var_3 = var_2.derive_key()
    assert var_3 == b'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'other'
    var_3 = var_1.derive_key(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'unknown'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
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
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'bytes-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-secret'])
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
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

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
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


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
    var_0 = b'key1'
    var_1 = b'key2'
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
    var_1 = b':'
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

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'not-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = '!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 4/6 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_algorithm. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_sep_in_base64_alphabet. Retrieved 1/6 statements.


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

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
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

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_invalid_separator. Retrieved 2/9 statements.


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
    var_7 = module_0._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

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
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

def test_case_0():
    var_0 = 'secret'
    var_1 = AssertionError(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
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
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
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
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

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
    var_1 = b'+'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = b'invalid_sig'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'bytes value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'string value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = b'test'
    var_6 = var_4.get_signature(var_5)
    var_7 = var_4.verify_signature(var_5, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = b'test'
    var_6 = module_0.Signer(var_0, var_3)
    var_7 = var_6.get_signature(var_5)
    var_8 = var_4.verify_signature(var_5, var_7)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'key2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #13
#--------------------------

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
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

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
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

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
    var_1 = b'|'
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
    var_1 = b'.'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


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
    var_7 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret-key'])
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
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'+'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = b'invalid_sig'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key-1'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'secret-key-2'
    var_4 = module_0.Signer(var_3, var_1)
    var_5 = b'test value'
    var_6 = var_2.get_signature(var_5)
    var_7 = var_4.verify_signature(var_5, var_6)
    assert var_7 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = b'test value'
    var_6 = [var_0]
    var_7 = module_0.Signer(var_6, var_3)
    var_8 = var_7.get_signature(var_5)
    var_9 = var_4.get_signature(var_5)
    var_10 = var_4.verify_signature(var_5, var_8)
    assert var_10 is True
    var_11 = var_4.verify_signature(var_5, var_9)
    assert var_11 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test bytes value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'test string value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b''
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = b'!!!invalid_base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = b''
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #16
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
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 'different-secret'
    var_5 = module_0.Signer(var_4)
    var_6 = var_5.verify_signature(var_2, var_3)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test_value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
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
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

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
    var_2 = b''
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'héllo'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True



