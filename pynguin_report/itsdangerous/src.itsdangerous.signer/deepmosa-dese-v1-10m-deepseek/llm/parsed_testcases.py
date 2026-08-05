####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_derive_key_concat. Retrieved 8/9 statements.
# Partially parsed test_derive_key_django_concat. Retrieved 12/13 statements.
# Partially parsed test_derive_key_hmac. Retrieved 10/11 statements.


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'concat'
    var_3 = 'sha256'
    var_4 = module_0.Signer(var_0, var_1, key_derivation=var_2, digest_method=var_3)
    var_5 = var_4.derive_key()
    var_6 = b'saltsecret-key'
    var_7 = module_1.digest()

import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'django-concat'
    var_3 = 'sha256'
    var_4 = module_0.Signer(var_0, var_1, key_derivation=var_2, digest_method=var_3)
    var_5 = var_4.derive_key()
    var_6 = b'salt'
    var_7 = b'signer'
    var_8 = var_6 + var_7
    var_9 = b'secret-key'
    var_10 = var_8 + var_9
    var_11 = module_1.digest()

import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = 'hmac'
    var_3 = 'sha256'
    var_4 = module_0.Signer(var_0, var_1, key_derivation=var_2, digest_method=var_3)
    var_5 = var_4.derive_key()
    var_6 = b'secret-key'
    var_7 = module_1.new(var_6, digestmod=var_3)
    var_8 = b'salt'
    var_9 = module_1.digest()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    assert var_3 == b'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'ignored'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = 'custom-secret'
    var_4 = var_2.derive_key(var_3)
    assert var_4 == b'custom-secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    assert var_3 == b'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'invalid'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_string_value. Retrieved 4/6 statements.


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
    var_2 = b'noseparator'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = 'value'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.Signer(var_0, sep=var_1)
    var_3 = 'value|with|separator'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'value|with|separator'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_signer_constructor_default_secret_key_str. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_custom_algorithm. Retrieved 1/6 statements.
# Partially parsed test_signer_constructor_sep_in_base64_alphabet. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #4
#--------------------------




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
    var_3 = b'invalid_sig'
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
    var_2 = b'bytes value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'string value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
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
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = 'not bytes'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #5
#--------------------------




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
    var_0 = 'key1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'key2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

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
    var_2 = 'test string'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = None
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'none'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'.'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_signer_init_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_init_with_bytes_secret_key. Retrieved 3/4 statements.
# Partially parsed test_signer_init_with_list_of_strings. Retrieved 5/6 statements.
# Partially parsed test_signer_init_with_list_of_bytes. Retrieved 5/6 statements.
# Partially parsed test_signer_init_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_init_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = var_3.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'key'

def test_case_0():
    var_0 = 'key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 4/6 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.digest_method
    var_3 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_verify_signature_string_signature. Retrieved 5/7 statements.


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
    var_2 = None
    var_3 = b'some_signature'
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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
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
    var_2 = b'test value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/6 statements.
# Partially parsed test_signer_constructor_algorithm_none. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-bytes'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = b'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Signer(var_3)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, digest_method=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, algorithm=var_1)
    var_3 = var_2.algorithm



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_with_sep_in_base64_alphabet. Retrieved 1/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = 'invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'bytes-secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/8 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'='
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_verify_signature_with_bytes_value_and_text_sig. Retrieved 4/6 statements.
# Partially parsed test_verify_signature_with_multiple_secret_keys_oldest. Retrieved 9/10 statements.
# Partially parsed test_verify_signature_with_multiple_secret_keys_only_oldest_valid. Retrieved 8/9 statements.
# Partially parsed test_verify_signature_with_multiple_secret_keys_only_newest_valid. Retrieved 8/9 statements.
# Partially parsed test_verify_signature_with_none_secret_key. Retrieved 6/7 statements.


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
    var_2 = 'test'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = var_1.get_signature(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = var_3.get_signature(var_4)
    var_6 = b'old_key'
    var_7 = b'new_key'
    var_8 = var_3.verify_signature(var_4, var_5)
    assert var_8 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = var_3.get_signature(var_4)
    var_6 = b'old_key'
    var_7 = var_3.verify_signature(var_4, var_5)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)
    var_4 = b'test'
    var_5 = var_3.get_signature(var_4)
    var_6 = b'new_key'
    var_7 = var_3.verify_signature(var_4, var_5)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test'
    var_3 = var_1.get_signature(var_2)
    var_4 = b'different_key'
    var_5 = var_1.verify_signature(var_2, var_3)
    assert var_5 is False

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
    var_2 = b'test'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_4 = b'invalid_signature'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

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
    var_4 = var_2.get_signature(var_3)
    var_5 = 'different-secret-key'
    var_6 = module_0.Signer(var_5, var_1)
    var_7 = var_6.verify_signature(var_3, var_4)
    assert var_7 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = 'salt2'
    var_6 = module_0.Signer(var_0, var_5)
    var_7 = var_6.verify_signature(var_3, var_4)
    assert var_7 is False

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
    var_4 = None
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
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = b'test value'
    var_6 = var_4.get_signature(var_5)
    var_7 = var_4.verify_signature(var_5, var_6)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = b'test value'
    var_6 = var_4.get_signature(var_5)
    var_7 = module_0.Signer(var_0, var_3)
    var_8 = var_7.get_signature(var_5)
    var_9 = var_4.verify_signature(var_5, var_8)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = None
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_derive_key_with_none_secret_key_uses_last_secret_key. Retrieved 3/4 statements.
# Partially parsed test_derive_key_with_custom_secret_key. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_concat_method. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_django_concat_method. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_hmac_method. Retrieved 4/5 statements.
# Partially parsed test_derive_key_with_string_secret_key. Retrieved 3/4 statements.
# Partially parsed test_derive_key_with_custom_string_secret_key. Retrieved 4/5 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'custom'
    var_3 = var_1.derive_key(var_2)

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
    var_0 = b'secret'
    var_1 = 'unknown'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = 'custom'
    var_3 = var_1.derive_key(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_verify_signature_with_bytes_value_and_str_signature. Retrieved 6/8 statements.


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
    var_4 = b'invalid_signature'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = b'!!!not_base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False

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
    var_6 = var_4.get_signature(var_5)
    var_7 = module_0.Signer(var_0, var_3)
    var_8 = var_7.get_signature(var_5)
    var_9 = var_4.verify_signature(var_5, var_8)
    assert var_9 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = 'ascii'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'salt2'
    var_4 = module_0.Signer(var_0, var_3)
    var_5 = b'test value'
    var_6 = var_2.get_signature(var_5)
    var_7 = var_4.verify_signature(var_5, var_6)
    assert var_7 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'A'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.Signer(var_3)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '-'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = b'invalid_base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret_bytes'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #9
#--------------------------




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
    var_3 = b''
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'!!!invalid_base64!!!'
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
    var_0 = 'secret-key-1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'secret-key-2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'-'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #13
#--------------------------




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
    var_2 = b'bytes value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'string value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = b'!!!invalid base64'
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is False

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
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = 'new-key'
    var_5 = [var_0, var_4]
    var_6 = module_0.Signer(var_5)
    var_7 = var_6.verify_signature(var_2, var_3)
    assert var_7 is True

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = module_0.Signer(var_0)
    var_2 = 'key2'
    var_3 = module_0.Signer(var_2)
    var_4 = b'test value'
    var_5 = var_1.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_0.Signer(var_1, algorithm=var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'+'
    var_2 = module_0.Signer(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_signer_constructor_digest_method_custom. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_algorithm_default. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_algorithm_custom. Retrieved 1/6 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'my-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'my-salt'
    var_2 = b'|'
    var_3 = module_0.Signer(var_0, var_1, var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'my-salt'
    var_2 = b'a'
    var_3 = module_0.Signer(var_0, var_1, var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'my-secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = 'invalid-base64!!!'
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'value'
    var_3 = b'!!!invalid-base64!!!'
    var_4 = var_1.verify_signature(var_2, var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_signer_constructor_default_parameters. Retrieved 3/4 statements.
# Partially parsed test_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.
# Partially parsed test_signer_constructor_sep_in_base64_alphabet_raises. Retrieved 1/5 statements.
# Partially parsed test_signer_constructor_sep_bytes_in_base64_alphabet_raises. Retrieved 1/4 statements.


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.Signer(var_2)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.Signer(var_0, sep=var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Signer(var_0, var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



