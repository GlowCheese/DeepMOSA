####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default_algorithm. Retrieved 5/10 statements.
# Partially parsed test_certificate_fingerprint_sha256. Retrieved 6/11 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/11 statements.
# Partially parsed test_certificate_fingerprint_format. Retrieved 4/9 statements.
# Partially parsed test_certificate_fingerprint_uppercase. Retrieved 2/4 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = ':'
    var_4 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 95
    var_4 = ':'
    var_5 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 59
    var_4 = ':'
    var_5 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown algorithm'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = var_0.certificate_fingerprint()
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_key_default. Retrieved 4/8 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 4/7 statements.
# Partially parsed test_api_key_with_custom_length. Retrieved 4/6 statements.
# Partially parsed test_api_key_with_prefix_and_length. Retrieved 5/8 statements.
# Partially parsed test_api_key_hex_format. Retrieved 4/8 statements.
# Partially parsed test_api_key_base64_format. Retrieved 5/7 statements.
# Partially parsed test_api_key_base64_with_prefix. Retrieved 5/8 statements.
# Partially parsed test_api_key_empty_prefix. Retrieved 4/6 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 64
    var_3 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 67

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 16
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'api_'
    var_2 = 20
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 44

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'hex'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'base64'
    var_2 = 32
    var_3 = var_0.api_key(length=var_2, fmt=var_1)
    var_4 = len(var_3)
    assert var_4 == 32

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 'base64'
    var_3 = 24
    var_4 = var_0.api_key(var_1, var_3, var_2)

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown format'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = ''
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 64

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = var_0.api_key()
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_api_key_default. Retrieved 3/5 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 4/7 statements.
# Partially parsed test_api_key_custom_length. Retrieved 4/6 statements.
# Partially parsed test_api_key_with_prefix_and_length. Retrieved 5/8 statements.
# Partially parsed test_api_key_base64_format. Retrieved 4/6 statements.
# Partially parsed test_api_key_base64_with_prefix. Retrieved 4/7 statements.
# Partially parsed test_api_key_hex_format. Retrieved 4/6 statements.
# Partially parsed test_api_key_empty_prefix. Retrieved 4/6 statements.
# Partially parsed test_api_key_all_parameters. Retrieved 5/8 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 64

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 67

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 16
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 24
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 51

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'base64'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 <= 32)
    assert var_4 is True

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'api_'
    var_2 = 'base64'
    var_3 = var_0.api_key(var_1, fmt=var_2)

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'hex'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 64

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown format'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = ''
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 64

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'test_'
    var_2 = 20
    var_3 = 'hex'
    var_4 = var_0.api_key(var_1, var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_certificate_fingerprint_sha256_default. Retrieved 5/10 statements.
# Partially parsed test_certificate_fingerprint_sha256_explicit. Retrieved 6/11 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/11 statements.
# Partially parsed test_certificate_fingerprint_format. Retrieved 4/9 statements.
# Partially parsed test_certificate_fingerprint_uppercase. Retrieved 2/4 statements.
# Partially parsed test_certificate_fingerprint_sha1_format. Retrieved 5/10 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = ':'
    var_4 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 95
    var_4 = ':'
    var_5 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 59
    var_4 = ':'
    var_5 = '0123456789ABCDEF:'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown algorithm'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2



