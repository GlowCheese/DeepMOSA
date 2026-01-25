####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default. Retrieved 4/10 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 5/11 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_key_default. Retrieved 3/4 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 4/5 statements.
# Partially parsed test_api_key_hex_format. Retrieved 4/6 statements.
# Partially parsed test_api_key_base64_format. Retrieved 4/6 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 35

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
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_='

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 16
    var_2 = 'hex'
    var_3 = var_0.api_key(length=var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 16

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default. Retrieved 5/9 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/10 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 59
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 59
    var_4 = '0123456789ABCDEF:'
    var_5 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default. Retrieved 5/9 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/10 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 59
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = len(var_2)
    assert var_3 == 59
    var_4 = '0123456789ABCDEF:'
    var_5 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_key_default. Retrieved 4/7 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 7/10 statements.
# Partially parsed test_api_key_hex_format. Retrieved 5/8 statements.
# Partially parsed test_api_key_base64_format. Retrieved 5/8 statements.
# Partially parsed test_api_key_custom_length. Retrieved 6/9 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 35
    var_4 = 3
    var_5 = var_2[var_4:]
    var_6 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'hex'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'base64'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_='

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 16
    var_2 = 'hex'
    var_3 = var_0.api_key(length=var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 16
    var_5 = '0123456789abcdef'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True



