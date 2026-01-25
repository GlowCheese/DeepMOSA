####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default_algorithm. Retrieved 5/11 statements.
# Partially parsed test_certificate_fingerprint_sha256. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_uppercase. Retrieved 2/3 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2
    var_4 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown algorithm'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_key_default. Retrieved 3/4 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 7/9 statements.
# Partially parsed test_api_key_with_custom_length. Retrieved 4/5 statements.
# Partially parsed test_api_key_with_prefix_and_length. Retrieved 7/9 statements.
# Partially parsed test_api_key_format_hex. Retrieved 4/7 statements.
# Partially parsed test_api_key_format_base64. Retrieved 4/7 statements.
# Partially parsed test_api_key_format_base64_with_prefix. Retrieved 7/11 statements.


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
    var_4 = 32
    var_5 = len(var_1)
    var_6 = var_4 + var_5
    var_7 = bool(var_3 == var_6)
    assert var_7 is True

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 64
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 64

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'api_'
    var_2 = 48
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    var_5 = len(var_1)
    var_6 = var_2 + var_5
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

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
    var_3 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 'base64'
    var_3 = var_0.api_key(var_1, fmt=var_2)
    var_4 = len(var_1)
    var_5 = var_3[var_4:]
    var_6 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown format'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_certificate_fingerprint_default_algorithm. Retrieved 5/11 statements.
# Partially parsed test_certificate_fingerprint_sha256. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_uppercase_output. Retrieved 2/3 statements.
# Partially parsed test_certificate_fingerprint_colon_separated. Retrieved 3/4 statements.
# Partially parsed test_certificate_fingerprint_sha1_colon_separated. Retrieved 4/5 statements.
# Partially parsed test_certificate_fingerprint_hex_characters_only. Retrieved 5/8 statements.


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2
    var_4 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'

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

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = bool(':' in var_1)
    assert var_3 is True
    var_4 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = bool(':' in var_2)
    assert var_4 is True
    var_5 = ':'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = ''
    var_4 = '0123456789ABCDEF'

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = var_0.certificate_fingerprint()
    var_3 = bool(var_1 != var_2)
    assert var_3 is True

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = var_0.certificate_fingerprint(var_1)
    var_4 = bool(var_2 != var_3)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_key_default. Retrieved 3/4 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 4/6 statements.
# Partially parsed test_api_key_with_length. Retrieved 4/5 statements.
# Partially parsed test_api_key_with_prefix_and_length. Retrieved 5/7 statements.
# Partially parsed test_api_key_format_hex. Retrieved 4/7 statements.
# Partially parsed test_api_key_format_base64. Retrieved 4/5 statements.
# Partially parsed test_api_key_format_base64_with_prefix. Retrieved 5/7 statements.
# Partially parsed test_api_key_format_base64_with_length. Retrieved 5/6 statements.


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
    var_1 = 'api_'
    var_2 = 24
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 52

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
    var_3 = len(var_2)
    assert var_3 == 32

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 'base64'
    var_3 = var_0.api_key(var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 35

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 48
    var_2 = 'base64'
    var_3 = var_0.api_key(length=var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 48

import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown format'



