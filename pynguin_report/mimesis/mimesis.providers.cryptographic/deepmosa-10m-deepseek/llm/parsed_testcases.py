####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_api_key_default. Retrieved 4/7 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 7/11 statements.
# Partially parsed test_api_key_with_length. Retrieved 5/8 statements.
# Partially parsed test_api_key_with_prefix_and_length. Retrieved 8/12 statements.
# Partially parsed test_api_key_format_hex. Retrieved 5/8 statements.
# Partially parsed test_api_key_format_base64. Retrieved 5/8 statements.
# Partially parsed test_api_key_format_base64_with_prefix. Retrieved 8/12 statements.
# Partially parsed test_api_key_format_base64_with_length. Retrieved 6/9 statements.
# Partially parsed test_api_key_empty_prefix. Retrieved 5/8 statements.
# Partially parsed test_api_key_length_zero. Retrieved 4/5 statements.
# Partially parsed test_api_key_length_zero_with_prefix. Retrieved 4/5 statements.
# Partially parsed test_api_key_length_odd. Retrieved 5/8 statements.
# Partially parsed test_api_key_length_odd_with_prefix. Retrieved 8/12 statements.


import mimesis.providers.cryptographic as module_0


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 35
    var_4 = 3
    var_5 = var_2[var_4:]
    var_6 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 64
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 64
    var_4 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'api_'
    var_2 = 48
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 52
    var_5 = 4
    var_6 = var_3[var_5:]
    var_7 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'hex'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'base64'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 'base64'
    var_3 = var_0.api_key(var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 35
    var_5 = 3
    var_6 = var_3[var_5:]
    var_7 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 64
    var_2 = 'base64'
    var_3 = var_0.api_key(length=var_1, fmt=var_2)
    var_4 = len(var_3)
    assert var_4 == 64
    var_5 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = ''
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 0
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 0


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'test_'
    var_2 = 0
    var_3 = var_0.api_key(var_1, var_2)
    assert var_3 == 'test_'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 31
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 31
    var_4 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'odd_'
    var_2 = 31
    var_3 = var_0.api_key(var_1, var_2)
    var_4 = len(var_3)
    assert var_4 == 35
    var_5 = 4
    var_6 = var_3[var_5:]
    var_7 = '0123456789abcdef'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_certificate_fingerprint_default_algorithm. Retrieved 5/12 statements.
# Partially parsed test_certificate_fingerprint_sha256. Retrieved 6/13 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/13 statements.
# Partially parsed test_certificate_fingerprint_format_consistency. Retrieved 5/13 statements.



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2
    var_4 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = var_0.certificate_fingerprint()
    var_3 = bool(var_1 != var_2)
    assert var_3 is True
    var_4 = ':'
    var_5 = 2



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_api_key_default. Retrieved 4/7 statements.
# Partially parsed test_api_key_with_prefix. Retrieved 7/11 statements.
# Partially parsed test_api_key_with_custom_length. Retrieved 5/8 statements.
# Partially parsed test_api_key_base64_format. Retrieved 11/14 statements.
# Partially parsed test_api_key_base64_with_prefix. Retrieved 15/19 statements.



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 64
    var_3 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sk_'
    var_2 = var_0.api_key(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 == 64 + 3)
    assert var_4 is True
    var_5 = 3
    var_6 = var_2[var_5:]
    var_7 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 16
    var_2 = var_0.api_key(length=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = '0123456789abcdef'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'base64'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = len(var_2)
    assert var_3 == 32
    var_4 = '='
    var_5 = 4
    var_6 = len(var_2)
    var_7 = var_6 % var_5
    var_8 = var_5 - var_7
    var_9 = var_4 * var_8
    var_10 = var_2 + var_9
    var_11 = bool(True)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'pk_'
    var_2 = 'base64'
    var_3 = var_0.api_key(var_1, fmt=var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 == 32 + 3)
    assert var_5 is True
    var_6 = 3
    var_7 = var_3[var_6:]
    var_8 = '='
    var_9 = 4
    var_10 = var_3[var_6:]
    var_11 = len(var_10)
    var_12 = var_11 % var_9
    var_13 = var_9 - var_12
    var_14 = var_8 * var_13
    var_15 = var_7 + var_14
    var_16 = bool(True)
    assert var_16 is True
    var_17 = bool(False)
    assert var_17 is True


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'invalid'
    var_2 = var_0.api_key(fmt=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown format'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_certificate_fingerprint_default_algorithm. Retrieved 5/11 statements.
# Partially parsed test_certificate_fingerprint_sha256. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_sha1. Retrieved 6/12 statements.
# Partially parsed test_certificate_fingerprint_uppercase_output. Retrieved 2/3 statements.
# Partially parsed test_certificate_fingerprint_colon_separated. Retrieved 3/4 statements.
# Partially parsed test_certificate_fingerprint_sha1_colon_separated. Retrieved 4/5 statements.



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = 2
    var_4 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha256'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = 2
    var_5 = '0123456789ABCDEF'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'md5'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown algorithm'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = ':'
    var_3 = bool(':' in var_1)
    assert var_3 is True
    var_4 = ':'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = 'sha1'
    var_2 = var_0.certificate_fingerprint(var_1)
    var_3 = ':'
    var_4 = bool(':' in var_2)
    assert var_4 is True
    var_5 = ':'


def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = var_0.certificate_fingerprint()
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



