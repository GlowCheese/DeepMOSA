####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import re as module_1

import mimesis.providers.cryptographic as module_0


def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    assert var_0 == 2
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = ':'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = '0123456789ABCDEF'
    var_7 = all(var_3)
    var_8 = 'sha1'
    var_9 = var_1.certificate_fingerprint(var_8)
    var_10 = module_1.split(var_3)
    var_11 = len(var_10)
    assert var_11 == 20
    var_12 = '0123456789ABCDEF'
    var_13 = all(var_3)
    var_14 = 'md5'
    var_15 = var_1.certificate_fingerprint(var_14)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 64
    var_3 = 'sk_'
    var_4 = var_0.api_key(var_3)
    var_5 = len(var_4)
    var_6 = 'base64'
    var_7 = var_0.api_key(fmt=var_6)
    var_8 = len(var_7)
    var_9 = 'pk_'
    var_10 = var_0.api_key(var_9, fmt=var_6)
    var_11 = 'invalid'
    var_12 = var_0.api_key(fmt=var_11)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = 'sk_'
    var_4 = var_0.api_key(var_3)
    var_5 = 'base64'
    var_6 = var_0.api_key(fmt=var_5)
    var_7 = 16
    var_8 = var_0.api_key(length=var_7)
    var_9 = len(var_8)
    assert var_9 == 16
    var_10 = 'pk_'
    var_11 = var_0.api_key(var_10, fmt=var_5)
    var_12 = 'invalid'
    var_13 = var_0.api_key(fmt=var_12)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    assert var_0 == 2
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = ':'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 16
    var_7 = 'sha1'
    var_8 = var_1.certificate_fingerprint(var_7)
    var_9 = module_1.split(var_6)
    var_10 = len(var_9)
    assert var_10 == 20
    var_11 = 16
    var_12 = 'md5'
    var_13 = var_1.certificate_fingerprint(var_12)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = len(var_2)
    assert var_3 == 95
    var_4 = '0123456789ABCDEF:'
    var_5 = ':'
    var_6 = 'sha1'
    var_7 = var_1.certificate_fingerprint(var_6)
    var_8 = len(var_7)
    assert var_8 == 59
    var_9 = 'invalid'
    var_10 = var_1.certificate_fingerprint(var_9)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    assert var_0 == 2
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = ':'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 16
    var_7 = 'sha1'
    var_8 = var_1.certificate_fingerprint(var_7)
    var_9 = module_1.split(var_3)
    var_10 = len(var_9)
    assert var_10 == 20
    var_11 = 16
    var_12 = 'md5'
    var_13 = var_1.certificate_fingerprint(var_12)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 64
    var_3 = '0123456789abcdef'
    var_4 = 'sk_'
    var_5 = var_0.api_key(var_4)
    var_6 = len(var_5)
    var_7 = 'base64'
    var_8 = var_0.api_key(fmt=var_7)
    var_9 = len(var_8)
    var_10 = 'pk_'
    var_11 = var_0.api_key(var_10, fmt=var_7)
    var_12 = len(var_11)
    var_13 = 16
    var_14 = var_0.api_key(length=var_13)
    var_15 = len(var_14)
    assert var_15 == 32
    var_16 = 'invalid'
    var_17 = var_0.api_key(fmt=var_16)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'Test certificate_fingerprint method.'
    assert var_0 == 2
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = ':'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = '0123456789ABCDEF'
    var_7 = all(var_3)
    var_8 = 'sha1'
    var_9 = var_1.certificate_fingerprint(var_8)
    var_10 = module_1.split(var_3)
    var_11 = len(var_10)
    assert var_11 == 20
    var_12 = '0123456789ABCDEF'
    var_13 = all(var_3)
    var_14 = 'md5'
    var_15 = var_1.certificate_fingerprint(var_14)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = len(var_2)
    assert var_3 == 95
    var_4 = '0123456789ABCDEF:'



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'Test method certificate_fingerprint of class Cryptographic.'
    var_1 = module_0.Cryptographic()
    var_2 = var_1.certificate_fingerprint()
    var_3 = ':'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 2
    var_7 = '0123456789ABCDEF'
    var_8 = 'sha1'
    var_9 = var_1.certificate_fingerprint(var_8)
    var_10 = module_1.split(var_3)
    var_11 = len(var_10)
    assert var_11 == 20
    var_12 = 'md5'
    var_13 = var_1.certificate_fingerprint(var_12)



