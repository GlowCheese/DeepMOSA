####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = '0123456789abcdef'
    var_4 = 'sk_'
    var_5 = var_0.api_key(var_4)
    var_6 = len(var_5)
    assert var_6 == 35
    var_7 = 3
    var_8 = var_5[var_7:]
    var_9 = 'base64'
    var_10 = var_0.api_key(fmt=var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    var_13 = 'pk_'
    var_14 = var_0.api_key(var_13, fmt=var_9)
    var_15 = len(var_14)
    assert var_15 == 35
    var_16 = var_14[var_7:]
    var_17 = 'invalid'
    var_18 = var_0.api_key(fmt=var_17)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    var_3 = '0123456789ABCDEF:'
    var_4 = 'sha1'
    var_5 = var_0.certificate_fingerprint(var_4)
    var_6 = len(var_5)
    var_7 = 'invalid'
    var_8 = var_0.certificate_fingerprint(var_7)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = 'sk_'
    var_4 = var_0.api_key(var_3)
    var_5 = len(var_4)
    assert var_5 == 35
    var_6 = 'base64'
    var_7 = var_0.api_key(fmt=var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 'pk_'
    var_10 = var_0.api_key(var_9, fmt=var_6)
    var_11 = len(var_10)
    assert var_11 == 35
    var_12 = 16
    var_13 = var_0.api_key(length=var_12)
    var_14 = len(var_13)
    assert var_14 == 16
    var_15 = 'api_'
    var_16 = var_0.api_key(var_15, var_12)
    var_17 = len(var_16)
    assert var_17 == 20
    var_18 = var_0.api_key(length=var_12, fmt=var_6)
    var_19 = len(var_18)
    assert var_19 == 16
    var_20 = var_0.api_key(var_15, var_12, var_6)
    var_21 = len(var_20)
    assert var_21 == 20
    var_22 = 'invalid_format'
    var_23 = var_0.api_key(fmt=var_22)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'
    var_5 = 'sha1'
    var_6 = var_0.certificate_fingerprint(var_5)
    var_7 = len(var_6)
    assert var_7 == 119
    var_8 = 'invalid'
    var_9 = var_0.certificate_fingerprint(var_8)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = '0123456789abcdef'
    var_4 = 'sk_'
    var_5 = var_0.api_key(var_4)
    var_6 = len(var_5)
    assert var_6 == 35
    var_7 = 3
    var_8 = var_5[var_7:]
    var_9 = 'base64'
    var_10 = var_0.api_key(fmt=var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_='
    var_13 = 'pk_'
    var_14 = var_0.api_key(var_13, fmt=var_9)
    var_15 = len(var_14)
    assert var_15 == 35
    var_16 = var_14[var_7:]
    var_17 = 16
    var_18 = 'hex'
    var_19 = var_0.api_key(length=var_17, fmt=var_18)
    var_20 = len(var_19)
    assert var_20 == 16
    var_21 = 'invalid'
    var_22 = var_0.api_key(fmt=var_21)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'
    var_5 = 'sha1'
    var_6 = var_0.certificate_fingerprint(var_5)
    var_7 = len(var_6)
    assert var_7 == 119
    var_8 = 'invalid'
    var_9 = var_0.certificate_fingerprint(var_8)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = 'sk_'
    var_4 = var_0.api_key(var_3)
    var_5 = len(var_4)
    assert var_5 == 35
    var_6 = 64
    var_7 = var_0.api_key(length=var_6)
    var_8 = len(var_7)
    assert var_8 == 64
    var_9 = 'base64'
    var_10 = var_0.api_key(fmt=var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = 'pk_'
    var_13 = var_0.api_key(var_12, fmt=var_9)
    var_14 = len(var_13)
    assert var_14 == 35
    var_15 = 'invalid'
    var_16 = var_0.api_key(fmt=var_15)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.api_key()
    var_2 = len(var_1)
    assert var_2 == 32
    var_3 = 'sk_'
    var_4 = var_0.api_key(var_3)
    var_5 = len(var_4)
    assert var_5 == 35
    var_6 = 16
    var_7 = var_0.api_key(length=var_6)
    var_8 = len(var_7)
    assert var_8 == 16
    var_9 = 'base64'
    var_10 = var_0.api_key(fmt=var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = 'pk_'
    var_13 = var_0.api_key(var_12, fmt=var_9)
    var_14 = len(var_13)
    assert var_14 == 35
    var_15 = 'invalid'
    var_16 = var_0.api_key(fmt=var_15)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'
    var_5 = 'sha1'
    var_6 = var_0.certificate_fingerprint(var_5)
    var_7 = len(var_6)
    assert var_7 == 119
    var_8 = 'invalid'
    var_9 = var_0.certificate_fingerprint(var_8)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.cryptographic as module_0

def test_case_0():
    var_0 = module_0.Cryptographic()
    var_1 = var_0.certificate_fingerprint()
    var_2 = len(var_1)
    assert var_2 == 95
    var_3 = '0123456789ABCDEF:'
    var_4 = ':'
    var_5 = 'sha1'
    var_6 = var_0.certificate_fingerprint(var_5)
    var_7 = len(var_6)
    assert var_7 == 119
    var_8 = 'invalid'
    var_9 = var_0.certificate_fingerprint(var_8)



