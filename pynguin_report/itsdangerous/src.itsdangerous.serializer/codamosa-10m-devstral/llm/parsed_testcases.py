####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'HELLO'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'olleh'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"custom": "test"}'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'test'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'{"key": "value"}'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #3
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_0, var_9)
    var_11 = var_10.dumps(var_4)
    var_12 = var_10.loads(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.dumps(var_4)
    var_18 = var_16.loads(var_17)
    var_19 = 42
    var_20 = var_1.dumps(var_19)
    var_21 = var_1.dumps(var_19)
    var_22 = 'string'
    var_23 = var_1.dumps(var_22)
    var_24 = var_1.dumps(var_22)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_1.dumps(var_28)
    var_30 = [var_25, var_26, var_27]
    var_31 = var_1.dumps(var_30)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.dumps(var_6)
    var_8 = var_1.loads(var_7)
    var_9 = module_1.dumps(var_6)
    var_10 = module_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = module_0.Serializer(var_0, var_11)
    var_13 = var_12.dumps(var_6)
    var_14 = var_12.loads(var_13)
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Serializer(var_17)
    var_19 = var_18.dumps(var_6)
    var_20 = var_18.loads(var_19)
    var_21 = 'string'
    var_22 = var_1.dumps(var_21)
    var_23 = var_1.dumps(var_21)
    var_24 = var_1.loads(var_23)
    var_25 = 123
    var_26 = var_1.dumps(var_25)
    var_27 = var_1.dumps(var_25)
    var_28 = var_1.loads(var_27)
    var_29 = 3
    var_30 = [var_4, var_5, var_29]
    var_31 = var_1.dumps(var_30)
    var_32 = [var_4, var_5, var_29]
    var_33 = var_1.dumps(var_32)
    var_34 = var_1.loads(var_33)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'42'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 42
    var_6 = b'hello'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'hello'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'anything'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #7
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test_payload'
    var_2 = b'test_payload'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test_payload'



# Parsed testcases at query #8
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded_test'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'loaded_test'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 'custom-salt'
    var_8 = module_0.Serializer(var_0, var_7)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'digest_method'
    var_13 = 'sha256'
    var_14 = {var_12: var_13}
    var_15 = 'sha512'
    var_16 = {var_12: var_15}
    var_17 = var_8.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = var_18[var_5]
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Serializer(var_23)
    var_25 = var_24.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = [var_21, var_22]
    var_29 = {var_12: var_13}
    var_30 = [var_29]
    var_31 = module_0.Serializer(var_28, fallback_signers=var_30)
    var_32 = var_31.iter_unsigners()
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2



# Parsed testcases at query #10
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'test_payload'
    var_3 = module_0.loads(var_2)
    var_4 = 'custom_payload'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'custom_test_data'
    var_6 = b'test_bytes'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'bytes_test_bytes'
    var_8 = b'test_data'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'loaded: test_data'
    var_6 = b'test_bytes'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == "bytes_loaded: b'test_bytes'"
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'test'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #13
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = 'test'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #14
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #15
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)
    var_5 = 'test_string'
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = 'test_string'
    var_9 = module_0.dumps(var_8)
    var_10 = module_0.loads(var_9)



# Parsed testcases at query #16
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)



# Parsed testcases at query #17
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'abc'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'abcabc'
    var_4 = 'test'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_payload'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'custom_test_payload'
    var_6 = b'test_payload'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'custom_test_payload'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid_json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'test_payload'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #19
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded: test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded: test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded: test'



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'HELLO'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'olleh'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = module_1.dumps(var_4)
    var_10 = module_1.loads(var_9)
    var_11 = module_1.dumps(var_4)
    var_12 = module_1.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = module_0.Serializer(var_0, var_13)
    var_15 = var_14.dumps(var_4)
    var_16 = var_14.loads(var_15)
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Serializer(var_19)
    var_21 = var_20.dumps(var_4)
    var_22 = var_20.loads(var_21)



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded: test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded: test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded: test'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = module_1.dumps(var_4)
    var_10 = module_1.loads(var_9)
    var_11 = module_1.dumps(var_4)
    var_12 = module_1.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = module_0.Serializer(var_0, var_13)
    var_15 = var_14.dumps(var_4)
    var_16 = var_14.loads(var_15)
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Serializer(var_19)
    var_21 = var_20.dumps(var_4)
    var_22 = var_20.loads(var_21)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_0, var_9)
    var_11 = var_10.dumps(var_4)
    var_12 = var_10.loads(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.dumps(var_4)
    var_18 = var_16.loads(var_17)



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test_payload'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'loaded: test_data'
    var_6 = b'test_data'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'loaded: test_data'
    var_8 = b'test_data'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #27
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_0, var_9)
    var_11 = var_10.dumps(var_4)
    var_12 = var_10.loads(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.dumps(var_4)
    var_18 = var_16.loads(var_17)
    var_19 = None
    var_20 = var_1.dumps(var_19)
    var_21 = 123
    var_22 = var_1.dumps(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = var_1.dumps(var_26)



# Parsed testcases at query #29
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'key': 'value'}"



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'custom_test_data'
    var_6 = b'test_data'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'bytes_test_data'
    var_8 = b'test_data'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    var_6 = {var_0: var_1}
    var_7 = module_0.dumps(var_6)
    assert var_7 == b"{'key': 'value'}"
    var_8 = {var_0: var_1}
    var_9 = module_0.dumps(var_8)
    var_10 = {var_0: var_1}
    var_11 = module_0.dumps(var_10)
    assert var_11 == '{"key": "value"}'
    var_12 = {var_0: var_1}
    var_13 = module_0.dumps(var_12)



# Parsed testcases at query #32
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'key': 'value'}"



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 'custom-salt'
    var_7 = module_0.Serializer(var_0, var_6)
    var_8 = var_7.dumps(var_4)
    var_9 = module_1.dumps(var_4)
    var_10 = module_1.dumps(var_4)
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Serializer(var_13)
    var_15 = var_14.dumps(var_4)



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 'hello'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'HELLO'



# Parsed testcases at query #35
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.dumps(var_7)
    assert var_8 == '[1, 2, 3]'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = [var_4, var_5, var_6]
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_0.dumps(var_13)
    assert var_14 == '{"a": {"b": [1, 2, 3]}}'
    var_15 = 'hello'
    var_16 = module_0.dumps(var_15)
    assert var_16 == '"hello"'
    var_17 = 42
    var_18 = module_0.dumps(var_17)
    assert var_18 == '42'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 'hello'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'HELLO'



# Parsed testcases at query #2
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = module_0.loads(var_0)
    var_5 = module_0.loads(var_2)



# Parsed testcases at query #3
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded: test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded: test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded: test'
    var_5 = module_0.loads(var_2)
    assert var_5 == 'loaded: test'



# Parsed testcases at query #4
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded_test'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'loaded_test'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 'custom-salt'
    var_8 = module_0.Serializer(var_0, var_7)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = 'digest_method'
    var_12 = 'sha256'
    var_13 = {var_11: var_12}
    var_14 = [var_13]
    var_15 = module_0.Serializer(var_0, fallback_signers=var_14)
    var_16 = var_15.iter_unsigners()
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_11: var_12}
    var_20 = var_15.iter_unsigners()
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = var_15.iter_unsigners()
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 1
    var_27 = var_24[var_26]
    var_28 = {var_11: var_12}
    var_29 = 'sha512'
    var_30 = {var_11: var_29}
    var_31 = var_15.iter_unsigners()
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 3
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = module_0.Serializer(var_0)
    var_42 = var_41.iter_unsigners(var_7)
    var_43 = list(var_42)



# Parsed testcases at query #6
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded: test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded: test'



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 'custom-salt'
    var_8 = module_0.Serializer(var_0, var_7)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = module_0.Serializer(var_0)
    var_13 = 'another-salt'
    var_14 = var_12.iter_unsigners(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'key1'
    var_18 = 'key2'
    var_19 = 'key3'
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.Serializer(var_20)
    var_22 = var_21.iter_unsigners()
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'digest_method'
    var_26 = 'sha256'
    var_27 = {var_25: var_26}
    var_28 = [var_27]
    var_29 = module_0.Serializer(var_0, fallback_signers=var_28)
    var_30 = var_29.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'sha512'
    var_34 = {var_25: var_33}
    var_35 = var_29.iter_unsigners()
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = 1
    var_39 = var_36[var_38]
    var_40 = var_29.iter_unsigners()
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = var_41[var_38]
    var_44 = [var_17, var_18]
    var_45 = {var_25: var_26}
    var_46 = {var_25: var_33}
    var_47 = var_29.iter_unsigners()
    var_48 = list(var_47)
    var_49 = len(var_48)



# Parsed testcases at query #8
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #9
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = module_0.dumps(var_2)
    assert var_4 == b'{"key": "value"}'



# Parsed testcases at query #10
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.dumps(var_7)
    assert var_8 == '[1, 2, 3]'
    var_9 = 'test string'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"test string"'
    var_11 = 42
    var_12 = module_0.dumps(var_11)
    assert var_12 == '42'
    var_13 = True
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'true'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'



# Parsed testcases at query #11
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = module_0.loads(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'test_data'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'test_data'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.dumps(var_4)
    var_7 = var_1.dumps(var_4)



# Parsed testcases at query #13
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #14
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_0.loads(var_5)
    var_7 = 'hello'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'HELLO'
    var_9 = b'hello'
    var_10 = module_0.loads(var_9)
    assert var_10 == b'HELLO'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'loaded: test_data'
    var_6 = b'test_bytes'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == "bytes_loaded: b'test_bytes'"
    var_8 = b'test_data'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #16
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)



# Parsed testcases at query #17
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.dumps(var_7)
    assert var_8 == '[1, 2, 3]'
    var_9 = 'test'
    var_10 = module_0.dumps(var_9)
    assert var_10 == 'test'



# Parsed testcases at query #18
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded_test'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'loaded_test'



# Parsed testcases at query #19
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded_test'



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.dumps(var_7)
    assert var_8 == '[1, 2, 3]'
    var_9 = 'test string'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"test string"'
    var_11 = 42
    var_12 = module_0.dumps(var_11)
    assert var_12 == '42'
    var_13 = True
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'true'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'



# Parsed testcases at query #21
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #23
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #24
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'key': 'value'}"



# Parsed testcases at query #26
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)



# Parsed testcases at query #27
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'{"key": "value"}'
    var_2 = b'custom_payload'
    var_3 = b'{"key": "value"}'
    var_4 = b'invalid_payload'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'custom_data'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'custom_custom_data'
    var_6 = b'bytes_data'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'bytes_bytes_data'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid_json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'{"key": "value"}'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = module_1.dumps(var_4)
    var_8 = module_1.loads(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_0, var_9)
    var_11 = var_10.dumps(var_4)
    var_12 = var_10.loads(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.dumps(var_4)
    var_18 = var_16.loads(var_17)
    var_19 = None
    var_20 = 123
    var_21 = 45.67
    var_22 = 'string'
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_23, var_28: var_24}
    var_30 = True
    var_31 = False
    var_32 = [var_19, var_20, var_21, var_22, var_26, var_29, var_30, var_31]
    var_33 = 'value1'
    var_34 = {var_2: var_33}
    var_35 = 'value2'
    var_36 = {var_2: var_35}
    var_37 = var_1.dumps(var_34)
    var_38 = var_1.dumps(var_36)



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #32
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #33
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'key': 'value'}"



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test-payload'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'loaded-test-payload'
    var_6 = b'test-payload'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'loaded-test-payload'
    var_8 = b'test-payload'
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_payload'
    var_5 = var_1.load_payload(var_4)
    assert var_5 == 'custom_test_payload'
    var_6 = b'test_payload'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'custom_test_payload'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid_json'
    var_10 = var_8.load_payload(var_9)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = 42
    var_8 = module_1.dumps(var_7)
    var_9 = module_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = var_11.dumps(var_4)
    var_13 = var_11.loads(var_12)
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Serializer(var_16)
    var_18 = var_17.dumps(var_4)
    var_19 = var_17.loads(var_18)
    var_20 = module_1.dumps(var_4)
    var_21 = module_1.loads(var_20)



# Parsed testcases at query #37
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #38
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'HELLO'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'olleh'



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = 'test-data'
    var_8 = var_1.dumps(var_7)
    var_9 = var_1.loads(var_8)
    var_10 = 'test-data'
    var_11 = var_1.dumps(var_10)
    var_12 = var_1.loads(var_11)
    var_13 = module_0.Serializer(var_0)
    var_14 = 'invalid-signature'
    var_15 = var_13.loads(var_14)
    var_16 = module_0.Serializer(var_14)
    var_17 = 'valid-signature-but-invalid-payload'
    var_18 = var_16.loads(var_17)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_1.dumps(var_9)
    var_11 = 'test string'
    var_12 = var_1.dumps(var_11)
    var_13 = 'custom-salt'
    var_14 = var_1.dumps(var_11, var_13)
    var_15 = {var_2: var_3}
    var_16 = module_1.dumps(var_15)



# Parsed testcases at query #41
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)



# Parsed testcases at query #42
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)
    var_5 = module_0.dumps(var_2)
    var_6 = module_0.loads(var_5)
    var_7 = 42
    var_8 = module_0.dumps(var_7)
    var_9 = module_0.loads(var_8)
    assert var_9 == 42



# Parsed testcases at query #43
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'key': 'value'}"



# Parsed testcases at query #44
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = module_0.loads(var_0)
    assert var_4 == 'HELLO'
    var_5 = module_0.loads(var_2)
    assert var_5 == b'olleh'



# Parsed testcases at query #45
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    var_10 = module_0.loads(var_9)
    var_11 = None
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = 'test string'
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 42
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)



