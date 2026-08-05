####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_7 = {var_2: var_3}
    var_8 = var_1.dumps(var_7)
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13)
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Serializer(var_17)
    var_19 = {var_2: var_3}
    var_20 = var_18.dumps(var_19)
    var_21 = var_18.loads(var_20)
    var_22 = {var_2: var_3}
    var_23 = var_18.dumps(var_22)
    var_24 = var_18.loads(var_23)



# Parsed testcases at query #2
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = module_0.dumps(var_0)
    assert var_4 == b'test'
    var_5 = module_0.dumps(var_2)
    assert var_5 == b'123'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    assert var_9 == '{"key": "value"}'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = 'test_value'
    var_16 = 'CustomObject(test_value)'
    var_17 = module_0.loads(var_16)
    var_18 = var_17.value
    assert var_18 == 'test_value'



# Parsed testcases at query #3
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
    var_5 = module_0.loads(var_2)
    assert var_5 == "loaded: b'test'"



# Parsed testcases at query #4
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #5
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = module_0.dumps(var_0)
    assert var_4 == b'test'
    var_5 = module_0.dumps(var_2)
    assert var_5 == b'123'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    assert var_9 == '{"key": "value"}'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = module_0.dumps(var_2)
    assert var_15 == '123'



# Parsed testcases at query #6
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'
    var_13 = {var_0: var_1}
    var_14 = module_0.dumps(var_13)
    assert var_14 == '{"key": "value"}'
    var_15 = [var_4, var_5, var_6]
    var_16 = module_0.dumps(var_15)
    assert var_16 == '[1, 2, 3]'
    var_17 = 'simple string'
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'simple string'



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = var_2.loads(var_6)
    var_8 = module_1.dumps(var_5)
    var_9 = module_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = var_11.dumps(var_5)
    var_13 = var_11.loads(var_12)
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = 'rotation-salt'
    var_18 = module_0.Serializer(var_16, var_17)
    var_19 = var_18.dumps(var_5)
    var_20 = var_18.loads(var_19)
    var_21 = None
    var_22 = 123
    var_23 = 45.67
    var_24 = 'string'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_25, var_30: var_26}
    var_32 = True
    var_33 = False
    var_34 = [var_21, var_22, var_23, var_24, var_28, var_31, var_32, var_33]



# Parsed testcases at query #8
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
    var_15 = [var_14]
    var_16 = module_0.Serializer(var_0, fallback_signers=var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = {var_12: var_13}
    var_21 = var_16.iter_unsigners()
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_16.iter_unsigners()
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'old-key'
    var_28 = 'new-key'
    var_29 = [var_27, var_28]
    var_30 = module_0.Serializer(var_29)
    var_31 = var_30.iter_unsigners()
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = [var_27, var_28]
    var_35 = {var_12: var_13}
    var_36 = [var_35]
    var_37 = module_0.Serializer(var_34, fallback_signers=var_36)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 2



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = 'custom-salt'
    var_9 = module_0.Serializer(var_0, var_8)
    var_10 = var_9.dumps(var_5)
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Serializer(var_13, var_1)
    var_15 = var_14.dumps(var_5)



# Parsed testcases at query #12
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
    var_33 = 'data'
    var_34 = {var_33: var_30}
    var_35 = {var_33: var_24}
    var_36 = var_1.dumps(var_34)
    var_37 = var_1.dumps(var_35)



# Parsed testcases at query #13
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'



# Parsed testcases at query #14
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
    var_7 = b'salt'
    var_8 = module_1.dumps(var_4)
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



# Parsed testcases at query #15
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #16
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
    var_10 = 3.14
    var_11 = module_1.dumps(var_10)
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_7 = -1
    var_8 = var_5[:var_7]
    var_9 = b'x'
    var_10 = var_8 + var_9
    var_11 = var_1.loads(var_10)
    var_12 = b'corrupted'
    var_13 = var_1.loads(var_12)
    var_14 = 42
    var_15 = module_1.dumps(var_14)
    var_16 = module_1.loads(var_15)
    assert var_16 == 42
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Serializer(var_19)
    var_21 = var_20.dumps(var_4)
    var_22 = var_20.loads(var_21)



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'
    var_4 = 'test'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #21
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)
    assert var_25 is None



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = 'test'
    var_7 = module_0.loads(var_6)



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
    var_4 = module_0.loads(var_3)



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'hello'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'HELLO'



# Parsed testcases at query #26
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = module_0.dumps(var_0)
    assert var_4 == b'test'
    var_5 = module_0.dumps(var_2)
    assert var_5 == b'123'



# Parsed testcases at query #27
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'



# Parsed testcases at query #28
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = True
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = None
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    assert var_22 is None



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'test'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)



# Parsed testcases at query #32
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"



# Parsed testcases at query #33
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'



# Parsed testcases at query #34
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)
    assert var_25 is None



# Parsed testcases at query #35
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    assert var_7 == "{'key': 'value'}"
    var_8 = module_0.dumps(var_0)
    assert var_8 == b'test'
    var_9 = module_0.dumps(var_2)
    assert var_9 == b'123'
    var_10 = {var_4: var_5}
    var_11 = module_0.dumps(var_10)
    assert var_11 == b"{'key': 'value'}"



# Parsed testcases at query #36
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '{"key": "value"'
    var_3 = module_0.loads(var_2)
    var_4 = 'test'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'TEST'



# Parsed testcases at query #37
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
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 'key3'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'digest_method'
    var_21 = 'sha256'
    var_22 = {var_20: var_21}
    var_23 = [var_22]
    var_24 = module_0.Serializer(var_0, fallback_signers=var_23)
    var_25 = var_24.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_24.iter_unsigners()
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 1
    var_32 = var_29[var_31]
    var_33 = {var_20: var_21}
    var_34 = var_24.iter_unsigners()
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = {var_20: var_21}
    var_38 = 'sha512'
    var_39 = {var_20: var_38}
    var_40 = var_24.iter_unsigners()
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 4
    var_43 = 2
    var_44 = var_41[var_43]
    var_45 = 'default-salt'
    var_46 = module_0.Serializer(var_0, var_45)
    var_47 = var_46.iter_unsigners(var_7)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1



# Parsed testcases at query #38
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.loads(var_0)
    var_5 = b'hello'
    var_6 = 'HELLO'
    var_7 = module_0.loads(var_5)
    var_8 = 'a,b,c'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.loads(var_8)



# Parsed testcases at query #39
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)



# Parsed testcases at query #40
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
    var_7 = var_1.dumps(var_4)
    var_8 = var_1.loads(var_7)
    var_9 = var_1.dumps(var_4)
    var_10 = var_1.loads(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = var_1.dumps(var_14)
    var_16 = var_1.loads(var_15)
    var_17 = 'test string'
    var_18 = var_1.dumps(var_17)
    var_19 = var_1.loads(var_18)
    var_20 = 'custom-salt'
    var_21 = module_0.Serializer(var_0, var_20)
    var_22 = var_21.dumps(var_4)
    var_23 = var_21.loads(var_22)



# Parsed testcases at query #41
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b'{"key": "value"}'
    var_6 = 'test'



# Parsed testcases at query #42
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'
    var_13 = {var_0: var_1}
    var_14 = module_0.dumps(var_13)
    assert var_14 == '{"key": "value"}'
    var_15 = [var_4, var_5, var_6]
    var_16 = module_0.dumps(var_15)
    assert var_16 == '[1, 2, 3]'
    var_17 = 'test'
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'test'



# Parsed testcases at query #43
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
    var_7 = var_1.dumps(var_4)
    var_8 = 'custom-'
    var_9 = var_1.loads(var_7)
    var_10 = var_1.dumps(var_4)
    var_11 = b'custom-'
    var_12 = var_1.loads(var_10)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.dumps(var_4)
    var_18 = var_16.loads(var_17)
    var_19 = 'custom-salt'
    var_20 = module_0.Serializer(var_0, var_19)
    var_21 = var_20.dumps(var_4)
    var_22 = var_20.loads(var_21)



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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = True
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = None
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)



# Parsed testcases at query #46
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'



# Parsed testcases at query #47
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    assert var_7 == "{'key': 'value'}"
    var_8 = module_0.dumps(var_0)
    assert var_8 == b'test'
    var_9 = module_0.dumps(var_2)
    assert var_9 == b'123'
    var_10 = {var_4: var_5}
    var_11 = module_0.dumps(var_10)
    assert var_11 == b"{'key': 'value'}"



# Parsed testcases at query #48
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = '{"a": 1}'
    var_5 = module_0.loads(var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = [var_2, var_3]
    var_5 = 'custom-salt'
    var_6 = 'indent'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = 'sep'
    var_10 = '|'
    var_11 = {var_9: var_10}
    var_12 = ':'
    var_13 = {var_9: var_12}
    var_14 = [var_13]
    var_15 = None
    var_16 = module_0.Serializer(var_0, var_15)
    var_17 = {var_9: var_12}
    var_18 = b'secret-key'
    var_19 = module_0.Serializer(var_18)
    var_20 = b'key1'
    var_21 = b'key2'
    var_22 = [var_20, var_21]
    var_23 = module_0.Serializer(var_22)
    var_24 = [var_2, var_3]
    var_25 = module_0.Serializer(var_24)
    var_26 = b'custom-salt'
    var_27 = module_0.Serializer(var_0, var_26)
    var_28 = module_0.Serializer(var_0, var_5)



# Parsed testcases at query #2
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
    var_11 = var_8.iter_unsigners()
    var_12 = list(var_11)
    var_13 = var_12[var_5]
    var_14 = 'digest_method'
    var_15 = 'sha256'
    var_16 = {var_14: var_15}
    var_17 = module_0.Serializer(var_0, signer_kwargs=var_16)
    var_18 = var_17.iter_unsigners()
    var_19 = list(var_18)
    var_20 = {var_14: var_15}
    var_21 = [var_20]
    var_22 = module_0.Serializer(var_0, fallback_signers=var_21)
    var_23 = var_22.iter_unsigners()
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = {var_14: var_15}
    var_27 = var_22.iter_unsigners()
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_22.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 1
    var_34 = var_31[var_33]
    var_35 = {var_14: var_15}
    var_36 = 'sha384'
    var_37 = {var_14: var_36}
    var_38 = var_22.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 3
    var_41 = 2
    var_42 = var_39[var_41]
    var_43 = 'old-key'
    var_44 = 'new-key'
    var_45 = [var_43, var_44]
    var_46 = module_0.Serializer(var_45)
    var_47 = var_46.iter_unsigners()
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = [var_43, var_44]
    var_51 = {var_14: var_15}
    var_52 = [var_51]
    var_53 = module_0.Serializer(var_50, fallback_signers=var_52)
    var_54 = var_53.iter_unsigners()
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 3
    var_57 = 'default-salt'
    var_58 = module_0.Serializer(var_0, var_57)
    var_59 = var_58.iter_unsigners(var_7)
    var_60 = list(var_59)



# Parsed testcases at query #3
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = True
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = None
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    assert var_22 is None



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
    var_9 = str(var_4)
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
    var_20 = None
    var_21 = 123
    var_22 = 45.67
    var_23 = 'string'
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_24, var_29: var_25}
    var_31 = True
    var_32 = False
    var_33 = [var_20, var_21, var_22, var_23, var_27, var_30, var_31, var_32]



# Parsed testcases at query #5
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)
    var_5 = {var_0: var_1}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_0.dumps(var_8)
    var_10 = module_0.loads(var_9)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = 'custom-salt'
    var_9 = module_0.Serializer(var_0, var_8)
    var_10 = var_9.dumps(var_5)
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Serializer(var_13, var_1)
    var_15 = var_14.dumps(var_5)



# Parsed testcases at query #7
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'



# Parsed testcases at query #8
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'test'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = module_0.loads(var_0)
    assert var_8 == 'test'
    var_9 = module_0.loads(var_2)
    assert var_9 == 'test'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = 'custom-salt'
    var_9 = module_0.Serializer(var_0, var_8)
    var_10 = var_9.dumps(var_5)
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Serializer(var_13, var_1)
    var_15 = var_14.dumps(var_5)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_2.dumps(var_19)
    var_21 = {}
    var_22 = var_2.dumps(var_21)



# Parsed testcases at query #10
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
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 'key3'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Serializer(var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'digest_method'
    var_21 = 'sha256'
    var_22 = {var_20: var_21}
    var_23 = [var_22]
    var_24 = module_0.Serializer(var_0, fallback_signers=var_23)
    var_25 = var_24.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = {var_20: var_21}
    var_29 = var_24.iter_unsigners()
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_24.iter_unsigners()
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = {var_20: var_21}
    var_36 = 'sha512'
    var_37 = {var_20: var_36}
    var_38 = var_24.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 4
    var_41 = 'main-salt'
    var_42 = 'salt'
    var_43 = 'fallback-salt'
    var_44 = {var_42: var_43}
    var_45 = [var_44]
    var_46 = module_0.Serializer(var_0, var_41, fallback_signers=var_45)
    var_47 = var_46.iter_unsigners()
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'default-salt'
    var_51 = module_0.Serializer(var_0, var_50)
    var_52 = var_51.iter_unsigners(var_7)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1



# Parsed testcases at query #11
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded_test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'loaded_test'



# Parsed testcases at query #12
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'test'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = module_0.loads(var_0)
    assert var_8 == 'TEST'
    var_9 = module_0.loads(var_2)
    assert var_9 == b'TEST'



# Parsed testcases at query #13
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
    var_5 = module_0.loads(var_2)
    assert var_5 == 'loaded_test'



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


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'digest_method'
    var_9 = 'md5'
    var_10 = {var_8: var_9}
    var_11 = 'sha1'
    var_12 = {var_8: var_11}
    var_13 = len(var_4)
    assert var_13 == 3
    var_14 = var_4[var_6]
    var_15 = 1
    var_16 = var_4[var_15]
    var_17 = 2
    var_18 = var_4[var_17]
    var_19 = 'new-salt'
    var_20 = var_2.iter_unsigners(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'old-secret'
    var_24 = 'new-secret'
    var_25 = [var_23, var_24]
    var_26 = module_0.Serializer(var_25, var_1)
    var_27 = var_26.iter_unsigners()
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = len(var_28)
    assert var_30 == 5



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = module_1.loads(var_8)
    var_10 = module_0.Serializer(var_0)
    var_11 = {var_2: var_3}
    var_12 = var_10.dumps(var_11)
    var_13 = b'invalid'
    var_14 = var_12 + var_13
    var_15 = var_10.loads(var_14)
    var_16 = module_0.Serializer(var_15)
    var_17 = b'invalid_payload'
    var_18 = var_16.loads(var_17)
    var_19 = 'custom-salt'
    var_20 = module_0.Serializer(var_18, var_19)
    var_21 = {var_2: var_3}
    var_22 = var_20.dumps(var_21)
    var_23 = var_20.loads(var_22)
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Serializer(var_26)
    var_28 = {var_2: var_3}
    var_29 = var_27.dumps(var_28)
    var_30 = var_27.loads(var_29)



# Parsed testcases at query #18
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
    var_11 = 'hello world'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_payload'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'test_payload'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'test_payload'
    var_9 = var_1.load_payload(var_8)



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
    var_4 = 'test'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #21
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'loaded: test'
    var_2 = b'test'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'loaded: test'



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'test'
    var_2 = 123
    var_3 = module_0.dumps(var_2)
    assert var_3 == '123'
    var_4 = module_0.dumps(var_0)
    assert var_4 == b'test'
    var_5 = module_0.dumps(var_2)
    assert var_5 == b'123'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    assert var_9 == '{"key": "value"}'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = 42
    var_16 = module_0.dumps(var_0)
    assert var_16 == 'test'



# Parsed testcases at query #23
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'
    var_13 = {var_0: var_1}
    var_14 = module_0.dumps(var_13)
    assert var_14 == '{"key": "value"}'
    var_15 = [var_4, var_5, var_6]
    var_16 = module_0.dumps(var_15)
    assert var_16 == '[1, 2, 3]'
    var_17 = 'test'
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'test'



# Parsed testcases at query #24
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
    var_9 = {var_0: var_1}
    var_10 = module_0.dumps(var_9)
    assert var_10 == b"{'key': 'value'}"
    var_11 = [var_4, var_5, var_6]
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'[1, 2, 3]'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = module_0.dumps(var_15)



# Parsed testcases at query #25
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
    assert var_7 == b'loaded: test_data'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid_json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'test_data'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #26
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
    var_7 = var_1.dumps(var_4)
    var_8 = var_1.loads(var_7)
    var_9 = str(var_4)
    var_10 = var_1.dumps(var_4)
    var_11 = var_1.loads(var_10)
    var_12 = str(var_4)
    var_13 = 'custom-salt'
    var_14 = module_0.Serializer(var_0, var_13)
    var_15 = var_14.dumps(var_4)
    var_16 = var_14.loads(var_15, var_13)
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Serializer(var_19)
    var_21 = var_20.dumps(var_4)
    var_22 = var_20.loads(var_21)



# Parsed testcases at query #27
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
    var_11 = 'test string'
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 42
    var_15 = module_0.dumps(var_14)
    var_16 = module_0.loads(var_15)
    var_17 = 3.14
    var_18 = module_0.dumps(var_17)
    var_19 = module_0.loads(var_18)
    var_20 = True
    var_21 = module_0.dumps(var_20)
    var_22 = module_0.loads(var_21)
    var_23 = None
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)
    assert var_25 is None



# Parsed testcases at query #28
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
    var_7 = 42
    var_8 = var_1.dumps(var_7)
    var_9 = var_1.loads(var_8)
    var_10 = 42
    var_11 = var_1.dumps(var_10)
    var_12 = var_1.loads(var_11)
    var_13 = module_0.Serializer(var_0)
    var_14 = 'invalid-signature'
    var_15 = var_13.loads(var_14)
    var_16 = module_0.Serializer(var_14)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.dumps(var_19)
    var_21 = b'corrupted'
    var_22 = var_20 + var_21
    var_23 = var_16.loads(var_22)



# Parsed testcases at query #29
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'hello'
    var_3 = module_0.loads(var_2)
    assert var_3 == b'olleh'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = 'test'
    var_7 = module_0.loads(var_6)



# Parsed testcases at query #30
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'data'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'bytes_data'
    var_7 = var_1.load_payload(var_6)
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid_json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'data'
    var_12 = var_8.load_payload(var_11)



# Parsed testcases at query #32
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
    assert var_4 == 'loaded_test_default'
    var_5 = 'custom'



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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
    var_6 = module_1.dumps(var_4)
    var_7 = 'custom-salt'
    var_8 = module_0.Serializer(var_0, var_7)
    var_9 = var_8.dumps(var_4)
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Serializer(var_12)
    var_14 = var_13.dumps(var_4)
    var_15 = module_1.dumps(var_4)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'custom_data'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'custom_data'
    var_7 = var_1.load_payload(var_6)
    var_8 = module_0.Serializer(var_0)
    var_9 = b'{"key": "value"}'
    var_10 = module_0.Serializer(var_0)
    var_11 = b'invalid_json'
    var_12 = var_10.load_payload(var_11)



# Parsed testcases at query #36
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



