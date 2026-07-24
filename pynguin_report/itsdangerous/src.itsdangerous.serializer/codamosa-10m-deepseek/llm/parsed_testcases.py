####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = ''
    var_5 = module_0.loads(var_4)
    var_6 = '123'
    var_7 = module_0.loads(var_6)
    var_8 = 'hello world!'
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test data'
    var_5 = b'binary data'
    var_6 = module_0.Serializer(var_0)
    var_7 = b'data'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b'test'
    var_12 = b'test'
    var_13 = module_0.Serializer(var_12)
    var_14 = b''
    var_15 = var_13.load_payload(var_14)
    var_16 = module_0.Serializer(var_14)
    var_17 = b'null'
    var_18 = var_16.load_payload(var_17)
    assert var_18 is None
    var_19 = b'[]'
    var_20 = var_16.load_payload(var_19)
    var_21 = b'{}'
    var_22 = var_16.load_payload(var_21)



# Parsed testcases at query #3
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    assert var_5 == 42
    var_6 = '[1, 2, 3]'
    var_7 = module_0.loads(var_6)
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = '3.14'
    var_13 = module_0.loads(var_12)



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
    var_7 = {var_2: var_3}
    var_8 = 'custom-salt'
    var_9 = var_1.dumps(var_7, var_8)
    var_10 = var_1.loads(var_9, var_8)
    var_11 = 'data1'
    var_12 = var_1.dumps(var_11)
    var_13 = 'data2'
    var_14 = var_1.dumps(var_13)
    var_15 = {var_2: var_3}
    var_16 = module_1.dumps(var_15)
    var_17 = module_1.loads(var_16)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps correctly signs and serializes data.'
    var_1 = 'test-secret'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'different'
    var_8 = {var_3: var_7}
    var_9 = var_2.dumps(var_8)
    var_10 = {var_3: var_4}
    var_11 = module_1.dumps(var_10)
    var_12 = {var_3: var_4}
    var_13 = 'custom-salt'
    var_14 = var_2.dumps(var_12, var_13)
    var_15 = 'sort_keys'
    var_16 = 'separators'
    var_17 = True
    var_18 = ','
    var_19 = ':'
    var_20 = (var_18, var_19)
    var_21 = {var_15: var_17, var_16: var_20}
    var_22 = 'b'
    var_23 = 'a'
    var_24 = 2
    var_25 = {var_22: var_24, var_23: var_17}
    var_26 = module_1.dumps(var_25)
    var_27 = len(var_6)



# Parsed testcases at query #6
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'a,b,c'
    var_5 = module_0.loads(var_4)
    var_6 = '42'
    var_7 = module_0.loads(var_6)
    assert var_7 == 42
    var_8 = 'anything'
    var_9 = module_0.loads(var_8)
    assert var_9 is None



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that iter_unsigners yields the main signer first, then fallback signers.'
    var_1 = b'test-secret-key'
    var_2 = b'test-salt'
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = 'sha512'
    var_8 = {var_3: var_7}
    var_9 = 0
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = b'custom-salt'
    var_14 = []
    var_15 = module_0.Serializer(var_1, var_2, fallback_signers=var_14)
    var_16 = var_15.iter_unsigners()
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = var_17[var_9]
    var_20 = b'old-key'
    var_21 = b'newer-key'
    var_22 = b'current-key'
    var_23 = [var_20, var_21, var_22]
    var_24 = {var_3: var_4}
    var_25 = [var_24]
    var_26 = module_0.Serializer(var_23, var_2, fallback_signers=var_25)
    var_27 = var_26.iter_unsigners()
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 4



# Parsed testcases at query #8
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = '[1, 2, 3]'
    var_5 = module_0.loads(var_4)
    var_6 = '{"nested": {"inner": "data"}}'
    var_7 = module_0.loads(var_6)
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'false'
    var_11 = module_0.loads(var_10)
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = '""'
    var_15 = module_0.loads(var_14)
    assert var_15 == ''
    var_16 = '3.14'
    var_17 = module_0.loads(var_16)
    var_18 = '{}'
    var_19 = module_0.loads(var_18)
    var_20 = '[]'
    var_21 = module_0.loads(var_20)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = []
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = b'custom-salt'
    var_9 = []
    var_10 = module_0.Serializer(var_0, var_8, fallback_signers=var_9)
    var_11 = b'override-salt'
    var_12 = var_10.iter_unsigners(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'key_derivation'
    var_16 = 'none'
    var_17 = {var_15: var_16}
    var_18 = [var_17]
    var_19 = module_0.Serializer(var_0, fallback_signers=var_18)
    var_20 = var_19.iter_unsigners()
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = var_21[var_6]
    var_24 = 1
    var_25 = var_21[var_24]
    var_26 = 'hmac'
    var_27 = {var_15: var_26}
    var_28 = var_19.iter_unsigners()
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = var_29[var_6]
    var_32 = var_29[var_24]
    var_33 = var_19.iter_unsigners()
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = var_34[var_6]
    var_37 = var_34[var_24]
    var_38 = 'old-key'
    var_39 = 'new-key'
    var_40 = [var_38, var_39]
    var_41 = {var_15: var_16}
    var_42 = [var_41]
    var_43 = module_0.Serializer(var_40, fallback_signers=var_42)
    var_44 = var_43.iter_unsigners()
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 3
    var_47 = module_0.Serializer(var_0)
    var_48 = var_47.iter_unsigners()
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = {var_15: var_16}
    var_52 = 'digest_method'
    var_53 = 'sha256'
    var_54 = {var_52: var_53}
    var_55 = var_47.iter_unsigners()
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 4
    var_58 = var_56[var_6]
    var_59 = var_56[var_24]
    var_60 = 2
    var_61 = var_56[var_60]
    var_62 = 3
    var_63 = var_56[var_62]
    var_64 = b'base-salt'
    var_65 = {var_15: var_16}
    var_66 = [var_65]
    var_67 = module_0.Serializer(var_0, var_64, fallback_signers=var_66)
    var_68 = var_67.iter_unsigners()
    var_69 = list(var_68)
    var_70 = b'override'
    var_71 = var_67.iter_unsigners(var_70)
    var_72 = list(var_71)



# Parsed testcases at query #10
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"key": "value"}'
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value"}'
    var_6 = {}
    var_7 = module_0.dumps(var_6)
    assert var_7 == b'{}'
    var_8 = None
    var_9 = module_0.dumps(var_8)
    assert var_9 == b'null'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == b'[1, 2, 3]'
    var_15 = 42
    var_16 = module_0.dumps(var_15)
    assert var_16 == b'42'
    var_17 = 'test'
    var_18 = module_0.dumps(var_17)
    assert var_18 == b'"test"'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that _PDataSerializer.dumps returns the expected type.'
    var_1 = module_0._PDataSerializer()
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = {var_2: var_3}
    var_7 = module_1.dumps(var_6)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'not valid json'
    var_6 = var_2.load_payload(var_5)
    var_7 = b'hello'
    var_8 = b''
    var_9 = var_2.load_payload(var_8)
    var_10 = b'test'
    var_11 = b'hello'
    var_12 = b'\xc3\xa9'
    var_13 = b'\xff\xfe'
    var_14 = None
    var_15 = var_2.load_payload(var_14)
    var_16 = 123
    var_17 = var_2.load_payload(var_16)



# Parsed testcases at query #13
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = module_0.dumps(var_3)
    assert var_5 == b'{"key": "value"}'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = 'TextSerializer'
    var_6 = ()
    var_7 = 'dumps'
    var_8 = 'loads'
    var_9 = '{"text": "data"}'
    var_10 = lambda self, x: var_9
    var_11 = 'text'
    var_12 = 'data'
    var_13 = {var_11: var_12}
    var_14 = lambda self, x: var_13
    var_15 = {var_7: var_10, var_8: var_14}
    var_16 = b'{"text": "data"}'
    var_17 = 'CustomSerializer'
    var_18 = ()
    var_19 = '{"custom": "data"}'
    var_20 = lambda self, x: var_19
    var_21 = 'custom'
    var_22 = 'data_from_custom'
    var_23 = {var_21: var_22}
    var_24 = lambda self, x: var_23
    var_25 = {var_7: var_20, var_8: var_24}
    var_26 = b'{"custom": "data"}'
    var_27 = b'invalid json'
    var_28 = var_2.load_payload(var_27)
    var_29 = b''
    var_30 = var_2.load_payload(var_29)
    var_31 = 'BytesSerializer'
    var_32 = ()
    var_33 = b'{"bytes": "data"}'
    var_34 = lambda self, x: var_33
    var_35 = 'bytes'
    var_36 = 'data_from_bytes'
    var_37 = {var_35: var_36}
    var_38 = lambda self, x: var_37
    var_39 = {var_7: var_34, var_8: var_38}
    var_40 = b'{"level1": {"level2": [1, 2, 3]}}'
    var_41 = var_2.load_payload(var_40)
    var_42 = b'[1, 2, 3, "test"]'
    var_43 = var_2.load_payload(var_42)
    var_44 = b'{"test": "value"}'
    var_45 = None
    var_46 = var_2.load_payload(var_44, var_45)
    var_47 = 'ErrorSerializer'
    var_48 = ()
    var_49 = b'data'
    var_50 = lambda self, x: var_49
    var_51 = ()
    var_52 = 'Custom error'
    var_53 = b'some data'



# Parsed testcases at query #15
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_0.dumps(var_4)



# Parsed testcases at query #16
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test_data'
    var_5 = b'binary_data'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'\xff\xfe\x00\x01'
    var_11 = var_1.load_payload(var_10)
    var_12 = b'test'
    var_13 = var_1.load_payload(var_12, var_11)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = var_1.loads(var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 'test'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = var_1.dumps(var_11)
    var_13 = var_1.loads(var_12)
    var_14 = None
    var_15 = var_1.dumps(var_14)
    var_16 = var_1.loads(var_15)
    assert var_16 is None
    var_17 = 'custom-salt'
    var_18 = var_1.dumps(var_4, var_17)
    var_19 = var_1.loads(var_18, var_17)
    var_20 = 'different-salt'
    var_21 = var_1.dumps(var_4, var_20)
    var_22 = 'test-key'
    var_23 = 'data'
    var_24 = {var_10: var_23}
    var_25 = module_1.dumps(var_24)
    var_26 = module_1.loads(var_25)
    var_27 = {}
    var_28 = var_1.dumps(var_27)
    var_29 = var_1.loads(var_28)
    var_30 = 'level1'
    var_31 = 'level2'
    var_32 = [var_7, var_8, var_9]
    var_33 = {var_31: var_32}
    var_34 = {var_30: var_33}
    var_35 = var_1.dumps(var_34)
    var_36 = var_1.loads(var_35)
    var_37 = var_1.dumps(var_10)



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.dumps returns the expected serialized type.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 'Text serializer should return str'
    var_6 = {var_1: var_2}
    var_7 = module_0.dumps(var_6)
    assert var_7 == b'{"key": "value"}'
    var_8 = 'Bytes serializer should return bytes'
    var_9 = 'test'
    var_10 = 123
    var_11 = {var_9: var_10}



# Parsed testcases at query #21
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"key": "value"}'
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value"}'
    var_6 = None
    var_7 = module_0.dumps(var_6)
    assert var_7 == b'null'
    var_8 = {}
    var_9 = module_0.dumps(var_8)
    assert var_9 == b'{}'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == b'[1, 2, 3]'
    var_15 = 'test_string'
    var_16 = module_0.dumps(var_15)
    assert var_16 == '"test_string"'
    var_17 = 42
    var_18 = module_0.dumps(var_17)
    assert var_18 == b'42'
    var_19 = 3.14
    var_20 = module_0.dumps(var_19)
    assert var_20 == '3.14'
    var_21 = True
    var_22 = module_0.dumps(var_21)
    assert var_22 == b'true'
    var_23 = False
    var_24 = module_0.dumps(var_23)
    assert var_24 == b'false'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'test:data'
    assert var_3 == 'HELLO'
    var_4 = b'test:data'
    var_5 = b'hello'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'{"test": 123}'



# Parsed testcases at query #23
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps works correctly with different implementations.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = None
    var_8 = module_0.dumps(var_7)
    assert var_8 == 'null'
    var_9 = True
    var_10 = module_0.dumps(var_9)
    assert var_10 == 'true'
    var_11 = False
    var_12 = module_0.dumps(var_11)
    assert var_12 == 'false'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'
    var_15 = 3.14
    var_16 = module_0.dumps(var_15)
    assert var_16 == '3.14'
    var_17 = 'hello'
    var_18 = module_0.dumps(var_17)
    assert var_18 == '"hello"'
    var_19 = 2
    var_20 = 3
    var_21 = [var_9, var_19, var_20]
    var_22 = module_0.dumps(var_21)
    assert var_22 == '[1, 2, 3]'
    var_23 = module_0.dumps(var_7)
    assert var_23 == b'null'
    var_24 = module_0.dumps(var_9)
    assert var_24 == b'true'
    var_25 = module_0.dumps(var_13)
    assert var_25 == b'42'
    var_26 = [var_9, var_19, var_20]
    var_27 = module_0.dumps(var_26)
    assert var_27 == b'[1, 2, 3]'



# Parsed testcases at query #24
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method is properly defined."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '"string"'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'string'
    var_5 = '123'
    var_6 = module_0.loads(var_5)
    assert var_6 == 123
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = '{}'
    var_14 = module_0.loads(var_13)
    var_15 = '[]'
    var_16 = module_0.loads(var_15)
    var_17 = '{"a": {"b": [1, 2, {"c": "d"}]}}'
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 1
    var_21 = 2
    var_22 = 'c'
    var_23 = 'd'
    var_24 = {var_22: var_23}
    var_25 = [var_20, var_21, var_24]
    var_26 = {var_19: var_25}
    var_27 = {var_18: var_26}
    var_28 = module_0.loads(var_17)
    var_29 = 'invalid json'
    var_30 = module_0.loads(var_29)
    var_31 = ''
    var_32 = module_0.loads(var_31)



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"number": 42}'
    var_3 = module_0.loads(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.loads(var_4)
    var_6 = '"string"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'string'
    var_8 = '123'
    var_9 = module_0.loads(var_8)
    assert var_9 == 123
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = 'null'
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = '{}'
    var_15 = module_0.loads(var_14)
    var_16 = '[]'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #26
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
    var_6 = '.'
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = b'.'
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = 'sort_keys'
    var_14 = 'separators'
    var_15 = True
    var_16 = ','
    var_17 = ':'
    var_18 = (var_16, var_17)
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = 'b'
    var_21 = 'a'
    var_22 = 2
    var_23 = {var_20: var_15, var_21: var_22}
    var_24 = module_1.dumps(var_23)
    var_25 = 'old-key'
    var_26 = 'new-key'
    var_27 = [var_25, var_26]
    var_28 = module_0.Serializer(var_27)
    var_29 = {var_2: var_3}
    var_30 = var_28.dumps(var_29)
    var_31 = var_28.make_signer()
    var_32 = {var_2: var_3}
    var_33 = var_28.dump_payload(var_32)
    var_34 = var_31.sign(var_33)
    var_35 = {}
    var_36 = var_1.dumps(var_35)
    var_37 = None
    var_38 = var_1.dumps(var_37)
    var_39 = 3
    var_40 = [var_15, var_22, var_39]
    var_41 = var_1.dumps(var_40)
    var_42 = 'secret'
    var_43 = 'test'
    var_44 = 'data'
    var_45 = {var_43: var_44}
    var_46 = module_1.dumps(var_45)
    var_47 = module_0.Serializer(var_42)
    var_48 = {var_43: var_44}
    var_49 = var_47.dumps(var_48)



# Parsed testcases at query #27
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works with various types."
    var_1 = 'hello'
    var_2 = module_0.loads(var_1)
    assert var_2 == 'HELLO'
    var_3 = 'test'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'TEST'
    var_5 = ''
    var_6 = module_0.loads(var_5)
    assert var_6 == ''
    var_7 = b'hello'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'HELLO'
    var_9 = b'test'
    var_10 = module_0.loads(var_9)
    assert var_10 == 'TEST'
    var_11 = b''
    var_12 = module_0.loads(var_11)
    assert var_12 == ''



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps returns correctly signed and serialized data.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = len(var_6)
    var_8 = module_1.dumps(var_5)
    var_9 = len(var_6)
    var_10 = var_2.loads(var_6)
    var_11 = 'sort_keys'
    var_12 = 'indent'
    var_13 = True
    var_14 = 2
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.Serializer(var_9, serializer_kwargs=var_15)
    var_17 = var_16.dumps(var_5)



# Parsed testcases at query #29
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'hello'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'HELLO'
    var_6 = 'test'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'test'
    var_8 = '123'
    var_9 = module_0.loads(var_8)
    assert var_9 == '123'
    var_10 = 'data'
    var_11 = module_0.loads(var_10)
    assert var_11 == 'data'



# Parsed testcases at query #30
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"key": "value"}'
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value"}'
    var_6 = None
    var_7 = module_0.dumps(var_6)
    assert var_7 == 'null'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'



# Parsed testcases at query #31
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
    var_6 = {var_2: var_3}
    var_7 = module_1.dumps(var_6)
    var_8 = 'sort_keys'
    var_9 = 'separators'
    var_10 = True
    var_11 = ','
    var_12 = ':'
    var_13 = (var_11, var_12)
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = module_0.Serializer(var_0, serializer_kwargs=var_14)
    var_16 = 'b'
    var_17 = 'a'
    var_18 = 2
    var_19 = {var_16: var_10, var_17: var_18}
    var_20 = var_15.dumps(var_19)
    var_21 = 'custom-salt'
    var_22 = module_0.Serializer(var_0, var_21)
    var_23 = {var_2: var_3}
    var_24 = var_22.dumps(var_23)
    var_25 = 'key1'
    var_26 = module_0.Serializer(var_25)
    var_27 = 'key2'
    var_28 = module_0.Serializer(var_27)
    var_29 = 'data'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = var_26.dumps(var_31)
    var_33 = {var_29: var_30}
    var_34 = var_28.dumps(var_33)
    var_35 = {}
    var_36 = var_1.dumps(var_35)
    var_37 = 3
    var_38 = [var_10, var_18, var_37]
    var_39 = var_1.dumps(var_38)
    var_40 = None
    var_41 = var_1.dumps(var_40)
    var_42 = 'test string'
    var_43 = var_1.dumps(var_42)
    var_44 = 42
    var_45 = var_1.dumps(var_44)
    var_46 = var_1.loads(var_5)
    var_47 = 'nested'
    var_48 = {var_29: var_10}
    var_49 = {var_47: var_48}
    var_50 = 'two'
    var_51 = [var_10, var_50, var_37]
    var_52 = 'simple string'
    var_53 = 12345
    var_54 = 'complex'
    var_55 = [var_10, var_18, var_37]
    var_56 = {var_54: var_55}
    var_57 = [var_56]
    var_58 = [var_49, var_51, var_52, var_53, var_40, var_57]



# Parsed testcases at query #32
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #33
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = {}
    var_8 = module_0.dumps(var_7)
    assert var_8 == '{}'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = 'c'
    var_16 = 'd'
    var_17 = {var_15: var_16}
    var_18 = {var_9: var_14, var_10: var_17}
    var_19 = module_0.dumps(var_18)
    assert var_19 == '{"a": [1, 2, 3], "b": {"c": "d"}}'
    var_20 = 'dumps'



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test _PDataSerializer protocol loads method'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'false'
    var_14 = module_0.loads(var_13)
    assert var_14 is False
    var_15 = ''
    var_16 = module_0.loads(var_15)
    var_17 = b''
    var_18 = module_0.loads(var_17)
    var_19 = '{"test": "data"}'
    var_20 = module_0.loads(var_19)
    var_21 = '{}'
    var_22 = module_0.loads()



# Parsed testcases at query #35
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"hello"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'hello'
    var_6 = '42'
    var_7 = module_0.loads(var_6)
    assert var_7 == 42
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = 'false'
    var_13 = module_0.loads(var_12)
    assert var_13 is False
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)
    var_16 = b'[1, 2, 3]'
    var_17 = module_0.loads(var_16)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = '"hello"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'hello'
    var_6 = 'true'
    var_7 = module_0.loads(var_6)
    assert var_7 is True
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = '[1, 2, 3]'
    var_11 = module_0.loads(var_10)
    var_12 = '{}'
    var_13 = module_0.loads(var_12)
    var_14 = '[]'
    var_15 = module_0.loads(var_14)
    var_16 = 'a'
    var_17 = 1
    var_18 = 2
    var_19 = 'b'
    var_20 = 'c'
    var_21 = {var_19: var_20}
    var_22 = [var_17, var_18, var_21]
    var_23 = {var_16: var_22}
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.loads(var_24)
    var_26 = 'invalid json'
    var_27 = module_0.loads(var_26)
    var_28 = ''
    var_29 = module_0.loads(var_28)
    var_30 = '{invalid}'
    var_31 = module_0.loads(var_30)
    var_32 = b'{"key": "value"}'
    var_33 = module_0.loads(var_32)
    var_34 = b'42'
    var_35 = module_0.loads(var_34)
    assert var_35 == 42



# Parsed testcases at query #2
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'test_string'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'test_string'
    var_6 = '42'
    var_7 = module_0.loads(var_6)
    assert var_7 == '42'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = '123'
    var_5 = module_0.loads(var_4)
    assert var_5 == 123
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = '[1, 2, 3]'
    var_9 = module_0.loads(var_8)
    var_10 = 'invalid json'
    var_11 = module_0.loads(var_10)
    var_12 = '[]'
    var_13 = module_0.loads(var_12)
    var_14 = 'null'
    var_15 = module_0.loads(var_14)
    assert var_15 is None



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
    var_7 = b'custom-salt'
    var_8 = var_1.iter_unsigners(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'key_derivation'
    var_12 = 'none'
    var_13 = {var_11: var_12}
    var_14 = [var_13]
    var_15 = module_0.Serializer(var_0, fallback_signers=var_14)
    var_16 = var_15.iter_unsigners()
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = var_17[var_5]
    var_20 = 1
    var_21 = var_17[var_20]
    var_22 = 'CustomSigner'
    var_23 = {}
    var_24 = 'digest_method'
    var_25 = 'sha256'
    var_26 = {var_24: var_25}
    var_27 = var_15.iter_unsigners()
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = var_28[var_5]
    var_31 = var_28[var_20]
    var_32 = 'old-key'
    var_33 = 'new-key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Serializer(var_34)
    var_36 = var_35.iter_unsigners()
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = var_35.iter_unsigners()
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = [var_32, var_33]
    var_43 = {var_11: var_12}
    var_44 = [var_43]
    var_45 = module_0.Serializer(var_42, fallback_signers=var_44)
    var_46 = var_45.iter_unsigners()
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 3
    var_49 = var_47[var_5]
    var_50 = var_47[var_20]
    var_51 = 2
    var_52 = var_47[var_51]
    var_53 = b'test-salt'
    var_54 = var_45.iter_unsigners(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = []
    var_58 = module_0.Serializer(var_0, fallback_signers=var_57)
    var_59 = var_58.iter_unsigners()
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 1
    var_62 = b'default-salt'
    var_63 = 'hmac'
    var_64 = {var_11: var_63}
    var_65 = {var_11: var_12}
    var_66 = [var_65]
    var_67 = module_0.Serializer(var_0, var_62, signer_kwargs=var_64, fallback_signers=var_66)
    var_68 = var_67.iter_unsigners()
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 2



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test Serializer.load_payload method.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'test bytes payload'
    var_6 = b'not valid json'
    var_7 = var_2.load_payload(var_6)
    var_8 = b''
    var_9 = var_2.load_payload(var_8)
    var_10 = b'custom'
    var_11 = b'invalid'
    var_12 = var_2.load_payload(var_11)
    var_13 = b'test'



# Parsed testcases at query #7
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = '[1, 2, 3]'
    var_5 = module_0.loads(var_4)
    var_6 = 'null'
    var_7 = module_0.loads(var_6)
    assert var_7 is None
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = b'{"key": "value"}'
    var_11 = module_0.loads(var_10)
    var_12 = 'invalid json'
    var_13 = module_0.loads(var_12)
    var_14 = ''
    var_15 = module_0.loads(var_14)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'hello'
    var_5 = b'world'
    var_6 = module_0.Serializer(var_0)
    var_7 = b'test'
    var_8 = module_0.Serializer(var_0)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = module_0.Serializer(var_9)
    var_12 = b''
    var_13 = var_11.load_payload(var_12)
    var_14 = module_0.Serializer(var_12)
    var_15 = b'\xff\xfe'
    var_16 = var_14.load_payload(var_15)
    var_17 = b'invalid'
    var_18 = var_14.load_payload(var_17)



# Parsed testcases at query #9
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol supports dumps method.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.dumps(var_5)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = 'test'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    var_14 = module_0.dumps(var_12)
    var_15 = 'string'
    var_16 = module_0.dumps(var_15)
    var_17 = module_0.dumps(var_15)
    var_18 = 123
    var_19 = module_0.dumps(var_18)
    var_20 = module_0.dumps(var_18)
    var_21 = True
    var_22 = module_0.dumps(var_21)
    var_23 = True
    var_24 = module_0.dumps(var_23)
    var_25 = None
    var_26 = module_0.dumps(var_25)
    var_27 = module_0.dumps(var_25)
    var_28 = 'data'
    var_29 = {var_28: var_11}
    var_30 = module_0.dumps(var_29)
    assert var_30 == b'{"data": "test"}'



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = b'custom-salt'
    var_8 = var_1.iter_unsigners(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'key'
    var_12 = b'fallback-key'
    var_13 = {var_11: var_12}
    var_14 = [var_13]
    var_15 = module_0.Serializer(var_0, fallback_signers=var_14)
    var_16 = var_15.iter_unsigners()
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_11: var_12}
    var_20 = len(var_17)
    assert var_20 == 2
    var_21 = len(var_17)
    assert var_21 == 2
    var_22 = b'old-key'
    var_23 = b'new-key'
    var_24 = [var_22, var_23]
    var_25 = {var_11: var_12}
    var_26 = [var_25]
    var_27 = module_0.Serializer(var_24, fallback_signers=var_26)
    var_28 = var_27.iter_unsigners()
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 3
    var_31 = b'custom-salt'
    var_32 = {var_11: var_12}
    var_33 = [var_32]
    var_34 = module_0.Serializer(var_0, var_31, fallback_signers=var_33)
    var_35 = b'override-salt'
    var_36 = var_34.iter_unsigners(var_35)
    var_37 = list(var_36)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == b'test_data'
    assert var_3 == 'custom_hello'
    var_4 = b'test_data'
    var_5 = b'hello'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'invalid json'
    var_9 = var_1.load_payload(var_8)
    var_10 = b''
    var_11 = var_1.load_payload(var_10)
    var_12 = b'{"number": 42, "list": [1, 2, 3]}'
    var_13 = var_1.load_payload(var_12)



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
    var_5 = module_1.dumps(var_4)
    var_6 = 'utf-8'
    var_7 = b'test bytes data'
    var_8 = 'hello'
    var_9 = b'invalid json'
    var_10 = var_1.load_payload(var_9)
    var_11 = b'test'



# Parsed testcases at query #13
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
    var_5 = {}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{}'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 'c'
    var_14 = 'd'
    var_15 = {var_13: var_14}
    var_16 = {var_7: var_12, var_8: var_15}
    var_17 = module_0.dumps(var_16)
    assert var_17 == '{"a": [1, 2, 3], "b": {"c": "d"}}'
    var_18 = 42
    var_19 = module_0.dumps(var_18)
    assert var_19 == '42'
    var_20 = 'hello'
    var_21 = module_0.dumps(var_20)
    assert var_21 == '"hello"'
    var_22 = 'two'
    var_23 = [var_9, var_22, var_11]
    var_24 = module_0.dumps(var_23)
    assert var_24 == '[1, "two", 3.0]'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = {var_2: var_3}
    var_7 = module_1.dumps(var_6)
    var_8 = 'test'
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = var_1.dumps(var_8)
    var_12 = var_1.dumps(var_8)
    var_13 = var_1.dumps(var_8)
    var_14 = 123
    var_15 = var_1.dumps(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_1.dumps(var_19)
    var_21 = None
    var_22 = var_1.dumps(var_21)
    var_23 = True
    var_24 = var_1.dumps(var_23)



# Parsed testcases at query #15
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"data": "test"}'
    var_1 = module_0.loads(var_0)
    var_2 = 'hello'
    var_3 = module_0.loads(var_2)
    var_4 = ''
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #16
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method works correctly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = '"hello"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = 'null'
    var_12 = module_0.loads(var_11)
    assert var_12 is None
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)
    var_15 = b'{"key": "value"}'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #17
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method handles payload correctly.'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"a": 1, "b": 2}'
    var_4 = module_0.loads(var_3)
    var_5 = b'invalid'
    var_6 = module_0.loads(var_5)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"numbers": [1, 2, 3], "nested": {"a": 1}}'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'"string"'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'string'
    var_8 = b'42'
    var_9 = var_1.load_payload(var_8)
    assert var_9 == 42
    var_10 = b'true'
    var_11 = var_1.load_payload(var_10)
    assert var_11 is True
    var_12 = b'null'
    var_13 = var_1.load_payload(var_12)
    assert var_13 is None
    var_14 = b'{}'
    var_15 = var_1.load_payload(var_14)
    var_16 = b'[1, 2, 3]'
    var_17 = var_1.load_payload(var_16)
    var_18 = b'invalid json'
    var_19 = var_1.load_payload(var_18)
    var_20 = b''
    var_21 = var_1.load_payload(var_20)
    var_22 = b'\xff\xfe\x00\x00'
    var_23 = var_1.load_payload(var_22)
    var_24 = b'[1, 2, 3]'
    var_25 = b'\x01\x02\x03'
    var_26 = b'{"different": "serializer"}'
    var_27 = b'some data'
    var_28 = b'invalid'
    var_29 = var_1.load_payload(var_28)
    var_30 = '{"unicode": "测试"}'
    var_31 = 'utf-8'
    var_32 = var_1.load_payload(var_27)



# Parsed testcases at query #19
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.loads properly handles the payload parameter.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines dumps method correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 123
    var_6 = module_0.dumps(var_5)
    assert var_6 == '123'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3]'
    var_12 = None
    var_13 = module_0.dumps(var_12)
    assert var_13 == 'null'
    var_14 = True
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'true'
    var_16 = {}
    var_17 = module_0.dumps(var_16)
    assert var_17 == '{}'
    var_18 = []
    var_19 = module_0.dumps(var_18)
    assert var_19 == '[]'
    var_20 = 'test'
    var_21 = module_0.dumps(var_20)
    assert var_21 == '"test"'



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
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = module_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = 'data'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = var_1.dumps(var_14, var_10)
    var_16 = {var_12: var_13}
    var_17 = var_11.dumps(var_16)
    var_18 = {}
    var_19 = var_1.dumps(var_18)
    var_20 = var_1.loads(var_19)
    var_21 = None
    var_22 = var_1.dumps(var_21)
    var_23 = var_1.loads(var_22)
    assert var_23 is None
    var_24 = 2
    var_25 = 3
    var_26 = [var_13, var_24, var_25]
    var_27 = var_1.dumps(var_26)
    var_28 = var_1.loads(var_27)
    var_29 = 'nested'
    var_30 = 'list'
    var_31 = [var_13, var_24, var_25]
    var_32 = {var_30: var_31}
    var_33 = {var_29: var_32}
    var_34 = var_1.dumps(var_33)
    var_35 = var_1.loads(var_34)
    var_36 = 'test'
    var_37 = {var_36: var_12}
    var_38 = var_1.dumps(var_37)
    var_39 = {var_36: var_12}
    var_40 = var_1.dumps(var_39)
    var_41 = 'different-secret-key'
    var_42 = module_0.Serializer(var_41)
    var_43 = {var_36: var_12}
    var_44 = var_1.dumps(var_43)
    var_45 = {var_36: var_12}
    var_46 = var_42.dumps(var_45)
    var_47 = 'sort_keys'
    var_48 = True
    var_49 = {var_47: var_48}
    var_50 = 'b'
    var_51 = 'a'
    var_52 = {var_50: var_48, var_51: var_24}
    var_53 = module_1.dumps(var_52)
    var_54 = module_1.loads(var_53)



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol has a dumps method that works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = module_0.dumps(var_5)
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = None
    var_6 = module_0.dumps(var_5)
    assert var_6 == 'null'
    var_7 = True
    var_8 = module_0.dumps(var_7)
    assert var_8 == 'true'
    var_9 = 42
    var_10 = module_0.dumps(var_9)
    assert var_10 == '42'
    var_11 = 2
    var_12 = 3
    var_13 = [var_7, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = [var_11, var_12]
    var_19 = 'd'
    var_20 = 'e'
    var_21 = {var_19: var_20}
    var_22 = {var_15: var_7, var_16: var_18, var_17: var_21}
    var_23 = module_0.dumps(var_22)
    var_24 = module_0.loads(var_23)



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '"test"'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'test'
    var_5 = '123'
    var_6 = module_0.loads(var_5)
    assert var_6 == 123
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = '{}'
    var_14 = module_0.loads(var_13)
    var_15 = '[]'
    var_16 = module_0.loads(var_15)
    var_17 = 'invalid json'
    var_18 = module_0.loads(var_17)
    var_19 = b'{"key": "value"}'
    var_20 = module_0.loads(var_19)
    var_21 = module_0.loads(var_3)



# Parsed testcases at query #26
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = '"string"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'string'
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = 'null'
    var_12 = module_0.loads(var_11)
    assert var_12 is None
    var_13 = b'{"key": "value"}'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid json'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps returns a signed string serialized with the internal serializer.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = var_2.loads(var_6)
    var_8 = {var_3: var_4}
    var_9 = module_1.dumps(var_8)
    var_10 = module_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = module_0.Serializer(var_1, var_11)
    var_13 = {var_3: var_4}
    var_14 = var_12.dumps(var_13)
    var_15 = var_12.loads(var_14, var_11)
    var_16 = 'sort_keys'
    var_17 = True
    var_18 = {var_16: var_17}
    var_19 = module_0.Serializer(var_1, serializer_kwargs=var_18)
    var_20 = 'b'
    var_21 = 'a'
    var_22 = 2
    var_23 = {var_20: var_22, var_21: var_17}
    var_24 = var_19.dumps(var_23)
    var_25 = var_19.loads(var_24)
    var_26 = module_0.Serializer(var_1)
    var_27 = {}
    var_28 = var_26.dumps(var_27)
    var_29 = var_26.loads(var_28)
    var_30 = module_0.Serializer(var_1)
    var_31 = 3
    var_32 = [var_17, var_22, var_31]
    var_33 = var_30.dumps(var_32)
    var_34 = var_30.loads(var_33)
    var_35 = module_0.Serializer(var_1)
    var_36 = 'test string'
    var_37 = var_35.dumps(var_36)
    var_38 = var_35.loads(var_37)
    assert var_38 == 'test string'
    var_39 = module_0.Serializer(var_1)
    var_40 = 42
    var_41 = var_39.dumps(var_40)
    var_42 = var_39.loads(var_41)
    assert var_42 == 42
    var_43 = module_0.Serializer(var_1)
    var_44 = None
    var_45 = var_43.dumps(var_44)
    var_46 = var_43.loads(var_45)
    assert var_46 is None
    var_47 = module_0.Serializer(var_1)
    var_48 = var_47.dumps(var_17)
    var_49 = False
    var_50 = var_47.dumps(var_49)
    var_51 = var_47.loads(var_48)
    assert var_51 is True
    var_52 = var_47.loads(var_50)
    assert var_52 is False
    var_53 = 'old-key'
    var_54 = 'new-key'
    var_55 = [var_53, var_54]
    var_56 = module_0.Serializer(var_55)
    var_57 = {var_3: var_4}
    var_58 = var_56.dumps(var_57)
    var_59 = var_56.loads(var_58)
    var_60 = module_0.Serializer(var_54)
    var_61 = var_60.loads(var_58)



# Parsed testcases at query #28
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"



# Parsed testcases at query #29
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol loads method works correctly.'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = b'"string"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'string'
    var_7 = b'42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = b'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = b'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = b'false'
    var_14 = module_0.loads(var_13)
    assert var_14 is False
    var_15 = b'{}'
    var_16 = module_0.loads(var_15)
    var_17 = b'[]'
    var_18 = module_0.loads(var_17)
    var_19 = b'{"a": {"b": [1, 2, 3]}}'
    var_20 = module_0.loads(var_19)
    var_21 = b'"unicode text"'
    var_22 = module_0.loads(var_21)
    assert var_22 == 'unicode text'
    var_23 = b'invalid json'
    var_24 = module_0.loads(var_23)
    var_25 = b''
    var_26 = module_0.loads(var_25)
    var_27 = '{"key": "value"}'
    var_28 = module_0.loads(var_27)
    var_29 = b'test'
    var_30 = module_0.loads()



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'{"test": "data"}'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'\xff\xfe\x00\x00'
    var_11 = var_1.load_payload(var_10)
    var_12 = b'\xff\xfe'
    var_13 = b'not json'
    var_14 = var_1.load_payload(var_13)



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_0.dumps(var_4)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.dumps(var_7)
    var_9 = module_0.dumps(var_7)
    var_10 = 'utf-8'
    var_11 = module_0.dumps(var_3)
    assert var_11 == 42
    var_12 = 'test'
    var_13 = 'data'
    var_14 = {var_12: var_13}
    var_15 = module_0.dumps(var_14)
    assert var_15 == '{"test": "data"}'



# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
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
    var_6 = None
    var_7 = module_0.dumps(var_6)
    assert var_7 == 'null'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'
    var_15 = 'hello'
    var_16 = module_0.dumps(var_15)
    assert var_16 == '"hello"'



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = 'test_data'
    var_3 = module_0.loads(var_2)
    var_4 = b'binary_data'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'some bytes payload'
    var_5 = b'invalid json'
    var_6 = var_1.load_payload(var_5)
    var_7 = b''
    var_8 = var_1.load_payload(var_7)
    var_9 = b'{"key": "value"}'



# Parsed testcases at query #36
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol correctly defines loads method.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '"hello"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = '{}'
    var_14 = module_0.loads(var_13)
    var_15 = '[]'
    var_16 = module_0.loads(var_15)
    var_17 = 'invalid json'
    var_18 = module_0.loads(var_17)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"number": 42}'
    var_5 = b'{"list": [1, 2, 3]}'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = str(var_7)
    var_9 = b''
    var_10 = var_1.load_payload(var_9)
    var_11 = b'null'
    var_12 = var_1.load_payload(var_11)
    var_13 = b'{"text": "hello"}'
    var_14 = b'{"broken": }'
    var_15 = var_1.load_payload(var_14)
    var_16 = 'nested'
    var_17 = 'list'
    var_18 = 'dict'
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = 'a'
    var_24 = {var_23: var_19}
    var_25 = {var_17: var_22, var_18: var_24}
    var_26 = {var_16: var_25}
    var_27 = module_1.dumps(var_26)
    var_28 = 'utf-8'
    var_29 = var_1.load_payload(var_13)
    var_30 = b'"string"'
    var_31 = 'string'
    var_32 = (var_30, var_31)
    var_33 = b'123'
    var_34 = 123
    var_35 = (var_33, var_34)
    var_36 = b'true'
    var_37 = True
    var_38 = (var_36, var_37)
    var_39 = b'false'
    var_40 = False
    var_41 = (var_39, var_40)
    var_42 = b'null'
    var_43 = None
    var_44 = (var_42, var_43)
    var_45 = b'[1, "two", 3.0]'
    var_46 = 'two'
    var_47 = [var_37, var_46, var_21]
    var_48 = (var_45, var_47)
    var_49 = [var_32, var_35, var_38, var_41, var_44, var_48]
    var_50 = var_1.load_payload(var_13)



# Parsed testcases at query #38
#--------------------------


import json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = b'null'
    var_5 = module_0.loads(var_4)
    assert var_5 is None
    var_6 = b'"string"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'string'
    var_8 = b'invalid'
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #39
#--------------------------


import json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = b'true'
    var_5 = module_0.loads(var_4)
    assert var_5 is True
    var_6 = '{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = '42'
    var_9 = module_0.loads(var_8)
    assert var_9 == 42
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = b'invalid'
    var_13 = module_0.loads(var_12)
    var_14 = b''
    var_15 = module_0.loads(var_14)



# Parsed testcases at query #40
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = '3.14'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid json'
    var_16 = module_0.loads(var_15)



