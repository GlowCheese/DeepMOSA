####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13)
    var_15 = 'other-salt'
    var_16 = module_0.Serializer(var_0, var_15)
    var_17 = {var_2: var_3}
    var_18 = var_16.dumps(var_17)
    var_19 = 'sort_keys'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = module_0.Serializer(var_0, serializer_kwargs=var_21)
    var_23 = 'b'
    var_24 = 'a'
    var_25 = 2
    var_26 = {var_23: var_20, var_24: var_25}
    var_27 = var_22.dumps(var_26)
    var_28 = var_22.loads(var_27)
    var_29 = module_0.Serializer(var_0)
    var_30 = {}
    var_31 = var_29.dumps(var_30)
    var_32 = var_29.loads(var_31)
    var_33 = module_0.Serializer(var_0)
    var_34 = None
    var_35 = var_33.dumps(var_34)
    var_36 = var_33.loads(var_35)
    assert var_36 is None
    var_37 = module_0.Serializer(var_0)
    var_38 = 3
    var_39 = [var_20, var_25, var_38]
    var_40 = var_37.dumps(var_39)
    var_41 = var_37.loads(var_40)
    var_42 = 'old-key'
    var_43 = 'new-key'
    var_44 = [var_42, var_43]
    var_45 = module_0.Serializer(var_44)
    var_46 = {var_2: var_3}
    var_47 = var_45.dumps(var_46)
    var_48 = [var_42]
    var_49 = module_0.Serializer(var_48)
    var_50 = var_49.loads(var_47)
    var_51 = [var_43]
    var_52 = module_0.Serializer(var_51)
    var_53 = var_52.loads(var_47)



# Parsed testcases at query #2
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method is properly defined."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = 42
    var_8 = module_0.dumps(var_7)
    assert var_8 == '42'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == '[1, 2, 3]'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = True
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'true'
    var_18 = {}
    var_19 = module_0.dumps(var_18)
    assert var_19 == '{}'
    var_20 = []
    var_21 = module_0.dumps(var_20)
    assert var_21 == '[]'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
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
    var_9 = 'hello'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"hello"'
    var_11 = None
    var_12 = module_0.dumps(var_11)
    assert var_12 == 'null'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'



# Parsed testcases at query #5
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
    assert var_8 == b'[1, 2, 3]'
    var_9 = None
    var_10 = module_0.dumps(var_9)
    assert var_10 == b'null'
    var_11 = {}
    var_12 = module_0.dumps(var_11)
    assert var_12 == b'{}'



# Parsed testcases at query #6
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
    var_8 = module_0.Serializer(var_0)
    var_9 = 'test'
    var_10 = 'number'
    var_11 = 'data'
    var_12 = 42
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.dumps(var_13)
    var_15 = var_8.loads(var_14)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'test-secret'
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
    var_12 = {var_2: var_3}
    var_13 = var_1.dumps(var_12)
    var_14 = {var_2: var_3}
    var_15 = var_11.dumps(var_14)
    var_16 = 'sort_keys'
    var_17 = True
    var_18 = {var_16: var_17}
    var_19 = 'b'
    var_20 = 'a'
    var_21 = 2
    var_22 = {var_19: var_17, var_20: var_21}
    var_23 = module_1.dumps(var_22)
    var_24 = module_1.loads(var_23)
    var_25 = 'old-key'
    var_26 = 'new-key'
    var_27 = [var_25, var_26]
    var_28 = module_0.Serializer(var_27)
    var_29 = 'test-data'
    var_30 = var_28.dumps(var_29)
    var_31 = var_28.loads(var_30)
    assert var_31 == 'test-data'
    var_32 = 'same-data'
    var_33 = var_1.dumps(var_32)
    var_34 = var_1.dumps(var_32)



# Parsed testcases at query #8
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
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = module_0.Serializer(var_0, serializer_kwargs=var_10)
    var_12 = 'b'
    var_13 = 'a'
    var_14 = 2
    var_15 = {var_12: var_14, var_13: var_9}
    var_16 = var_11.dumps(var_15)
    var_17 = module_0.Serializer(var_0)
    var_18 = 'test'
    var_19 = var_17.dumps(var_18)
    var_20 = 'custom-salt'
    var_21 = var_17.dumps(var_18, var_20)
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Serializer(var_24)
    var_26 = var_25.dumps(var_18)
    var_27 = module_1.dumps(var_18)
    var_28 = module_0.Serializer(var_0)
    var_29 = var_28.dumps(var_18)
    var_30 = None
    var_31 = module_0.Serializer(var_0, var_30)
    var_32 = var_31.dumps(var_18)
    var_33 = 42
    var_34 = 3.14
    var_35 = 'string'
    var_36 = 3
    var_37 = [var_9, var_14, var_36]
    var_38 = 'nested'
    var_39 = 'data'
    var_40 = {var_39: var_3}
    var_41 = {var_38: var_40}
    var_42 = [var_30, var_9, var_33, var_34, var_35, var_37, var_41]
    var_43 = var_28.loads(var_5)



# Parsed testcases at query #9
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"key": "value"}'
    var_4 = module_0.dumps(var_2)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #10
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
    var_7 = b'{"key": "value"}'
    var_8 = b'invalid json'
    var_9 = var_1.load_payload(var_8)
    var_10 = b'some payload'
    var_11 = b'binary data'
    var_12 = 'text data'



# Parsed testcases at query #11
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol has dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 42
    var_6 = module_0.dumps(var_5)
    assert var_6 == '42'
    var_7 = 'hello'
    var_8 = module_0.dumps(var_7)
    assert var_8 == '"hello"'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == '[1, 2, 3]'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'



# Parsed testcases at query #12
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
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = 'invalid json'
    var_13 = module_0.loads(var_12)
    var_14 = ''
    var_15 = module_0.loads(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #13
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'BytesSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, payload: payload
    var_6 = lambda self, obj: want_bytes(json.dumps(obj))
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'StrSerializer'
    var_9 = ()
    var_10 = lambda self, payload: payload
    var_11 = lambda self, obj: json.dumps(obj)
    var_12 = {var_3: var_10, var_4: var_11}
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = module_0.dumps(var_15)
    assert var_16 == b'{"key": "value"}'
    var_17 = {var_13: var_14}
    var_18 = module_0.dumps(var_17)
    assert var_18 == '{"key": "value"}'



# Parsed testcases at query #14
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"string"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'string'
    var_6 = '42'
    var_7 = module_0.loads(var_6)
    assert var_7 == 42
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = 'invalid json'
    var_13 = module_0.loads(var_12)
    var_14 = 'hello'
    var_15 = module_0.loads(var_14)
    assert var_15 == 'HELLO'
    var_16 = b'{"key": "value"}'
    var_17 = module_0.loads(var_16)
    var_18 = '{"test": 123}'
    var_19 = module_0.loads(var_18)
    var_20 = 'test payload'
    var_21 = module_0.loads(var_20)
    assert var_21 == 'test payload'



# Parsed testcases at query #15
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = '"hello"'
    var_12 = module_0.loads(var_11)
    assert var_12 == 'hello'
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #16
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
    var_6 = 123
    var_7 = module_0.dumps(var_6)
    assert var_7 == '123'
    var_8 = 'test'
    var_9 = module_0.dumps(var_8)
    assert var_9 == '"test"'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'
    var_17 = True
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'true'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method with various configurations.'
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
    var_15 = var_12.loads(var_14)
    var_16 = 'different-salt'
    var_17 = module_0.Serializer(var_1, var_16)
    var_18 = {var_3: var_4}
    var_19 = var_17.dumps(var_18)
    var_20 = 'sort_keys'
    var_21 = True
    var_22 = {var_20: var_21}
    var_23 = 'b'
    var_24 = 'a'
    var_25 = 2
    var_26 = {var_23: var_25, var_24: var_21}
    var_27 = module_1.dumps(var_26)
    var_28 = '"a"'
    var_29 = '"b"'
    var_30 = None
    var_31 = False
    var_32 = 42
    var_33 = 3.14
    var_34 = 'string'
    var_35 = 3
    var_36 = [var_21, var_25, var_35]
    var_37 = 'nested'
    var_38 = {var_3: var_4}
    var_39 = {var_37: var_38}
    var_40 = [var_30, var_21, var_31, var_32, var_33, var_34, var_36, var_39]
    var_41 = 'secret-key'
    var_42 = module_0.Serializer(var_41)
    var_43 = var_42.loads(var_6)
    var_44 = 'secret-key-1'
    var_45 = module_0.Serializer(var_44)
    var_46 = 'secret-key-2'
    var_47 = module_0.Serializer(var_46)
    var_48 = 'data'
    var_49 = 'test'
    var_50 = {var_48: var_49}
    var_51 = var_45.dumps(var_50)
    var_52 = var_47.dumps(var_50)
    var_53 = 'old-key'
    var_54 = 'new-key'
    var_55 = [var_53, var_54]
    var_56 = module_0.Serializer(var_55)
    var_57 = {var_3: var_4}
    var_58 = var_56.dumps(var_57)
    var_59 = var_56.loads(var_58)
    var_60 = [var_53]
    var_61 = module_0.Serializer(var_60)
    var_62 = var_61.loads(var_58)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps returns the expected serialized and signed output.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = 'sort_keys'
    var_9 = 'separators'
    var_10 = True
    var_11 = ','
    var_12 = ':'
    var_13 = (var_11, var_12)
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = module_0.Serializer(var_1, serializer_kwargs=var_14)
    var_16 = 'b'
    var_17 = 'a'
    var_18 = 2
    var_19 = {var_16: var_10, var_17: var_18}
    var_20 = var_15.dumps(var_19)
    var_21 = var_2.dumps(var_5)
    var_22 = var_2.loads(var_21)
    var_23 = 'custom-salt'
    var_24 = var_2.dumps(var_5, var_23)
    var_25 = var_2.loads(var_24, var_23)
    var_26 = 'test'
    var_27 = b'bytes value'
    var_28 = {var_26: var_27}
    var_29 = var_2.dumps(var_28)
    var_30 = var_2.loads(var_29)
    var_31 = {}
    var_32 = var_2.dumps(var_31)
    var_33 = var_2.loads(var_32)
    var_34 = 3
    var_35 = [var_10, var_18, var_34]
    var_36 = var_2.dumps(var_35)
    var_37 = var_2.loads(var_36)
    var_38 = 'different-secret-key'
    var_39 = module_0.Serializer(var_38)
    var_40 = var_2.dumps(var_5)
    var_41 = var_39.dumps(var_5)
    var_42 = 'different'
    var_43 = 'data'
    var_44 = {var_42: var_43}
    var_45 = var_2.dumps(var_44)
    var_46 = None
    var_47 = var_2.dumps(var_46)
    var_48 = var_2.loads(var_47)
    assert var_48 is None
    var_49 = 'test-key'
    var_50 = module_1.dumps(var_5)
    var_51 = 'simple string'
    var_52 = var_2.dumps(var_51)
    var_53 = 123
    var_54 = var_2.dumps(var_53)
    var_55 = var_2.dumps(var_10)
    var_56 = 'nested'
    var_57 = {var_56: var_4}
    var_58 = [var_57]
    var_59 = var_2.dumps(var_58)



# Parsed testcases at query #19
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
    var_6 = {}
    var_7 = module_0.dumps(var_6)
    assert var_7 == '{}'
    var_8 = None
    var_9 = module_0.dumps(var_8)
    assert var_9 == 'null'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'



# Parsed testcases at query #20
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
    var_7 = b'{"key": "value"}'
    var_8 = module_0.loads(var_7)
    var_9 = b'"test"'
    var_10 = module_0.loads(var_9)
    assert var_10 == 'test'
    var_11 = b'123'
    var_12 = module_0.loads(var_11)
    assert var_12 == 123
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)
    var_15 = b'invalid json'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #21
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    var_2 = 'JSONSerializer'
    var_3 = ()
    var_4 = 'loads'
    var_5 = 'dumps'
    var_6 = '{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = b'{"number": 42}'
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #22
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
    var_11 = var_9[var_5]
    var_12 = 'digest_method'
    var_13 = 'sha256'
    var_14 = {var_12: var_13}
    var_15 = [var_14]
    var_16 = module_0.Serializer(var_0, fallback_signers=var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_18[var_5]
    var_21 = 1
    var_22 = var_18[var_21]
    var_23 = 'sha512'
    var_24 = {var_12: var_23}
    var_25 = len(var_18)
    assert var_25 == 2
    var_26 = var_18[var_5]
    var_27 = var_18[var_21]
    var_28 = len(var_18)
    assert var_28 == 2
    var_29 = var_18[var_5]
    var_30 = var_18[var_21]
    var_31 = 'old-key'
    var_32 = 'new-key'
    var_33 = [var_31, var_32]
    var_34 = {var_12: var_13}
    var_35 = [var_34]
    var_36 = module_0.Serializer(var_33, fallback_signers=var_35)
    var_37 = var_36.iter_unsigners()
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = var_38[var_5]
    var_41 = var_38[var_21]
    var_42 = 2
    var_43 = var_38[var_42]
    var_44 = 'key1'
    var_45 = 'key2'
    var_46 = [var_44, var_45]
    var_47 = len(var_38)
    assert var_47 == 3
    var_48 = var_38[var_5]
    var_49 = var_38[var_21]
    var_50 = var_38[var_42]
    var_51 = module_0.Serializer(var_0)
    var_52 = var_51.iter_unsigners()
    var_53 = '__next__'
    var_54 = hasattr(var_52, var_53)
    var_55 = '__iter__'
    var_56 = hasattr(var_52, var_55)



# Parsed testcases at query #23
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
    var_7 = 'digest_method'
    var_8 = 'sha256'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = module_0.Serializer(var_0, fallback_signers=var_10)
    var_12 = var_11.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_13[var_5]
    var_16 = 1
    var_17 = var_13[var_16]
    var_18 = 'key_derivation'
    var_19 = 'hmac'
    var_20 = {var_18: var_19}
    var_21 = var_11.iter_unsigners()
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_22[var_16]
    var_25 = var_11.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_26[var_16]
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = {var_7: var_8}
    var_33 = [var_32]
    var_34 = module_0.Serializer(var_31, fallback_signers=var_33)
    var_35 = var_34.iter_unsigners()
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 3
    var_38 = b'custom-salt'
    var_39 = module_0.Serializer(var_0, var_38)
    var_40 = var_39.iter_unsigners()
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = module_0.Serializer(var_0)
    var_44 = b'override-salt'
    var_45 = var_43.iter_unsigners(var_44)
    var_46 = list(var_45)
    var_47 = []
    var_48 = module_0.Serializer(var_0, fallback_signers=var_47)
    var_49 = var_48.iter_unsigners()
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = {var_7: var_8}
    var_53 = {var_18: var_19}
    var_54 = var_48.iter_unsigners()
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 4



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'test_bytes'
    var_6 = b'test_text'
    var_7 = b'invalid json'
    var_8 = var_2.load_payload(var_7)
    var_9 = b''
    var_10 = var_2.load_payload(var_9)
    var_11 = b'override_test'
    var_12 = b'override_text_test'
    var_13 = 'numbers'
    var_14 = 'nested'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = 'key'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = {var_13: var_18, var_14: var_21}
    var_23 = module_1.dumps(var_22)
    var_24 = 'utf-8'
    var_25 = b'\x80\x81\x82'
    var_26 = var_2.load_payload(var_25)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = module_1.loads(var_1)
    var_3 = '42'
    var_4 = module_1.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_1.loads(var_5)
    var_7 = 'null'
    var_8 = module_1.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_1.loads(var_9)
    assert var_10 is True
    var_11 = '"hello"'
    var_12 = module_1.loads(var_11)
    assert var_12 == 'hello'
    var_13 = 'invalid json'
    var_14 = module_1.loads(var_13)
    var_15 = b'{"key": "value"}'
    var_16 = module_1.loads(var_15)



# Parsed testcases at query #26
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines loads method correctly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '"string"'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'string'
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = b'{"key": "value"}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #27
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer dumps method returns correct type.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b"{'key': 'value'}"



# Parsed testcases at query #28
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'invalid json'
    var_5 = module_0.loads(var_4)
    var_6 = b'invalid json'
    var_7 = module_0.loads(var_6)
    var_8 = ''
    var_9 = module_0.loads(var_8)
    var_10 = 'nested'
    var_11 = 'list'
    var_12 = 'bool'
    var_13 = 'null'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = True
    var_19 = None
    var_20 = {var_11: var_17, var_12: var_18, var_13: var_19}
    var_21 = {var_10: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = module_0.loads(var_22)
    var_24 = module_0.dumps(var_21)



# Parsed testcases at query #29
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = b'world'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'WORLD'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = 'test'
    var_7 = module_0.loads(var_6)
    var_8 = ''
    var_9 = module_0.loads(var_8)
    assert var_9 == ''
    var_10 = '123'
    var_11 = module_0.loads(var_10)
    assert var_11 == '123'



# Parsed testcases at query #30
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
    var_6 = '123'
    var_7 = module_0.loads(var_6)
    assert var_7 == 123
    var_8 = '0'
    var_9 = module_0.loads(var_8)
    assert var_9 == 0



# Parsed testcases at query #31
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
    var_13 = 'test'
    var_14 = module_0.dumps(var_13)
    assert var_14 == '"test"'
    var_15 = 42
    var_16 = module_0.dumps(var_15)
    assert var_16 == '42'



# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'test data'
    assert var_3 == 'HELLO'
    assert var_3 == 'unicode text'
    var_4 = b'test data'
    var_5 = b'hello'
    var_6 = b'not valid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'unicode text'
    var_11 = b'\xff\xfe'
    var_12 = var_1.load_payload(var_11)



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test _PDataSerializer loads method protocol.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"number": 42}'
    var_4 = module_0.loads(var_3)
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = 'hello'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'HELLO'
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = '42'
    var_14 = module_0.loads(var_13)
    assert var_14 == 42



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method can be implemented."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #37
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines dumps method correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b"{'key': 'value'}"



# Parsed testcases at query #38
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol loads method works correctly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = 'invalid'
    var_4 = module_0.loads(var_3)
    var_5 = b'{"key": "value"}'
    var_6 = module_0.loads(var_5)



# Parsed testcases at query #39
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
    var_6 = 'eyJrZXkiOiAidmFsdWUifQ'
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = 'sort_keys'
    var_10 = 'separators'
    var_11 = True
    var_12 = ','
    var_13 = ':'
    var_14 = (var_12, var_13)
    var_15 = {var_9: var_11, var_10: var_14}
    var_16 = module_0.Serializer(var_0, serializer_kwargs=var_15)
    var_17 = 'b'
    var_18 = 'a'
    var_19 = 2
    var_20 = {var_17: var_19, var_18: var_11}
    var_21 = var_16.dumps(var_20)
    var_22 = module_0.Serializer(var_0)
    var_23 = {var_2: var_3}
    var_24 = 'custom-salt'
    var_25 = var_22.dumps(var_23, var_24)
    var_26 = {var_2: var_3}
    var_27 = 'different-salt'
    var_28 = var_22.dumps(var_26, var_27)
    var_29 = module_0.Serializer(var_0)
    var_30 = 42
    var_31 = var_29.dumps(var_30)
    var_32 = var_29.loads(var_31)
    assert var_32 == 42
    var_33 = module_0.Serializer(var_0)
    var_34 = {}
    var_35 = var_33.dumps(var_34)
    var_36 = var_33.loads(var_35)
    var_37 = module_0.Serializer(var_0)
    var_38 = 3
    var_39 = [var_11, var_19, var_38]
    var_40 = var_37.dumps(var_39)
    var_41 = var_37.loads(var_40)
    var_42 = module_0.Serializer(var_0)
    var_43 = None
    var_44 = var_42.dumps(var_43)
    var_45 = var_42.loads(var_44)
    assert var_45 is None
    var_46 = 'old-key'
    var_47 = 'new-key'
    var_48 = [var_46, var_47]
    var_49 = module_0.Serializer(var_48)
    var_50 = 'test'
    var_51 = var_49.dumps(var_50)
    var_52 = var_49.loads(var_51)
    assert var_52 == 'test'



# Parsed testcases at query #40
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method is properly defined."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = 'string'
    var_6 = module_0.dumps(var_5)
    assert var_6 == 'string'
    var_7 = 123
    var_8 = module_0.dumps(var_7)
    assert var_8 == 123
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    var_14 = {}
    var_15 = module_0.dumps(var_14)
    var_16 = {}
    var_17 = module_0.dumps(var_16)



# Parsed testcases at query #41
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method works correctly with various serializers.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = 'null'
    var_6 = module_0.loads(var_5)
    assert var_6 is None
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = '"hello"'
    var_10 = module_0.loads(var_9)
    assert var_10 == 'hello'
    var_11 = b'{"key": "value"}'
    var_12 = module_0.loads(var_11)
    var_13 = b'42'
    var_14 = module_0.loads(var_13)
    assert var_14 == 42
    var_15 = b'null'
    var_16 = module_0.loads(var_15)
    assert var_16 is None
    var_17 = b'true'
    var_18 = module_0.loads(var_17)
    assert var_18 is True
    var_19 = b'"hello"'
    var_20 = module_0.loads(var_19)
    assert var_20 == 'hello'
    var_21 = module_0.loads(var_1)
    var_22 = module_0.loads(var_11)
    var_23 = 'invalid json'
    var_24 = module_0.loads(var_23)
    var_25 = b'invalid json'
    var_26 = module_0.loads(var_25)



# Parsed testcases at query #42
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
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = module_0.Serializer(var_0, serializer_kwargs=var_10)
    var_12 = 'b'
    var_13 = 'a'
    var_14 = 2
    var_15 = {var_12: var_14, var_13: var_9}
    var_16 = var_11.dumps(var_15)
    var_17 = '{"a":1,"b":2}'
    var_18 = b'salt1'
    var_19 = module_0.Serializer(var_0, var_18)
    var_20 = b'salt2'
    var_21 = module_0.Serializer(var_0, var_20)
    var_22 = {var_2: var_3}
    var_23 = var_19.dumps(var_22)
    var_24 = {var_2: var_3}
    var_25 = var_21.dumps(var_24)
    var_26 = 'old-key'
    var_27 = 'new-key'
    var_28 = [var_26, var_27]
    var_29 = module_0.Serializer(var_28)
    var_30 = {var_2: var_3}
    var_31 = var_29.dumps(var_30)
    var_32 = module_0.Serializer(var_0)
    var_33 = {var_2: var_3}
    var_34 = var_32.dumps(var_33)
    var_35 = b'secret-key'
    var_36 = module_0.Serializer(var_35)
    var_37 = {var_2: var_3}
    var_38 = var_36.dumps(var_37)



# Parsed testcases at query #43
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_18 = module_0.loads(var_17)
    var_19 = '{"text": "hello\\nworld\\t\\u00e9"}'
    var_20 = module_0.loads(var_19)
    var_21 = 'invalid json'
    var_22 = module_0.loads(var_21)
    var_23 = '{broken}'
    var_24 = module_0.loads(var_23)



# Parsed testcases at query #44
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #45
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
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13)
    var_15 = 'other-salt'
    var_16 = module_0.Serializer(var_0, var_15)
    var_17 = {var_2: var_3}
    var_18 = var_16.dumps(var_17)
    var_19 = 'a'
    var_20 = 1
    var_21 = {var_19: var_20}
    var_22 = var_1.dumps(var_21)
    var_23 = 'b'
    var_24 = 2
    var_25 = {var_23: var_24}
    var_26 = var_1.dumps(var_25)
    var_27 = 'sort_keys'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = module_0.Serializer(var_0, serializer_kwargs=var_29)
    var_31 = {var_23: var_28, var_19: var_24}
    var_32 = var_30.dumps(var_31)
    var_33 = var_30.loads(var_32)
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = {var_2: var_3}
    var_39 = var_37.dumps(var_38)
    var_40 = var_37.loads(var_39)
    var_41 = 'verify-key'
    var_42 = module_0.Serializer(var_41)
    var_43 = 'user'
    var_44 = 'test'
    var_45 = {var_43: var_44}
    var_46 = var_42.dumps(var_45)
    var_47 = var_42.loads(var_46)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'invalid json'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #48
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
    var_8 = 42
    var_9 = module_0.dumps(var_8)
    assert var_9 == '42'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = 'nested'
    var_16 = 'list'
    var_17 = 'bool'
    var_18 = [var_10, var_11, var_12]
    var_19 = True
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = {var_15: var_20}
    var_22 = module_0.dumps(var_21)



# Parsed testcases at query #49
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol correctly defines loads method.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = '"text"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'text'
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = 'null'
    var_12 = module_0.loads(var_11)
    assert var_12 is None
    var_13 = '{}'
    var_14 = module_0.loads(var_13)
    var_15 = '{"nested": {"list": [1, 2, 3]}}'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #50
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
    var_6 = var_1.dumps(var_4)
    var_7 = var_1.loads(var_5)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_1.dumps(var_11)
    var_13 = 'custom-salt'
    var_14 = var_1.dumps(var_11, var_13)
    var_15 = var_1.dumps(var_11)
    var_16 = 'sort_keys'
    var_17 = True
    var_18 = {var_16: var_17}
    var_19 = module_0.Serializer(var_0, serializer_kwargs=var_18)
    var_20 = 'b'
    var_21 = 'a'
    var_22 = {var_20: var_9, var_21: var_17}
    var_23 = var_19.dumps(var_22)
    var_24 = b'"a"'
    var_25 = 'old-key'
    var_26 = 'new-key'
    var_27 = [var_25, var_26]
    var_28 = module_0.Serializer(var_27)
    var_29 = 'test'
    var_30 = var_28.dumps(var_29)
    var_31 = var_28.loads(var_30)
    var_32 = module_0.Serializer(var_0)
    var_33 = {}
    var_34 = var_32.dumps(var_33)
    var_35 = var_32.loads(var_34)
    var_36 = None
    var_37 = var_32.dumps(var_36)
    var_38 = var_32.loads(var_37)
    assert var_38 is None
    var_39 = 'two'
    var_40 = [var_17, var_39, var_10]
    var_41 = var_32.dumps(var_40)
    var_42 = var_32.loads(var_41)
    var_43 = 'nested'
    var_44 = 'list'
    var_45 = [var_17, var_9, var_10]
    var_46 = {var_44: var_45}
    var_47 = {var_43: var_46}
    var_48 = var_32.dumps(var_47)
    var_49 = var_32.loads(var_48)
    var_50 = 'test'
    var_51 = 'data'
    var_52 = {var_50: var_51}
    var_53 = module_1.dumps(var_52)



# Parsed testcases at query #51
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method works correctly with different serializers.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = '"string"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'string'
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = '42'
    var_12 = module_0.loads(var_11)
    assert var_12 == 42
    var_13 = 'hello'
    var_14 = module_0.loads(var_13)
    assert var_14 == 'HELLO'
    var_15 = 'test'
    var_16 = module_0.loads(var_15)
    assert var_16 == 'TEST'
    var_17 = b'a,b,c'
    var_18 = module_0.loads(var_17)
    var_19 = b'x,y,z'
    var_20 = module_0.loads(var_19)
    var_21 = module_0.loads(var_13)
    assert var_21 == 'hello'
    var_22 = b'\x00\x01\x02'
    var_23 = module_0.loads(var_22)
    assert var_23 == '000102'
    var_24 = []
    var_25 = 'test_payload'
    var_26 = module_0.loads(var_25)



# Parsed testcases at query #52
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_19 = 'test'
    var_20 = module_0.loads()



# Parsed testcases at query #53
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
    var_12 = var_11.dumps(var_7)
    var_13 = 'sort_keys'
    var_14 = 'separators'
    var_15 = True
    var_16 = ','
    var_17 = ':'
    var_18 = (var_16, var_17)
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = module_0.Serializer(var_0, serializer_kwargs=var_19)
    var_21 = var_20.dumps(var_7)
    var_22 = var_20.loads(var_21)
    var_23 = {}
    var_24 = var_1.dumps(var_23)
    var_25 = var_1.loads(var_24)
    var_26 = 2
    var_27 = 3
    var_28 = [var_15, var_26, var_27]
    var_29 = var_1.dumps(var_28)
    var_30 = var_1.loads(var_29)
    var_31 = 'test string'
    var_32 = var_1.dumps(var_31)
    var_33 = var_1.loads(var_32)
    var_34 = 42
    var_35 = var_1.dumps(var_34)
    var_36 = var_1.loads(var_35)
    var_37 = None
    var_38 = var_1.dumps(var_37)
    var_39 = var_1.loads(var_38)
    assert var_39 is None
    var_40 = var_1.dumps(var_15)
    var_41 = var_1.loads(var_40)
    assert var_41 is True
    var_42 = var_1.dumps(var_7)
    var_43 = var_1.dumps(var_7)
    var_44 = 'different-secret-key'
    var_45 = module_0.Serializer(var_44)
    var_46 = var_45.dumps(var_7)
    var_47 = 'old-key'
    var_48 = 'new-key'
    var_49 = [var_47, var_48]
    var_50 = module_0.Serializer(var_49)
    var_51 = var_50.dumps(var_7)
    var_52 = var_50.loads(var_51)
    var_53 = var_50.loads(var_51)



# Parsed testcases at query #54
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 42
    var_12 = var_1.dumps(var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = var_1.dumps(var_16)
    var_18 = None
    var_19 = var_1.dumps(var_18)
    var_20 = var_1.loads(var_7)



# Parsed testcases at query #55
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = var_1.loads(var_7)
    var_12 = 'sort_keys'
    var_13 = 'separators'
    var_14 = True
    var_15 = ','
    var_16 = ':'
    var_17 = (var_15, var_16)
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = module_0.Serializer(var_0, serializer_kwargs=var_18)
    var_20 = 'b'
    var_21 = 'a'
    var_22 = 2
    var_23 = {var_20: var_22, var_21: var_14}
    var_24 = var_19.dumps(var_23)
    var_25 = var_19.loads(var_24)
    var_26 = 42
    var_27 = var_1.dumps(var_26)
    var_28 = var_1.loads(var_27)
    assert var_28 == 42
    var_29 = 'string'
    var_30 = var_1.dumps(var_29)
    var_31 = var_1.loads(var_30)
    assert var_31 == 'string'
    var_32 = 3
    var_33 = [var_14, var_22, var_32]
    var_34 = var_1.dumps(var_33)
    var_35 = var_1.loads(var_34)
    var_36 = None
    var_37 = var_1.dumps(var_36)
    var_38 = var_1.loads(var_37)
    assert var_38 is None



# Parsed testcases at query #56
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps returns the expected type.'
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
    var_9 = 42
    var_10 = module_0.dumps(var_9)
    assert var_10 == '42'
    var_11 = 'hello'
    var_12 = module_0.dumps(var_11)
    assert var_12 == '"hello"'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.dumps(var_16)
    assert var_17 == '[1, 2, 3]'



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'some_bytes'
    var_5 = b'invalid json'
    var_6 = var_1.load_payload(var_5)
    var_7 = b'test'
    var_8 = b'{"text": "hello"}'
    var_9 = b''
    var_10 = var_1.load_payload(var_9)
    var_11 = b'{"custom": "data"}'



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dumps(var_3)
    var_5 = var_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = var_0.loads(var_6)
    var_8 = '{"key": "value"}'
    var_9 = var_0.loads(var_8)
    var_10 = '{}'
    var_11 = var_0.loads(var_10)
    var_12 = b'{}'
    var_13 = var_0.loads(var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = var_0.dumps(var_17)
    var_19 = var_0.loads(var_18)
    var_20 = 'null'
    var_21 = var_0.loads(var_20)
    assert var_21 is None
    var_22 = b'null'
    var_23 = var_0.loads(var_22)
    assert var_23 is None
    var_24 = 'text'
    var_25 = 'hello\nworld\t!'
    var_26 = {var_24: var_25}
    var_27 = var_0.dumps(var_26)
    var_28 = var_0.loads(var_27)
    var_29 = 'unicode'
    var_30 = 'üñîçødé'
    var_31 = {var_29: var_30}
    var_32 = var_0.dumps(var_31)
    var_33 = var_0.loads(var_32)



# Parsed testcases at query #59
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '123'
    var_3 = module_0.loads(var_2)
    assert var_3 == 123
    var_4 = '"string"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'string'
    var_6 = 'null'
    var_7 = module_0.loads(var_6)
    assert var_7 is None
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'false'
    var_11 = module_0.loads(var_10)
    assert var_11 is False
    var_12 = '[1, 2, 3]'
    var_13 = module_0.loads(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)
    var_16 = b'123'
    var_17 = module_0.loads(var_16)
    assert var_17 == 123
    var_18 = b'"string"'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'string'
    var_20 = b'null'
    var_21 = module_0.loads(var_20)
    assert var_21 is None
    var_22 = b'true'
    var_23 = module_0.loads(var_22)
    assert var_23 is True
    var_24 = b'false'
    var_25 = module_0.loads(var_24)
    assert var_25 is False
    var_26 = b'[1, 2, 3]'
    var_27 = module_0.loads(var_26)
    var_28 = 'invalid json'
    var_29 = module_0.loads(var_28)
    var_30 = b'invalid json'
    var_31 = module_0.loads(var_30)



# Parsed testcases at query #60
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '"string"'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'string'
    var_4 = '123'
    var_5 = module_0.loads(var_4)
    assert var_5 == 123
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
    var_14 = '{"a": {"b": [1, 2, {"c": 3}]}}'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 1
    var_18 = 2
    var_19 = 'c'
    var_20 = 3
    var_21 = {var_19: var_20}
    var_22 = [var_17, var_18, var_21]
    var_23 = {var_16: var_22}
    var_24 = {var_15: var_23}
    var_25 = module_0.loads(var_14)
    var_26 = 'invalid json'
    var_27 = module_0.loads(var_26)



# Parsed testcases at query #61
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol provides proper loads method signature.'
    var_1 = '{"key": "value", "num": 42}'
    var_2 = module_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = module_0.loads(var_3)
    var_5 = '"hello"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = 'true'
    var_8 = module_0.loads(var_7)
    assert var_8 is True
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = '42'
    var_12 = module_0.loads(var_11)
    assert var_12 == 42
    var_13 = '{invalid json}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #62
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = {}
    var_5 = module_0.dumps(var_4)
    var_6 = {var_0: var_1}
    var_7 = module_0.dumps(var_6)
    assert var_7 == b'{"key": "value"}'
    var_8 = {}
    var_9 = module_0.dumps(var_8)
    var_10 = None
    var_11 = module_0.dumps(var_10)
    assert var_11 == 'null'
    var_12 = 123
    var_13 = module_0.dumps(var_12)
    assert var_13 == '123'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.dumps(var_17)
    assert var_18 == '[1, 2, 3]'
    var_19 = 'test'
    var_20 = module_0.dumps(var_19)
    var_21 = module_0.dumps(var_19)



# Parsed testcases at query #63
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
    var_10 = b'custom_salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13)
    var_15 = module_0.Serializer(var_0)
    var_16 = {var_2: var_3}
    var_17 = b'call_salt'
    var_18 = var_15.dumps(var_16, var_17)
    var_19 = var_15.loads(var_18, var_17)
    var_20 = {var_2: var_3}
    var_21 = var_15.dumps(var_20)
    var_22 = 'a'
    var_23 = 1
    var_24 = {var_22: var_23}
    var_25 = var_1.dumps(var_24)
    var_26 = {var_22: var_23}
    var_27 = var_1.dumps(var_26)
    var_28 = {}
    var_29 = var_1.dumps(var_28)
    var_30 = var_1.loads(var_29)



# Parsed testcases at query #64
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
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    assert var_4 == b'hello bytes'
    var_5 = b'{"hello": "world"}'
    var_6 = b'custom_data'
    var_7 = module_0.Serializer(var_1)
    var_8 = b'[1, 2, 3]'
    var_9 = b'invalid json'
    var_10 = var_2.load_payload(var_9)
    var_11 = b'\xff\xfe\xff'
    var_12 = var_2.load_payload(var_11)
    var_13 = b''
    var_14 = var_2.load_payload(var_13)
    var_15 = b'hello bytes'
    var_16 = module_0.Serializer(var_14)
    var_17 = b'not json'
    var_18 = var_2.load_payload(var_17)



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"number": 42}'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'raw bytes data'
    var_7 = b'hello world'
    var_8 = b'test'
    var_9 = b'invalid json'
    var_10 = var_1.load_payload(var_9)
    var_11 = b''
    var_12 = var_1.load_payload(var_11)
    var_13 = b'\xff\xfe\x00\x01'
    var_14 = var_1.load_payload(var_13)
    var_15 = b'test'



# Parsed testcases at query #67
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    var_6 = 'true'
    var_7 = module_0.loads(var_6)
    var_8 = '{"nested": {"list": [1, 2, 3]}, "value": "test"}'
    var_9 = module_0.loads(var_8)
    var_10 = 'null'
    var_11 = module_0.loads(var_10)



# Parsed testcases at query #68
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{"key": "value", "number": 42}'
    var_7 = module_0.loads(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = 'hello'
    var_14 = module_0.dumps(var_13)
    assert var_14 == '"hello"'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'



# Parsed testcases at query #69
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = module_0.loads(var_0)
    var_2 = b'some_bytes'
    var_3 = module_0.loads(var_2)
    var_4 = 'test'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #70
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = 42
    var_8 = module_0.dumps(var_7)
    assert var_8 == '42'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == '[1, 2, 3]'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = {}
    var_17 = module_0.dumps(var_16)
    assert var_17 == '{}'
    var_18 = []
    var_19 = module_0.dumps(var_18)
    assert var_19 == '[]'



# Parsed testcases at query #71
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
    var_12 = 'custom-salt'
    var_13 = var_10.dumps(var_11, var_12)
    var_14 = var_10.loads(var_13, var_12)
    var_15 = 'sort_keys'
    var_16 = True
    var_17 = {var_15: var_16}
    var_18 = module_0.Serializer(var_0, serializer_kwargs=var_17)
    var_19 = 'b'
    var_20 = 'a'
    var_21 = 2
    var_22 = {var_19: var_21, var_20: var_16}
    var_23 = var_18.dumps(var_22)
    var_24 = var_18.loads(var_23)
    var_25 = 'secret-key-1'
    var_26 = module_0.Serializer(var_25)
    var_27 = 'secret-key-2'
    var_28 = module_0.Serializer(var_27)
    var_29 = 'test'
    var_30 = 'data'
    var_31 = {var_29: var_30}
    var_32 = var_26.dumps(var_31)
    var_33 = var_28.dumps(var_31)
    var_34 = module_0.Serializer(var_0)
    var_35 = 'salt-1'
    var_36 = var_34.dumps(var_31, var_35)
    var_37 = 'salt-2'
    var_38 = var_34.dumps(var_31, var_37)
    var_39 = module_0.Serializer(var_0)
    var_40 = {}
    var_41 = var_39.dumps(var_40)
    var_42 = var_39.loads(var_41)
    var_43 = 'old-key'
    var_44 = 'new-key'
    var_45 = [var_43, var_44]
    var_46 = module_0.Serializer(var_45)
    var_47 = {var_30: var_29}
    var_48 = var_46.dumps(var_47)
    var_49 = var_46.loads(var_48)



# Parsed testcases at query #72
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
    var_8 = 'test'
    var_9 = var_1.dumps(var_8)
    var_10 = 'custom-salt'
    var_11 = var_1.dumps(var_8, var_10)
    var_12 = var_1.dumps(var_8)
    var_13 = 'nested'
    var_14 = 'bool'
    var_15 = 'list'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_15: var_19}
    var_21 = True
    var_22 = {var_13: var_20, var_14: var_21}
    var_23 = var_1.dumps(var_22)
    var_24 = var_1.loads(var_23)
    var_25 = 'sort_keys'
    var_26 = 'separators'
    var_27 = True
    var_28 = ','
    var_29 = ':'
    var_30 = (var_28, var_29)
    var_31 = {var_25: var_27, var_26: var_30}
    var_32 = module_0.Serializer(var_0, serializer_kwargs=var_31)
    var_33 = 'b'
    var_34 = 'a'
    var_35 = {var_33: var_17, var_34: var_27}
    var_36 = var_32.dumps(var_35)
    var_37 = var_32.loads(var_36)
    var_38 = 'old-key'
    var_39 = 'new-key'
    var_40 = [var_38, var_39]
    var_41 = module_0.Serializer(var_40)
    var_42 = var_41.dumps(var_8)
    var_43 = var_41.loads(var_42)
    assert var_43 == 'test'



# Parsed testcases at query #73
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '1|2'
    var_6 = module_0.loads(var_5)
    var_7 = 'loads'
    var_8 = None



# Parsed testcases at query #74
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol handles loads method correctly.'
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
    var_13 = '{}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #75
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
    var_19 = 1
    var_20 = var_17[var_19]
    var_21 = 'key_derivation'
    var_22 = 'hmac'
    var_23 = {var_21: var_22}
    var_24 = var_15.iter_unsigners()
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = var_25[var_19]
    var_28 = var_15.iter_unsigners()
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = var_29[var_19]
    var_32 = 'old-key'
    var_33 = 'new-key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Serializer(var_34)
    var_36 = var_35.iter_unsigners()
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = [var_32, var_33]
    var_40 = {var_11: var_12}
    var_41 = [var_40]
    var_42 = module_0.Serializer(var_39, fallback_signers=var_41)
    var_43 = var_42.iter_unsigners()
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 3
    var_46 = var_44[var_5]
    var_47 = var_44[var_19]
    var_48 = 2
    var_49 = var_44[var_48]
    var_50 = b'default-salt'
    var_51 = module_0.Serializer(var_0, var_50)
    var_52 = b'override-salt'
    var_53 = var_51.iter_unsigners(var_52)
    var_54 = list(var_53)
    var_55 = []
    var_56 = module_0.Serializer(var_0, fallback_signers=var_55)
    var_57 = var_56.iter_unsigners()
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = None
    var_61 = module_0.Serializer(var_0, var_60)
    var_62 = var_61.iter_unsigners()
    var_63 = list(var_62)
    var_64 = 'key1'
    var_65 = 'key2'
    var_66 = 'key3'
    var_67 = [var_64, var_65, var_66]
    var_68 = {var_11: var_12}
    var_69 = var_61.iter_unsigners()
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 7
    var_72 = var_70[var_5]
    var_73 = 4
    var_74 = var_70[var_19:var_73]
    var_75 = 7
    var_76 = var_70[var_73:var_75]



# Parsed testcases at query #76
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires a dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'



# Parsed testcases at query #77
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
    var_6 = '"string"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'string'
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = '[1, 2, 3]'
    var_13 = module_0.loads(var_12)



# Parsed testcases at query #78
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = 'false'
    var_12 = module_0.loads(var_11)
    assert var_12 is False
    var_13 = '"hello"'
    var_14 = module_0.loads(var_13)
    assert var_14 == 'hello'
    var_15 = ''
    var_16 = module_0.loads(var_15)
    var_17 = '{invalid}'
    var_18 = module_0.loads(var_17)



# Parsed testcases at query #79
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
    var_21 = {var_2: var_3}
    var_22 = 'custom-salt'
    var_23 = var_1.dumps(var_21, var_22)
    var_24 = None
    var_25 = var_1.dumps(var_24)
    var_26 = 123
    var_27 = var_1.dumps(var_26)
    var_28 = 'string'
    var_29 = var_1.dumps(var_28)



# Parsed testcases at query #80
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"data": "test"}'
    var_3 = module_0.loads(var_2)
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    assert var_5 == 42
    var_6 = 'invalid json'
    var_7 = module_0.loads(var_6)
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = '[]'
    var_11 = module_0.loads(var_10)
    var_12 = '{}'
    var_13 = module_0.loads(var_12)
    var_14 = '"string"'
    var_15 = module_0.loads(var_14)
    assert var_15 == 'string'
    var_16 = '123'
    var_17 = module_0.loads(var_16)
    assert var_17 == 123
    var_18 = 'true'
    var_19 = module_0.loads(var_18)
    assert var_19 is True
    var_20 = 'false'
    var_21 = module_0.loads(var_20)
    assert var_21 is False
    var_22 = '  {"key": "value"}  '
    var_23 = module_0.loads(var_22)



# Parsed testcases at query #81
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = '"test"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'test'
    var_6 = 'null'
    var_7 = module_0.loads(var_6)
    assert var_7 is None
    var_8 = '[1, 2, 3]'
    var_9 = module_0.loads(var_8)
    var_10 = b'{"key": "value"}'
    var_11 = module_0.loads(var_10)
    var_12 = b'42'
    var_13 = module_0.loads(var_12)
    assert var_13 == 42
    var_14 = 'int:42'
    var_15 = module_0.loads(var_14)
    assert var_15 == 42
    var_16 = 'float:3.14'
    var_17 = module_0.loads(var_16)
    var_18 = 'plain text'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'plain text'
    var_20 = 'invalid json'
    var_21 = module_0.loads(var_20)
    var_22 = ''
    var_23 = module_0.loads(var_22)
    var_24 = module_0.loads(var_4)
    assert var_24 == 'test'



# Parsed testcases at query #82
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method works correctly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = '"hello"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = 'null'
    var_12 = module_0.loads(var_11)
    assert var_12 is None
    var_13 = '{"outer": {"inner": [1, 2, 3]}}'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid json'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == b'binary_data'
    var_4 = b'test_data'
    var_5 = b'invalid json'
    var_6 = var_1.load_payload(var_5)
    var_7 = b'test'
    var_8 = b'{"test": 1}'
    var_9 = b'binary_data'



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'test data'
    assert var_3 == 'HELLO'
    assert var_3 == b'\x00\x01\x02'
    var_4 = b'test data'
    var_5 = b'test'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'\xff\xfe'
    var_11 = var_1.load_payload(var_10)
    var_12 = b'hello'
    var_13 = b'\x00\x01\x02'



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test iter_unsigners yields signers in correct order and configuration.'
    var_1 = b'test-secret-key'
    var_2 = b'test-salt'
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = module_0.Serializer(var_1, var_2)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = [var_5]
    var_13 = module_0.Serializer(var_1, var_2, fallback_signers=var_12)
    var_14 = var_13.iter_unsigners()
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_13.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 1
    var_21 = var_18[var_20]
    var_22 = var_13.iter_unsigners()
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = var_23[var_20]
    var_26 = var_13.iter_unsigners()
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 4
    var_29 = b'custom-salt'
    var_30 = var_13.iter_unsigners(var_29)
    var_31 = list(var_30)
    var_32 = b'old-key'
    var_33 = b'newer-key'
    var_34 = b'newest-key'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.Serializer(var_35, var_2)
    var_37 = var_36.iter_unsigners()
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = module_0.Serializer(var_1, var_2)
    var_41 = var_40.iter_unsigners()



# Parsed testcases at query #86
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
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = var_1.loads(var_12, var_11)
    var_14 = 'sort_keys'
    var_15 = 'separators'
    var_16 = True
    var_17 = ','
    var_18 = ':'
    var_19 = (var_17, var_18)
    var_20 = {var_14: var_16, var_15: var_19}
    var_21 = module_0.Serializer(var_0, serializer_kwargs=var_20)
    var_22 = 'b'
    var_23 = 'a'
    var_24 = 2
    var_25 = {var_22: var_24, var_23: var_16}
    var_26 = var_21.dumps(var_25)
    var_27 = var_21.loads(var_26)
    var_28 = 'test'
    var_29 = 'data'
    var_30 = {var_28: var_29}
    var_31 = var_1.dumps(var_30)
    var_32 = {var_28: var_29}
    var_33 = var_1.dumps(var_32)



# Parsed testcases at query #87
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol correctly defines the dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'test'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"test"'
    var_12 = 123
    var_13 = module_0.dumps(var_12)
    assert var_13 == '123'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = {}
    var_17 = module_0.dumps(var_16)



# Parsed testcases at query #88
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"



# Parsed testcases at query #89
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
    var_6 = {var_0: var_1}
    var_7 = module_0.dumps(var_6)
    assert var_7 == '{"key": "value"}'
    var_8 = 123
    var_9 = module_0.dumps(var_8)
    assert var_9 == '123'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'
    var_17 = True
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'true'



# Parsed testcases at query #90
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = 'string'
    var_9 = module_0.dumps(var_8)
    assert var_9 == '"string"'
    var_10 = 123
    var_11 = module_0.dumps(var_10)
    assert var_11 == '123'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.dumps(var_15)
    assert var_16 == '[1, 2, 3]'
    var_17 = None
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'null'
    var_19 = {}
    var_20 = module_0.dumps(var_19)
    assert var_20 == '{}'
    var_21 = []
    var_22 = module_0.dumps(var_21)
    assert var_22 == '[]'



# Parsed testcases at query #91
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol is properly implemented by json serializer.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = module_0.loads(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.dumps(var_9)
    var_11 = module_0.loads(var_10)
    var_12 = None
    var_13 = module_0.dumps(var_12)
    var_14 = module_0.loads(var_13)
    assert var_14 is None
    var_15 = {}
    var_16 = module_0.dumps(var_15)
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #92
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"test": "data"}'
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
    var_10 = 'invalid'
    var_11 = module_0.loads(var_10)



# Parsed testcases at query #93
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'HELLO'
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'test data'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'test data'



# Parsed testcases at query #94
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines dumps method correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = b'{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '{"key": "value"}'
    var_4 = var_0.loads(var_3)
    var_5 = b'123'
    var_6 = var_0.loads(var_5)
    var_7 = '123'
    var_8 = var_0.loads(var_7)



# Parsed testcases at query #96
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 42
    var_6 = module_0.dumps(var_5)
    assert var_6 == '42'
    var_7 = 'hello'
    var_8 = module_0.dumps(var_7)
    assert var_8 == '"hello"'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == '[1, 2, 3]'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'



# Parsed testcases at query #97
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'hello'
    var_2 = b'world'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'world'
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    assert var_5 == '{"key": "value"}'
    var_6 = '12345'
    var_7 = module_0.loads(var_6)
    assert var_7 == '12345'
    var_8 = ''
    var_9 = module_0.loads(var_8)
    assert var_9 == ''
    var_10 = b''
    var_11 = module_0.loads(var_10)
    assert var_11 == ''
    var_12 = 'héllo wörld'
    var_13 = module_0.loads(var_12)
    assert var_13 == 'héllo wörld'
    var_14 = b'\xc3\xa9'
    var_15 = module_0.loads(var_14)
    assert var_15 == 'é'
    var_16 = None
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #98
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires a dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #99
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = 'valid'
    var_2 = module_0.loads(var_1)
    var_3 = 'number'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = 'list'
    var_6 = module_0.loads(var_5)
    var_7 = 'none'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'empty'
    var_10 = module_0.loads(var_9)
    assert var_10 == ''
    var_11 = b'test'
    var_12 = module_0.loads(var_11)
    assert var_12 == 'test'
    var_13 = '{"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #100
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '123'
    var_3 = module_0.loads(var_2)
    var_4 = ''
    var_5 = module_0.loads(var_4)
    var_6 = 'test!@#$%^&*()'
    var_7 = module_0.loads(var_6)
    var_8 = 'héllo wörld'
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #101
#--------------------------




# Parsed testcases at query #102
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
    var_12 = b'{"key": "value"}'
    var_13 = module_0.loads(var_12)



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = {var_3: var_4}
    var_8 = module_1.dumps(var_7)
    var_9 = 'sort_keys'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = module_0.Serializer(var_1, serializer_kwargs=var_11)
    var_13 = 'b'
    var_14 = 'a'
    var_15 = 2
    var_16 = {var_13: var_15, var_14: var_10}
    var_17 = var_12.dumps(var_16)
    var_18 = module_0.Serializer(var_1)
    var_19 = {var_3: var_4}
    var_20 = 'custom-salt'
    var_21 = var_18.dumps(var_19, var_20)
    var_22 = {var_3: var_4}
    var_23 = 'different-salt'
    var_24 = var_18.dumps(var_22, var_23)
    var_25 = module_0.Serializer(var_1)
    var_26 = 3
    var_27 = [var_10, var_15, var_26]
    var_28 = var_25.dumps(var_27)
    var_29 = {}
    var_30 = var_25.dumps(var_29)
    var_31 = module_0.Serializer(var_1)
    var_32 = 'test'
    var_33 = 'number'
    var_34 = 'data'
    var_35 = 42
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = var_31.dumps(var_36)
    var_38 = var_31.loads(var_37)



# Parsed testcases at query #104
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"text"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'text'
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
    var_18 = b'"text"'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'text'
    var_20 = 'custom:hello'
    var_21 = module_0.loads(var_20)
    assert var_21 == 'hello'
    var_22 = 'custom:'
    var_23 = module_0.loads(var_22)
    assert var_23 == ''
    var_24 = b'a|b|c'
    var_25 = module_0.loads(var_24)
    var_26 = b''
    var_27 = module_0.loads(var_26)
    var_28 = b'single'
    var_29 = module_0.loads(var_28)



# Parsed testcases at query #105
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
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = 'true'
    var_11 = module_0.loads(var_10)
    assert var_11 is True
    var_12 = 'false'
    var_13 = module_0.loads(var_12)
    assert var_13 is False
    var_14 = '{}'
    var_15 = module_0.loads(var_14)
    var_16 = '[]'
    var_17 = module_0.loads(var_16)
    var_18 = '{"a": {"b": [1, 2, 3]}}'
    var_19 = module_0.loads(var_18)



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = module_0.Serializer(var_0)
    var_5 = b'test'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'{"text": "data"}'
    var_11 = b'not json'
    var_12 = var_1.load_payload(var_11)



# Parsed testcases at query #107
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.dumps(var_0)
    assert var_1 == '42'
    var_2 = 'hello'
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'hello'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)
    assert var_7 == "{'key': 'value'}"
    var_8 = None
    var_9 = module_0.dumps(var_8)
    assert var_9 == 'None'



# Parsed testcases at query #108
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
    var_21 = {var_2: var_3}
    var_22 = 'custom-salt'
    var_23 = var_1.dumps(var_21, var_22)
    var_24 = 'different-secret'
    var_25 = module_0.Serializer(var_24)
    var_26 = {var_2: var_3}
    var_27 = var_25.dumps(var_26)
    var_28 = {}
    var_29 = var_1.dumps(var_28)
    var_30 = None
    var_31 = var_1.dumps(var_30)
    var_32 = 3
    var_33 = [var_10, var_18, var_32]
    var_34 = var_1.dumps(var_33)
    var_35 = [var_10, var_18, var_32]
    var_36 = {var_16: var_35}
    var_37 = {var_17: var_36}
    var_38 = var_1.dumps(var_37)



# Parsed testcases at query #109
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '{"number": 42, "list": [1, 2, 3]}'
    var_4 = module_0.loads(var_3)
    var_5 = '{}'
    var_6 = module_0.loads(var_5)
    var_7 = '"test string"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'test string'
    var_9 = '42'
    var_10 = module_0.loads(var_9)
    assert var_10 == 42
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'null'
    var_14 = module_0.loads(var_13)
    assert var_14 is None



# Parsed testcases at query #110
#--------------------------


import json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'utf-8'



# Parsed testcases at query #111
#--------------------------


def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'string'
    var_9 = 42
    var_10 = None
    var_11 = 'nested'
    var_12 = 'list'
    var_13 = [var_4, var_5]
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = [var_3, var_7, var_8, var_9, var_10, var_15]



# Parsed testcases at query #112
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'string'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"string"'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = True
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'true'



# Parsed testcases at query #113
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
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 1
    var_9 = 2
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '{"a": 1, "b": 2}'
    var_12 = (var_10, var_11)
    var_13 = 3
    var_14 = [var_8, var_9, var_13]
    var_15 = '[1, 2, 3]'
    var_16 = (var_14, var_15)
    var_17 = 'hello'
    var_18 = '"hello"'
    var_19 = (var_17, var_18)
    var_20 = 42
    var_21 = '42'
    var_22 = (var_20, var_21)
    var_23 = 3.14
    var_24 = '3.14'
    var_25 = (var_23, var_24)
    var_26 = True
    var_27 = 'true'
    var_28 = (var_26, var_27)
    var_29 = None
    var_30 = 'null'
    var_31 = (var_29, var_30)
    var_32 = [var_12, var_16, var_19, var_22, var_25, var_28, var_31]



# Parsed testcases at query #114
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test the loads method of _PDataSerializer protocol.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = '"hello"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'false'
    var_14 = module_0.loads(var_13)
    assert var_14 is False
    var_15 = '{}'
    var_16 = module_0.loads(var_15)
    var_17 = '{"a": {"b": [1, 2, 3]}}'
    var_18 = module_0.loads(var_17)
    var_19 = b'{"key": "value"}'
    var_20 = module_0.loads(var_19)



# Parsed testcases at query #115
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 'test'
    var_12 = 'nested'
    var_13 = 123
    var_14 = 'list'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = {var_11: var_13, var_12: var_19}
    var_21 = var_1.dumps(var_20)
    var_22 = var_1.loads(var_21)
    var_23 = 'sort_keys'
    var_24 = True
    var_25 = {var_23: var_24}
    var_26 = module_0.Serializer(var_0, serializer_kwargs=var_25)
    var_27 = 'b'
    var_28 = 'a'
    var_29 = {var_27: var_24, var_28: var_16}
    var_30 = var_26.dumps(var_29)
    var_31 = {var_27: var_24, var_28: var_16}
    var_32 = var_1.dumps(var_31)
    var_33 = 'test'
    var_34 = module_1.dumps(var_33)
    var_35 = isinstance(var_34, var_2)
    var_36 = module_1.loads(var_34)
    assert var_36 == 'test'
    var_37 = None
    var_38 = True
    var_39 = False
    var_40 = 42
    var_41 = 3.14
    var_42 = 'string'
    var_43 = [var_38, var_16, var_17]
    var_44 = {var_28: var_38}
    var_45 = (var_38, var_16)
    var_46 = [var_37, var_38, var_39, var_40, var_41, var_42, var_43, var_44, var_45]
    var_47 = var_1.dumps(var_20)
    var_48 = var_1.loads(var_47)



# Parsed testcases at query #116
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '123'
    var_3 = module_0.loads(var_2)
    assert var_3 == 123
    var_4 = '"text"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'text'
    var_6 = 'null'
    var_7 = module_0.loads(var_6)
    assert var_7 is None
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'false'
    var_11 = module_0.loads(var_10)
    assert var_11 is False
    var_12 = '[1, 2, 3]'
    var_13 = module_0.loads(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)
    var_16 = b'42'
    var_17 = module_0.loads(var_16)
    assert var_17 == 42
    var_18 = b'"hello"'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'hello'
    var_20 = 'invalid json'
    var_21 = module_0.loads(var_20)
    var_22 = b'invalid json'
    var_23 = module_0.loads(var_22)



# Parsed testcases at query #117
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method can be called with various payload types.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"number": 42}'
    var_4 = module_0.loads(var_3)
    var_5 = '{}'
    var_6 = module_0.loads(var_5)
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
    var_9 = '{"nested": {"a": 1, "b": [1, 2]}}'
    var_10 = module_0.loads(var_9)
    var_11 = '{"value": null}'
    var_12 = module_0.loads(var_11)
    var_13 = '{"flag": true, "active": false}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = {var_3: var_4}
    var_8 = module_1.dumps(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_1, var_9)
    var_11 = {var_3: var_4}
    var_12 = 'other-salt'
    var_13 = var_10.dumps(var_11, var_12)
    var_14 = 'sort_keys'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = module_0.Serializer(var_1, serializer_kwargs=var_16)
    var_18 = 'b'
    var_19 = 'a'
    var_20 = 2
    var_21 = {var_18: var_20, var_19: var_15}
    var_22 = var_17.dumps(var_21)
    var_23 = b'a'
    var_24 = {var_3: var_4}
    var_25 = var_2.dumps(var_24)
    var_26 = 'old-key'
    var_27 = 'new-key'
    var_28 = [var_26, var_27]
    var_29 = module_0.Serializer(var_28)
    var_30 = {var_3: var_4}
    var_31 = var_29.dumps(var_30)



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps returns a signed, serialized string.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = var_2.loads(var_6)
    var_9 = 'custom-salt'
    var_10 = var_2.dumps(var_5, var_9)
    var_11 = {}
    var_12 = var_2.dumps(var_11)



# Parsed testcases at query #120
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol loads method works with different serializer implementations.'
    var_1 = 'null'
    var_2 = module_0.loads(var_1)
    assert var_2 is None
    var_3 = 'true'
    var_4 = module_0.loads(var_3)
    assert var_4 is True
    var_5 = 'false'
    var_6 = module_0.loads(var_5)
    assert var_6 is False
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = '3.14'
    var_10 = module_0.loads(var_9)
    var_11 = '"hello"'
    var_12 = module_0.loads(var_11)
    assert var_12 == 'hello'
    var_13 = '{"key": "value"}'
    var_14 = module_0.loads(var_13)
    var_15 = '[1, 2, 3]'
    var_16 = module_0.loads(var_15)
    var_17 = b'null'
    var_18 = module_0.loads(var_17)
    assert var_18 is None
    var_19 = b'true'
    var_20 = module_0.loads(var_19)
    assert var_20 is True
    var_21 = b'false'
    var_22 = module_0.loads(var_21)
    assert var_22 is False
    var_23 = b'42'
    var_24 = module_0.loads(var_23)
    assert var_24 == 42
    var_25 = b'3.14'
    var_26 = module_0.loads(var_25)
    var_27 = b'"hello"'
    var_28 = module_0.loads(var_27)
    assert var_28 == 'hello'
    var_29 = b'{"key": "value"}'
    var_30 = module_0.loads(var_29)
    var_31 = b'[1, 2, 3]'
    var_32 = module_0.loads(var_31)
    var_33 = ''
    var_34 = module_0.loads(var_33)
    assert var_34 == ''
    var_35 = ' '
    var_36 = module_0.loads(var_35)
    assert var_36 == ' '
    var_37 = 'special_chars_!@#$%'
    var_38 = module_0.loads(var_37)
    assert var_38 == 'special_chars_!@#$%'
    var_39 = '{"a": [1, 2, {"b": 3}]}'
    var_40 = module_0.loads(var_39)
    var_41 = '[[1, 2], [3, 4]]'
    var_42 = module_0.loads(var_41)



# Parsed testcases at query #121
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)
    var_15 = b'{"key": "value"}'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #122
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = 'test'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = 'anything'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = module_0.loads(var_10)
    assert var_11 == ''



# Parsed testcases at query #123
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
    var_9 = 'hello'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"hello"'
    var_11 = None
    var_12 = module_0.dumps(var_11)
    assert var_12 == 'null'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'
    var_15 = True
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'true'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = {var_18: var_19}
    var_21 = [var_15, var_5, var_20]
    var_22 = {var_17: var_21}
    var_23 = module_0.dumps(var_22)
    assert var_23 == '{"a": [1, 2, {"b": "c"}]}'



# Parsed testcases at query #124
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
    var_8 = b'{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = '{invalid json}'
    var_11 = module_0.loads(var_10)
    var_12 = 'null'
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = 'true'
    var_15 = module_0.loads(var_14)
    assert var_15 is True
    var_16 = 'false'
    var_17 = module_0.loads(var_16)
    assert var_17 is False



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = '{"key": "value"}'
    var_5 = 'utf-8'
    var_6 = b'test_data'
    var_7 = b'invalid json'
    var_8 = var_1.load_payload(var_7)
    var_9 = var_1.load_payload(var_7)
    var_10 = b'test'
    var_11 = b'hello'



# Parsed testcases at query #126
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires a dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #127
#--------------------------




# Parsed testcases at query #128
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '123'
    var_6 = module_0.loads(var_5)
    assert var_6 == 123
    var_7 = '"string"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'string'
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = '[1, 2, 3]'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps produces a signed serialized string that can be verified.'
    var_1 = b'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = var_2.loads(var_6)
    var_8 = 'list'
    var_9 = 123
    var_10 = True
    var_11 = [var_8, var_9, var_10]
    var_12 = var_2.dumps(var_11)
    var_13 = var_2.loads(var_12)
    var_14 = None
    var_15 = var_2.dumps(var_14)
    var_16 = var_2.loads(var_15)
    assert var_16 is None
    var_17 = b'custom-salt'
    var_18 = var_2.dumps(var_5, var_17)
    var_19 = var_2.loads(var_18, var_17)
    var_20 = var_2.dumps(var_5)
    var_21 = lambda : var_14
    var_22 = module_0.Serializer(var_1, serializer=var_21)
    var_23 = 'test'
    var_24 = 'bytes'
    var_25 = {var_23: var_24}
    var_26 = module_1.dumps(var_25)
    var_27 = module_1.loads(var_26)
    var_28 = b'old-key'
    var_29 = b'newer-key'
    var_30 = b'newest-key'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.Serializer(var_31)
    var_33 = 'rotated'
    var_34 = {var_33: var_10}
    var_35 = var_32.dumps(var_34)
    var_36 = var_32.loads(var_35)



# Parsed testcases at query #130
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
    var_9 = 'test'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"test"'
    var_11 = None
    var_12 = module_0.dumps(var_11)
    assert var_12 == 'null'
    var_13 = 42
    var_14 = module_0.dumps(var_13)
    assert var_14 == '42'
    var_15 = True
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'true'
    var_17 = 'a'
    var_18 = 'c'
    var_19 = 'b'
    var_20 = {var_19: var_6}
    var_21 = [var_15, var_5, var_20]
    var_22 = 'd'
    var_23 = {var_17: var_21, var_18: var_22}
    var_24 = module_0.dumps(var_23)
    var_25 = module_0.dumps(var_23)
    var_26 = module_0.dumps(var_9)
    assert var_26 == b'"test"'



# Parsed testcases at query #131
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '"test"'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'test'
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
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



# Parsed testcases at query #132
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
    assert var_7 == '{}'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = None
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'null'



# Parsed testcases at query #133
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = {}
    var_12 = var_1.dumps(var_11)
    var_13 = 'different-secret-key'
    var_14 = module_0.Serializer(var_13)
    var_15 = {var_2: var_3}
    var_16 = var_14.dumps(var_15)
    var_17 = var_1.loads(var_5)



# Parsed testcases at query #134
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'binary_payload'
    var_5 = module_0.loads(var_4)
    var_6 = 5
    var_7 = module_0.loads(var_6)
    assert var_7 == 10
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.loads(var_11)
    var_13 = ''
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #135
#--------------------------




# Parsed testcases at query #136
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol is structural and can be used with any object\n    that has dumps and loads methods.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = 123
    var_7 = module_0.dumps(var_6)
    assert var_7 == '123'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = None
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'null'
    var_15 = module_0.dumps(var_3)
    var_16 = 'utf-8'



# Parsed testcases at query #137
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
    var_12 = 'false'
    var_13 = module_0.loads(var_12)
    assert var_13 is False
    var_14 = 'invalid json'
    var_15 = module_0.loads(var_14)
    var_16 = ''
    var_17 = module_0.loads(var_16)
    var_18 = 'hello'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'HELLO'
    var_20 = 'test'
    var_21 = 'extra_arg'



# Parsed testcases at query #138
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol requires dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'hello'
    var_6 = b'data'
    var_7 = b'{}'
    var_8 = b'invalid json'
    var_9 = var_2.load_payload(var_8)
    var_10 = b''
    var_11 = var_2.load_payload(var_10)
    var_12 = b'test'



# Parsed testcases at query #140
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
    var_8 = b'{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = b'[1, 2, 3]'
    var_11 = module_0.loads(var_10)
    var_12 = 'bad data'
    var_13 = module_0.loads(var_12)



# Parsed testcases at query #141
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
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = '{}'
    var_13 = module_0.loads(var_12)
    var_14 = '[]'
    var_15 = module_0.loads(var_14)
    var_16 = '{"outer": {"inner": [1, 2, 3]}}'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #142
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = None
    var_11 = module_0.dumps(var_10)
    assert var_11 == 'null'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = True
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'true'



# Parsed testcases at query #143
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
    var_15 = True
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'true'



# Parsed testcases at query #144
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines loads method properly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '42'
    var_6 = module_0.loads(var_5)
    assert var_6 == 42
    var_7 = '"string"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'string'
    var_9 = '[1, 2, 3]'
    var_10 = module_0.loads(var_9)



# Parsed testcases at query #145
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = {}
    var_5 = module_0.dumps(var_4)
    var_6 = 42
    var_7 = module_0.dumps(var_6)
    assert var_7 == '42'
    var_8 = None
    var_9 = module_0.dumps(var_8)
    assert var_9 == 'null'
    var_10 = {var_0: var_1}
    var_11 = module_0.dumps(var_10)
    assert var_11 == b'{"key": "value"}'
    var_12 = {}
    var_13 = module_0.dumps(var_12)
    var_14 = module_0.dumps(var_6)
    assert var_14 == b'42'
    var_15 = module_0.dumps(var_8)
    assert var_15 == b'null'
    var_16 = 'test_data'
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'test_data'



# Parsed testcases at query #146
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'key': 'value'}"



# Parsed testcases at query #147
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
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
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = 'hello'
    var_14 = module_0.dumps(var_13)
    assert var_14 == '"hello"'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'
    var_17 = module_0.dumps(var_5)
    var_18 = module_0.loads(var_17)



# Parsed testcases at query #148
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.dumps(var_0)
    assert var_1 == '42'
    var_2 = 'hello'
    var_3 = module_0.dumps(var_2)
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.dumps(var_7)
    assert var_8 == '[1, 2, 3]'
    var_9 = 'a'
    var_10 = {var_9: var_4}
    var_11 = module_0.dumps(var_10)
    assert var_11 == "{'a': 1}"
    var_12 = {}
    var_13 = module_0.dumps(var_12)



# Parsed testcases at query #149
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
    var_6 = {}
    var_7 = module_0.dumps(var_6)
    assert var_7 == '{}'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3]'
    var_13 = None
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'null'
    var_15 = 'hello'
    var_16 = module_0.dumps(var_15)



# Parsed testcases at query #150
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value", "number": 42}'
    var_6 = module_0.dumps(var_4)
    assert var_6 == b'{"key": "value", "number": 42}'



# Parsed testcases at query #151
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 'test data'
    var_12 = var_1.dumps(var_11)
    var_13 = var_1.dumps(var_11)
    var_14 = 123
    var_15 = var_1.dumps(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_1.dumps(var_19)
    var_21 = None
    var_22 = var_1.dumps(var_21)
    var_23 = 'nested'
    var_24 = 'list'
    var_25 = [var_16, var_17, var_18]
    var_26 = {var_24: var_25}
    var_27 = {var_23: var_26}
    var_28 = var_1.dumps(var_27)
    var_29 = var_1.loads(var_28)
    var_30 = 'indent'
    var_31 = {var_30: var_17}
    var_32 = 'test'
    var_33 = 'data'
    var_34 = {var_32: var_33}
    var_35 = module_1.dumps(var_34)



# Parsed testcases at query #152
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method.'
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
    var_11 = 'sort_keys'
    var_12 = 'separators'
    var_13 = True
    var_14 = ','
    var_15 = ':'
    var_16 = (var_14, var_15)
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = module_0.Serializer(var_1, serializer_kwargs=var_17)
    var_19 = 'b'
    var_20 = 'a'
    var_21 = 2
    var_22 = {var_19: var_21, var_20: var_13}
    var_23 = var_18.dumps(var_22)
    var_24 = var_18.loads(var_23)
    var_25 = 'custom-salt'
    var_26 = module_0.Serializer(var_1, var_25)
    var_27 = {var_3: var_4}
    var_28 = var_26.dumps(var_27)
    var_29 = var_26.loads(var_28)
    var_30 = {var_3: var_4}
    var_31 = 'different-salt'
    var_32 = var_2.dumps(var_30, var_31)
    var_33 = var_2.loads(var_32)
    var_34 = var_2.loads(var_32, var_31)



# Parsed testcases at query #153
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"string"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'string'
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = b'[1, 2, 3]'
    var_9 = module_0.loads(var_8)
    var_10 = b'"string"'
    var_11 = module_0.loads(var_10)
    assert var_11 == 'string'
    var_12 = 'a,b,c'
    var_13 = module_0.loads(var_12)
    var_14 = 'single'
    var_15 = module_0.loads(var_14)
    var_16 = 'null'
    var_17 = module_0.loads(var_16)
    assert var_17 is None
    var_18 = b'null'
    var_19 = module_0.loads(var_18)
    assert var_19 is None



# Parsed testcases at query #154
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
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = '{}'
    var_13 = module_0.loads(var_12)
    var_14 = '{"a": {"b": [1, 2, 3]}}'
    var_15 = module_0.loads(var_14)
    var_16 = b'test_bytes'
    var_17 = module_0.loads(var_16)
    assert var_17 == b'test_bytes'
    var_18 = b'special'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'special_value'
    var_20 = 'test_string'
    var_21 = module_0.loads(var_20)
    assert var_21 == 'test_string'



# Parsed testcases at query #155
#--------------------------




# Parsed testcases at query #156
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'string'
    var_9 = 42
    var_10 = 3.14
    var_11 = True
    var_12 = None
    var_13 = 'nested'
    var_14 = 'data'
    var_15 = [var_11, var_5, var_6]
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = [var_3, var_7, var_8, var_9, var_10, var_11, var_12, var_17]
    var_19 = 'test'
    var_20 = {var_19: var_14}
    var_21 = module_0.dumps(var_20)
    var_22 = 'utf-8'



# Parsed testcases at query #157
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol works with loads method.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = 'null'
    var_6 = module_0.loads(var_5)
    assert var_6 is None
    var_7 = '[1, 2, 3]'
    var_8 = module_0.loads(var_7)
    var_9 = ''
    var_10 = module_0.loads(var_9)
    var_11 = 'invalid'
    var_12 = module_0.loads(var_11)
    var_13 = '{"name": "test", "items": [1, 2, {"nested": True}]}'
    var_14 = module_0.loads(var_13)
    var_15 = 'loads'



# Parsed testcases at query #158
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '"hello"'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'hello'
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



# Parsed testcases at query #159
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'true'
    var_5 = module_0.loads(var_4)
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.loads(var_6)
    assert var_7 is False
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = '"hello"'
    var_11 = module_0.loads(var_10)
    assert var_11 == 'hello'
    var_12 = '123'
    var_13 = module_0.loads(var_12)
    assert var_13 == 123
    var_14 = '3.14'
    var_15 = module_0.loads(var_14)
    var_16 = '[1, 2, 3]'
    var_17 = module_0.loads(var_16)
    var_18 = '{"a": {"b": [1, 2, 3]}}'
    var_19 = module_0.loads(var_18)
    var_20 = '{}'
    var_21 = module_0.loads(var_20)
    var_22 = '{"key": null}'
    var_23 = module_0.loads(var_22)
    var_24 = '{"key": "value with \\"quotes\\""}'
    var_25 = module_0.loads(var_24)



# Parsed testcases at query #160
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
    var_10 = b'{"key": "value"}'
    var_11 = module_0.loads(var_10)
    var_12 = b'[1, 2, 3]'
    var_13 = module_0.loads(var_12)
    var_14 = b'"hello"'
    var_15 = module_0.loads(var_14)
    assert var_15 == 'hello'
    var_16 = b'42'
    var_17 = module_0.loads(var_16)
    assert var_17 == 42
    var_18 = b'null'
    var_19 = module_0.loads(var_18)
    assert var_19 is None



# Parsed testcases at query #161
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == '{"key": "value"}'
    var_4 = 123
    var_5 = module_0.dumps(var_4)
    assert var_5 == '123'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.dumps(var_9)
    assert var_10 == '[1, 2, 3]'
    var_11 = None
    var_12 = module_0.dumps(var_11)
    assert var_12 == 'null'
    var_13 = 'test'
    var_14 = 'nested'
    var_15 = 42
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_17}
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = module_0.dumps(var_19)
    var_21 = module_0.loads(var_20)



# Parsed testcases at query #162
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol supports dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'string'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"string"'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = {}
    var_17 = module_0.dumps(var_16)



# Parsed testcases at query #163
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"test": "data"}'
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"test": "data"}'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'secret-key'
    var_3 = module_0.Serializer(var_2)
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'key3'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.Serializer(var_7)
    var_9 = 'custom-salt'
    var_10 = module_0.Serializer(var_0, var_9)
    var_11 = None
    var_12 = module_0.Serializer(var_0, var_11)
    var_13 = 'CustomSerializer'
    var_14 = ()
    var_15 = 'dumps'
    var_16 = 'loads'
    var_17 = lambda self, obj: str(obj)
    var_18 = lambda self, s: int(s)
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'BytesSerializer'
    var_21 = ()
    var_22 = lambda self, obj: str(obj).encode()
    var_23 = lambda self, s: int(s.decode())
    var_24 = {var_15: var_22, var_16: var_23}
    var_25 = 'key_derivation'
    var_26 = 'hmac'
    var_27 = {var_25: var_26}
    var_28 = module_0.Serializer(var_0, signer_kwargs=var_27)
    var_29 = {var_25: var_26}
    var_30 = [var_29]
    var_31 = module_0.Serializer(var_0, fallback_signers=var_30)
    var_32 = 'sort_keys'
    var_33 = True
    var_34 = {var_32: var_33}
    var_35 = module_0.Serializer(var_0, serializer_kwargs=var_34)
    var_36 = module_0.Serializer(var_0)
    var_37 = module_0.Serializer(var_0)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
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
    var_20 = var_18[var_5]
    var_21 = 1
    var_22 = var_18[var_21]
    var_23 = 'CustomSigner'
    var_24 = {}
    var_25 = {var_12: var_13}
    var_26 = var_16.iter_unsigners()
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = var_27[var_21]
    var_30 = var_16.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = var_31[var_21]
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = [var_34, var_35]
    var_42 = {var_12: var_13}
    var_43 = [var_42]
    var_44 = module_0.Serializer(var_41, fallback_signers=var_43)
    var_45 = var_44.iter_unsigners()
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = None
    var_49 = module_0.Serializer(var_0, var_48)
    var_50 = var_49.iter_unsigners(var_48)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = module_0.Serializer(var_0)
    var_54 = 'override-salt'
    var_55 = var_53.iter_unsigners(var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1



# Parsed testcases at query #3
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value", "number": 42}'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.dumps(var_9)
    assert var_10 == '[1, 2, 3]'
    var_11 = 'hello'
    var_12 = module_0.dumps(var_11)
    assert var_12 == '"hello"'
    var_13 = None
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'null'
    var_15 = True
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'true'



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
    var_6 = {var_2: var_3}
    var_7 = module_1.dumps(var_6)
    var_8 = {var_2: var_3}
    var_9 = b'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 'number'
    var_12 = 42
    var_13 = {var_2: var_3, var_11: var_12}
    var_14 = var_1.dumps(var_13)
    var_15 = var_1.loads(var_14)
    var_16 = 'list'
    var_17 = 'nested'
    var_18 = 'bool'
    var_19 = 'none'
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_25}
    var_27 = True
    var_28 = None
    var_29 = {var_16: var_23, var_17: var_26, var_18: var_27, var_19: var_28}
    var_30 = var_1.dumps(var_29)
    var_31 = var_1.loads(var_30)
    var_32 = 'old-key'
    var_33 = 'new-key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Serializer(var_34)
    var_36 = {var_2: var_3}
    var_37 = var_35.dumps(var_36)
    var_38 = var_35.loads(var_37)



# Parsed testcases at query #5
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_13 = '{"a": {"b": [1, 2]}}'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid json'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #6
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = module_0.loads(var_3)
    var_5 = None
    var_6 = module_0.dumps(var_5)
    assert var_6 == 'null'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3]'
    var_12 = 'string'
    var_13 = module_0.dumps(var_12)
    assert var_13 == '"string"'



# Parsed testcases at query #7
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
    var_6 = 'user_id'
    var_7 = 'name'
    var_8 = 1
    var_9 = 'Alice'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_1.dumps(var_10)
    var_12 = var_1.loads(var_11)
    var_13 = {var_2: var_3}
    var_14 = module_1.dumps(var_13)
    var_15 = 'number'
    var_16 = 42
    var_17 = {var_15: var_16}
    var_18 = module_1.dumps(var_17)
    var_19 = module_1.loads(var_18)
    var_20 = 'custom-salt'
    var_21 = module_0.Serializer(var_0, var_20)
    var_22 = {var_2: var_3}
    var_23 = var_21.dumps(var_22)
    var_24 = 'salt1'
    var_25 = module_0.Serializer(var_0, var_24)
    var_26 = 'salt2'
    var_27 = module_0.Serializer(var_0, var_26)
    var_28 = 'test'
    var_29 = 'data'
    var_30 = {var_28: var_29}
    var_31 = var_25.dumps(var_30)
    var_32 = var_27.dumps(var_30)
    var_33 = 'sort_keys'
    var_34 = 'indent'
    var_35 = True
    var_36 = 2
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = module_0.Serializer(var_0, serializer_kwargs=var_37)
    var_39 = 'b'
    var_40 = 'a'
    var_41 = {var_39: var_36, var_40: var_35}
    var_42 = var_38.dumps(var_41)
    var_43 = 0
    var_44 = '.'
    var_45 = result_kwargs.rsplit(var_44, var_35)[var_43]
    var_46 = '=='
    var_47 = var_45 + var_46
    var_48 = 'old-key'
    var_49 = 'new-key'
    var_50 = [var_48, var_49]
    var_51 = module_0.Serializer(var_50)
    var_52 = 'version'
    var_53 = {var_52: var_35}
    var_54 = var_51.dumps(var_53)
    var_55 = var_51.loads(var_54)
    var_56 = 'secret'
    var_57 = module_0.Serializer(var_56)
    var_58 = var_57.dumps(var_28)
    var_59 = module_1.dumps(var_28)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'hello'
    assert var_3 == 42
    assert var_3 == 'HELLO'
    var_4 = b'hello'
    var_5 = b'42'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'{"key": "value"}'



# Parsed testcases at query #9
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 42
    var_6 = module_0.dumps(var_5)
    assert var_6 == '42'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3]'
    var_12 = None
    var_13 = module_0.dumps(var_12)
    assert var_13 == 'null'
    var_14 = {}
    var_15 = module_0.dumps(var_14)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = var_0.loads(var_3)
    var_5 = '"hello"'
    var_6 = var_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = '42'
    var_8 = var_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'null'
    var_10 = var_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = var_0.loads(var_11)
    assert var_12 is True
    var_13 = 'false'
    var_14 = var_0.loads(var_13)
    assert var_14 is False
    var_15 = module_0._PDataSerializer()
    var_16 = 'hello'
    var_17 = var_15.loads(var_16)
    assert var_17 == 'HELLO'
    var_18 = module_0._PDataSerializer()
    var_19 = 'utf-8'
    var_20 = b'{"key": "value"}'
    var_21 = var_18.loads(var_20)



# Parsed testcases at query #11
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{"key": "value", "number": 42}'
    var_7 = None
    var_8 = module_0.dumps(var_7)
    assert var_8 == 'null'
    var_9 = True
    var_10 = module_0.dumps(var_9)
    assert var_10 == 'true'
    var_11 = 2
    var_12 = 3
    var_13 = [var_9, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'
    var_15 = module_0.loads(var_6)
    var_16 = {}
    var_17 = module_0.dumps(var_16)
    assert var_17 == '{}'
    var_18 = []
    var_19 = module_0.dumps(var_18)
    assert var_19 == '[]'



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
    var_6 = '.'
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = b'.'
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = 'string data'
    var_14 = var_1.dumps(var_13)
    var_15 = 42
    var_16 = var_1.dumps(var_15)
    var_17 = 'test'
    var_18 = 'number'
    var_19 = 'data'
    var_20 = 123
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = var_1.dumps(var_21)
    var_23 = var_1.loads(var_22)
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Serializer(var_26)
    var_28 = {var_2: var_3}
    var_29 = var_27.dumps(var_28)
    var_30 = var_27.loads(var_29)
    var_31 = 'sort_keys'
    var_32 = True
    var_33 = {var_31: var_32}
    var_34 = 'b'
    var_35 = 'a'
    var_36 = 2
    var_37 = {var_34: var_36, var_35: var_32}
    var_38 = module_1.dumps(var_37)
    var_39 = module_1.loads(var_38)



# Parsed testcases at query #13
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
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = var_1.loads(var_12)
    var_14 = 'wrong-salt'
    var_15 = var_1.loads(var_12, var_14)
    var_16 = 'sort_keys'
    var_17 = 'separators'
    var_18 = True
    var_19 = ','
    var_20 = ':'
    var_21 = (var_19, var_20)
    var_22 = {var_16: var_18, var_17: var_21}
    var_23 = module_0.Serializer(var_14, serializer_kwargs=var_22)
    var_24 = 'b'
    var_25 = 'a'
    var_26 = 2
    var_27 = {var_24: var_26, var_25: var_18}
    var_28 = var_23.dumps(var_27)
    var_29 = var_23.loads(var_28)
    var_30 = 'old-key'
    var_31 = 'new-key'
    var_32 = [var_30, var_31]
    var_33 = module_0.Serializer(var_32)
    var_34 = 'data'
    var_35 = 'test'
    var_36 = {var_34: var_35}
    var_37 = var_33.dumps(var_36)
    var_38 = var_33.loads(var_37)
    var_39 = {}
    var_40 = var_1.dumps(var_39)
    var_41 = var_1.loads(var_40)
    var_42 = None
    var_43 = var_1.dumps(var_42)
    var_44 = var_1.loads(var_43)
    assert var_44 is None
    var_45 = 3
    var_46 = [var_18, var_26, var_45]
    var_47 = var_1.dumps(var_46)
    var_48 = var_1.loads(var_47)
    var_49 = {var_15: var_3}
    var_50 = module_1.dumps(var_49)
    var_51 = module_1.loads(var_50)
    var_52 = 'key_derivation'
    var_53 = 'hmac'
    var_54 = {var_52: var_53}
    var_55 = module_0.Serializer(var_14, signer_kwargs=var_54)
    var_56 = {var_15: var_3}
    var_57 = var_55.dumps(var_56)
    var_58 = var_55.loads(var_57)



# Parsed testcases at query #14
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.dumps can be called with an object and returns the expected type.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{"key": "value", "number": 42}'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    assert var_4 == 'olleh'
    var_5 = b'hello'
    var_6 = b'hello'
    var_7 = b'hello'
    var_8 = module_0.Serializer(var_1)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = b''
    var_12 = var_2.load_payload(var_11)
    var_13 = b'\x00\x01\x02'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test loading a JSON payload from text.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test loading a JSON payload from binary with text serializer.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"number": 42}'
    var_4 = var_2.load_payload(var_3)

def test_case_0():
    var_0 = 'Test loading with a custom bytes serializer.'
    var_1 = 'secret-key'
    var_2 = b'hello world'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test loading payload with an explicit serializer parameter.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'test data'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that invalid JSON raises BadPayload.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'invalid json'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that empty payload raises BadPayload.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b''
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that text serializer decodes UTF-8 properly.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = '{"message": "héllo"}'
    var_4 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that non-UTF-8 bytes with text serializer raises BadPayload.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'\xff\xfe\x00\x00'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that the original error is preserved in BadPayload.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'not json'
    var_4 = var_2.load_payload(var_3)

def test_case_0():
    var_0 = 'Test that custom serializer errors are wrapped in BadPayload.'
    var_1 = 'secret-key'
    var_2 = b'some data'

def test_case_0():
    var_0 = 'Test loading with a bytes serializer (non-text).'
    var_1 = 'secret-key'
    var_2 = b'test payload'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = "Test that None serializer parameter uses the instance's serializer."
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that multiple calls to load_payload work correctly.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"a": 1}'
    var_4 = b'{"b": 2}'
    var_5 = var_2.load_payload(var_3)
    var_6 = var_2.load_payload(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test that multiple calls to load_payload work correctly.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"a": 1}'
    var_4 = b'{"b": 2}'
    var_5 = var_2.load_payload(var_3)
    var_6 = var_2.load_payload(var_4)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.serializer as module_0
import re as module_1
import json as module_2
import src.itsdangerous.encoding as module_3

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = '.'
    var_7 = module_1.split(var_6)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = {var_2: var_3}
    var_10 = module_2.dumps(var_9)
    var_11 = {var_2: var_3}
    var_12 = 'custom-salt'
    var_13 = var_1.dumps(var_11, var_12)
    var_14 = 'sort_keys'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = module_0.Serializer(var_0, serializer_kwargs=var_16)
    var_18 = 'b'
    var_19 = 'a'
    var_20 = 2
    var_21 = {var_18: var_15, var_19: var_20}
    var_22 = var_17.dumps(var_21)
    var_23 = {var_19: var_20, var_18: var_15}
    var_24 = module_2.dumps(var_23, sort_keys=var_15)
    var_25 = 'utf-8'
    var_26 = module_3.want_bytes(var_22)
    var_27 = 'test'
    var_28 = var_1.dumps(var_27)
    var_29 = var_1.dumps(var_27)
    var_30 = 'digest_method'
    var_31 = 'sha256'
    var_32 = {var_30: var_31}
    var_33 = [var_32]
    var_34 = module_0.Serializer(var_0, fallback_signers=var_33)
    var_35 = var_34.dumps(var_27)



# Parsed testcases at query #18
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that dumps method works correctly for protocol compliance.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



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
    var_7 = {var_2: var_3}
    var_8 = module_1.dumps(var_7)
    var_9 = module_1.loads(var_8)
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = var_1.loads(var_12, var_11)
    var_14 = 'sort_keys'
    var_15 = 'separators'
    var_16 = True
    var_17 = ','
    var_18 = ':'
    var_19 = (var_17, var_18)
    var_20 = {var_14: var_16, var_15: var_19}
    var_21 = module_0.Serializer(var_0, serializer_kwargs=var_20)
    var_22 = 'b'
    var_23 = 'a'
    var_24 = 2
    var_25 = {var_22: var_24, var_23: var_16}
    var_26 = var_21.dumps(var_25)
    var_27 = '"a"'
    var_28 = '"b"'
    var_29 = {var_2: var_3}
    var_30 = var_1.dumps(var_29)
    var_31 = {var_2: var_3}
    var_32 = var_1.dumps(var_31)
    var_33 = {}
    var_34 = var_1.dumps(var_33)
    var_35 = var_1.loads(var_34)
    var_36 = 3
    var_37 = [var_16, var_24, var_36]
    var_38 = var_1.dumps(var_37)
    var_39 = var_1.loads(var_38)
    var_40 = None
    var_41 = var_1.dumps(var_40)
    var_42 = var_1.loads(var_41)
    assert var_42 is None



# Parsed testcases at query #20
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"string"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'string'
    var_6 = 'null'
    var_7 = module_0.loads(var_6)
    assert var_7 is None
    var_8 = b'{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = b'[1, 2, 3]'
    var_11 = module_0.loads(var_10)
    var_12 = '42'
    var_13 = module_0.loads(var_12)
    assert var_13 == 42
    var_14 = '-10'
    var_15 = module_0.loads(var_14)
    assert var_15 == -10
    var_16 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_17 = module_0.loads(var_16)
    var_18 = 'users'
    var_19 = var_17[var_18]
    var_20 = len(var_19)
    assert var_20 == 2



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
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_4, var_9)
    var_11 = var_1.loads(var_10, var_9)
    var_12 = {}
    var_13 = var_1.dumps(var_12)
    var_14 = var_1.loads(var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = var_1.dumps(var_18)
    var_20 = var_1.loads(var_19)
    var_21 = 'test'
    var_22 = var_1.dumps(var_21)
    var_23 = var_1.loads(var_22)
    assert var_23 == 'test'



# Parsed testcases at query #22
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that loads method works with string input.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '{"key": "value"}'
    var_6 = module_0.loads(var_5)
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = '[1, 2, 3]'
    var_10 = module_0.loads(var_9)



# Parsed testcases at query #23
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
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
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #24
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
    var_6 = '[1, 2, 3]'
    var_7 = module_0.loads(var_6)
    var_8 = b'{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = b'42'
    var_11 = module_0.loads(var_10)
    assert var_11 == 42
    var_12 = b'"hello"'
    var_13 = module_0.loads(var_12)
    assert var_13 == 'hello'
    var_14 = b'[1, 2, 3]'
    var_15 = module_0.loads(var_14)
    var_16 = '{"a": 1}'
    var_17 = module_0.loads(var_16)
    var_18 = b'{"a": 1}'
    var_19 = module_0.loads(var_18)



# Parsed testcases at query #25
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method is callable with appropriate types."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = {var_1: var_2}
    var_8 = module_0.dumps(var_7)
    assert var_8 == '{"key": "value"}'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'hello'
    var_5 = b'test'
    var_6 = module_0.Serializer(var_0)
    var_7 = b'{"a": 1}'
    var_8 = b'invalid json'
    var_9 = var_1.load_payload(var_8)
    var_10 = b''
    var_11 = var_1.load_payload(var_10)
    var_12 = b'test'
    var_13 = b'test'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'not valid json'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'some bytes data'
    var_7 = b'hello world'
    var_8 = b'any data'
    var_9 = b'hello world'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = var_0.loads(var_3)
    var_5 = '"hello"'
    var_6 = var_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = '42'
    var_8 = var_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'null'
    var_10 = var_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = var_0.loads(var_11)
    assert var_12 is True
    var_13 = module_0._PDataSerializer()
    var_14 = b'{"key": "value"}'
    var_15 = var_13.loads(var_14)
    var_16 = '{invalid json}'
    var_17 = var_0.loads(var_16)
    var_18 = ''
    var_19 = var_0.loads(var_18)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'[1, 2, 3]'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'"hello"'
    var_7 = var_1.load_payload(var_6)
    assert var_7 == 'hello'
    var_8 = b'42'
    var_9 = var_1.load_payload(var_8)
    assert var_9 == 42
    var_10 = b'null'
    var_11 = var_1.load_payload(var_10)
    assert var_11 is None
    var_12 = b'true'
    var_13 = var_1.load_payload(var_12)
    assert var_13 is True
    var_14 = b'{invalid json}'
    var_15 = var_1.load_payload(var_14)
    var_16 = b''
    var_17 = var_1.load_payload(var_16)
    var_18 = b'\xff\xfe'
    var_19 = var_1.load_payload(var_18)
    var_20 = b'test data'
    var_21 = b'test data'
    var_22 = b'any data'
    var_23 = b'override test'
    var_24 = b'data'



# Parsed testcases at query #30
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '123'
    var_6 = module_0.loads(var_5)
    assert var_6 == 123
    var_7 = '"hello"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = '[1, 2, 3]'
    var_10 = module_0.loads(var_9)
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'null'
    var_14 = module_0.loads(var_13)
    assert var_14 is None
    var_15 = '{}'
    var_16 = module_0.loads(var_15)
    var_17 = '[]'
    var_18 = module_0.loads(var_17)



# Parsed testcases at query #31
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test the loads method of _PDataSerializer protocol.'
    var_1 = 'hello'
    var_2 = module_0.loads(var_1)
    var_3 = b'world'
    var_4 = module_0.loads(var_3)
    var_5 = 42
    var_6 = module_0.loads(var_5)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = module_0.loads(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.loads(var_14)
    var_16 = None
    var_17 = module_0.loads(var_16)
    var_18 = 'loads'



# Parsed testcases at query #32
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '"hello"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test payload'
    var_5 = b'test bytes'
    var_6 = b'override payload'
    var_7 = b'invalid json'
    var_8 = var_1.load_payload(var_7)
    var_9 = b''
    var_10 = var_1.load_payload(var_9)
    var_11 = b'{"test": 123}'
    var_12 = None
    var_13 = var_1.load_payload(var_11, var_12)



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines dumps method correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = 42
    var_8 = module_0.dumps(var_7)
    assert var_8 == '42'



# Parsed testcases at query #35
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = var_1.loads(var_7)
    var_12 = 'different-secret'
    var_13 = module_0.Serializer(var_12)
    var_14 = {var_2: var_3}
    var_15 = var_13.dumps(var_14)
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Serializer(var_18)
    var_20 = {var_2: var_3}
    var_21 = var_19.dumps(var_20)
    var_22 = var_19.loads(var_21)



# Parsed testcases at query #36
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test the loads method of _PDataSerializer protocol.'
    var_1 = 'hello'
    var_2 = module_0.loads(var_1)
    var_3 = ''
    var_4 = module_0.loads(var_3)
    var_5 = '{"key": "value"}'
    var_6 = module_0.loads(var_5)
    var_7 = '123'
    var_8 = module_0.loads(var_7)
    var_9 = '!@#$%^&*()'
    var_10 = module_0.loads(var_9)
    var_11 = 'héllo wörld'
    var_12 = module_0.loads(var_11)
    var_13 = 'test'
    var_14 = module_0.loads(var_13)
    assert var_14 == 'test'
    var_15 = 'int'
    var_16 = module_0.loads(var_15)
    assert var_16 == 42
    var_17 = 'list'
    var_18 = module_0.loads(var_17)
    var_19 = 'dict'
    var_20 = module_0.loads(var_19)
    var_21 = 'none'
    var_22 = module_0.loads(var_21)
    assert var_22 is None
    var_23 = 'string'
    var_24 = module_0.loads(var_23)
    assert var_24 == 'string'



# Parsed testcases at query #37
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '"hello"'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'hello'
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    assert var_5 == 42
    var_6 = '[1, 2, 3]'
    var_7 = module_0.loads(var_6)
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = '{}'
    var_13 = module_0.loads(var_12)
    var_14 = '[]'
    var_15 = module_0.loads(var_14)
    var_16 = '{"a": {"b": [1, 2, 3]}}'
    var_17 = module_0.loads(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.loads(var_18)



# Parsed testcases at query #38
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'test'
    var_5 = 'extra_arg'
    var_6 = 'hello'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'HELLO'



# Parsed testcases at query #39
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'string'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"string"'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = True
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'true'
    var_18 = False
    var_19 = module_0.dumps(var_18)
    assert var_19 == 'false'
    var_20 = {}
    var_21 = module_0.dumps(var_20)



# Parsed testcases at query #40
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol is properly duck-typed.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #41
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
    var_10 = b'custom-salt'
    var_11 = module_0.Serializer(var_0, var_10)
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13)
    var_15 = {var_2: var_3}
    var_16 = 'different-salt'
    var_17 = var_1.dumps(var_15, var_16)
    var_18 = var_1.loads(var_17, var_16)
    var_19 = 'different-secret'
    var_20 = module_0.Serializer(var_19)
    var_21 = var_20.loads(var_5)
    var_22 = {}
    var_23 = var_1.dumps(var_22)
    var_24 = var_1.loads(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_1.dumps(var_28)
    var_30 = var_1.loads(var_29)
    var_31 = 'a'
    var_32 = 'd'
    var_33 = 'b'
    var_34 = 'c'
    var_35 = {var_33: var_34}
    var_36 = [var_25, var_35]
    var_37 = None
    var_38 = {var_31: var_36, var_32: var_37}
    var_39 = var_1.dumps(var_38)
    var_40 = var_1.loads(var_39)
    var_41 = 'sort_keys'
    var_42 = True
    var_43 = {var_41: var_42}
    var_44 = module_0.Serializer(var_21, serializer_kwargs=var_43)
    var_45 = {var_33: var_42, var_31: var_26}
    var_46 = var_44.dumps(var_45)
    var_47 = b'"a"'
    var_48 = 'old-key'
    var_49 = 'new-key'
    var_50 = [var_48, var_49]
    var_51 = module_0.Serializer(var_50)
    var_52 = 'test'
    var_53 = var_51.dumps(var_52)
    var_54 = var_51.loads(var_53)
    assert var_54 == 'test'
    var_55 = var_1.dumps(var_52)
    var_56 = module_1.dumps(var_52)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    assert var_3 == 'hello'
    assert var_3 == b'test'
    var_4 = b'{"key": "value"}'
    var_5 = b'test'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'{invalid}'
    var_11 = var_1.load_payload(var_10)
    var_12 = b'hello'
    var_13 = 'extra'
    var_14 = 'arg'
    var_15 = {var_13: var_14}



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method with various scenarios.'
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
    var_13 = 'data'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = var_12.dumps(var_15)
    var_17 = module_0.Serializer(var_1)
    var_18 = {var_13: var_14}
    var_19 = var_17.dumps(var_18)
    var_20 = 'sort_keys'
    var_21 = 'separators'
    var_22 = True
    var_23 = ','
    var_24 = ':'
    var_25 = (var_23, var_24)
    var_26 = {var_20: var_22, var_21: var_25}
    var_27 = module_0.Serializer(var_1, serializer_kwargs=var_26)
    var_28 = 'b'
    var_29 = 'a'
    var_30 = 2
    var_31 = {var_28: var_30, var_29: var_22}
    var_32 = var_27.dumps(var_31)
    var_33 = '{"a":1,"b":2}'
    var_34 = 'nested'
    var_35 = 'number'
    var_36 = 'list'
    var_37 = 'bool'
    var_38 = 3
    var_39 = [var_22, var_30, var_38]
    var_40 = True
    var_41 = {var_36: var_39, var_37: var_40}
    var_42 = 42
    var_43 = {var_34: var_41, var_35: var_42}
    var_44 = var_2.dumps(var_43)
    var_45 = var_2.loads(var_44)
    var_46 = {}
    var_47 = var_2.dumps(var_46)
    var_48 = var_2.loads(var_47)
    var_49 = None
    var_50 = {var_4: var_49}
    var_51 = var_2.dumps(var_50)
    var_52 = var_2.loads(var_51)
    var_53 = 'two'
    var_54 = [var_40, var_53, var_38]
    var_55 = var_2.dumps(var_54)
    var_56 = var_2.loads(var_55)
    var_57 = 'secret1'
    var_58 = module_0.Serializer(var_57)
    var_59 = 'secret2'
    var_60 = module_0.Serializer(var_59)
    var_61 = 'test'
    var_62 = {var_61: var_13}
    var_63 = var_58.dumps(var_62)
    var_64 = var_60.dumps(var_62)
    var_65 = 'old_key'
    var_66 = 'new_key'
    var_67 = [var_65, var_66]
    var_68 = module_0.Serializer(var_67)
    var_69 = {var_13: var_61}
    var_70 = var_68.dumps(var_69)
    var_71 = var_68.loads(var_70)
    var_72 = module_0.Serializer(var_3)
    var_73 = var_72.dumps(var_61)
    var_74 = 'test_string'
    var_75 = module_1.dumps(var_74)
    var_76 = module_1.loads(var_75)
    assert var_76 == 'test_string'



# Parsed testcases at query #44
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b'{"key": "value"}'
    var_6 = {var_0: var_1}
    var_7 = module_0.dumps(var_6)
    var_8 = {var_0: var_1}
    var_9 = module_0.dumps(var_8)
    assert var_9 == '{"key": "value"}'
    var_10 = {var_0: var_1}
    var_11 = module_0.dumps(var_10)
    var_12 = {var_0: var_1}
    var_13 = module_0.dumps(var_12)
    assert var_13 == '{"key": "value"}'



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.serializer as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = 'Test Serializer.dumps method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 0
    var_8 = '.'
    var_9 = 1
    var_10 = result.rsplit(var_8, var_9)[var_7]
    var_11 = '=='
    var_12 = var_10 + var_11
    var_13 = module_0.Serializer(var_1)
    var_14 = 'data'
    var_15 = 'custom-salt'
    var_16 = var_13.dumps(var_14, var_15)
    var_17 = 'different-salt'
    var_18 = var_13.dumps(var_14, var_17)
    var_19 = {var_3: var_4}
    var_20 = var_13.dumps(var_19)
    var_21 = 'sort_keys'
    var_22 = 'separators'
    var_23 = True
    var_24 = ','
    var_25 = ':'
    var_26 = (var_24, var_25)
    var_27 = {var_21: var_23, var_22: var_26}
    var_28 = module_0.Serializer(var_1, serializer_kwargs=var_27)
    var_29 = 'b'
    var_30 = 'a'
    var_31 = 2
    var_32 = {var_29: var_31, var_30: var_23}
    var_33 = var_28.dumps(var_32)
    var_34 = result.rsplit(var_8, var_23)[var_7]
    var_35 = var_34 + var_11
    var_36 = 'utf-8'
    var_37 = '"a"'
    var_38 = '"b"'
    var_39 = 'old-key'
    var_40 = 'new-key'
    var_41 = [var_39, var_40]
    var_42 = module_0.Serializer(var_41)
    var_43 = 'test'
    var_44 = var_42.dumps(var_43)
    var_45 = -1
    var_46 = var_42.secret_keys[var_45]
    var_47 = var_42.salt
    var_48 = module_1.Signer(var_46, var_47)
    var_49 = var_42.dump_payload(var_43)
    var_50 = module_2.want_bytes(var_49)
    var_51 = var_48.sign(var_50)
    var_52 = 'utf-8'
    var_53 = 'key_derivation'
    var_54 = 'none'
    var_55 = {var_53: var_54}
    var_56 = var_42.dumps(var_43)
    var_57 = module_0.Serializer(var_1)
    var_58 = var_57.dumps(var_43)
    var_59 = var_57.dumps(var_43)
    var_60 = None
    var_61 = module_0.Serializer(var_1, var_60)
    var_62 = var_61.dumps(var_43)



# Parsed testcases at query #46
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = b'{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = b'[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = b'"hello"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = b'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = b'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = b'false'
    var_14 = module_0.loads(var_13)
    assert var_14 is False
    var_15 = b'3.14'
    var_16 = module_0.loads(var_15)
    var_17 = b'invalid json'
    var_18 = module_0.loads(var_17)
    var_19 = '{"key": "value"}'
    var_20 = module_0.loads(var_19)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test iter_unsigners method of Serializer class.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_1, fallback_signers=var_11)
    var_13 = var_12.iter_unsigners()
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'none'
    var_17 = {var_8: var_16}
    var_18 = var_12.iter_unsigners()
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_12.iter_unsigners()
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'old-secret'
    var_25 = 'new-secret'
    var_26 = [var_24, var_25]
    var_27 = {var_8: var_9}
    var_28 = [var_27]
    var_29 = module_0.Serializer(var_26, fallback_signers=var_28)
    var_30 = var_29.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 3
    var_33 = b'custom-salt'
    var_34 = module_0.Serializer(var_1, var_33)
    var_35 = var_34.iter_unsigners()
    var_36 = list(var_35)
    var_37 = var_36[var_6]
    var_38 = b'default-salt'
    var_39 = module_0.Serializer(var_1, var_38)
    var_40 = b'override-salt'
    var_41 = var_39.iter_unsigners(var_40)
    var_42 = list(var_41)
    var_43 = var_42[var_6]
    var_44 = []
    var_45 = module_0.Serializer(var_1, fallback_signers=var_44)
    var_46 = var_45.iter_unsigners()
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = {var_8: var_9}
    var_50 = {var_8: var_16}
    var_51 = var_45.iter_unsigners()
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 4
    var_54 = module_0.Serializer(var_1)
    var_55 = 'test data'
    var_56 = var_54.make_signer()



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1
import src.itsdangerous.signer as module_2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dumps(var_4)
    var_6 = {var_2: var_3}
    var_7 = module_1.dumps(var_6)
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = '.'
    var_12 = 1
    var_13 = {}
    var_14 = var_1.dumps(var_13)
    var_15 = 2
    var_16 = 3
    var_17 = [var_12, var_15, var_16]
    var_18 = var_1.dumps(var_17)
    var_19 = 'test'
    var_20 = var_1.dumps(var_19)
    var_21 = module_2.Signer(var_0)
    var_22 = 'utf-8'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test iter_unsigners returns the correct signers in order.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'key_derivation'
    var_9 = 'none'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_1, fallback_signers=var_11)
    var_13 = var_12.iter_unsigners()
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_6]
    var_17 = 1
    var_18 = var_14[var_17]
    var_19 = 'hmac'
    var_20 = {var_8: var_19}
    var_21 = var_12.iter_unsigners()
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_22[var_17]
    var_25 = var_12.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_26[var_17]
    var_29 = module_0.Serializer(var_1)
    var_30 = b'custom-salt'
    var_31 = var_29.iter_unsigners(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = [var_34, var_35]
    var_42 = {var_8: var_9}
    var_43 = [var_42]
    var_44 = module_0.Serializer(var_41, fallback_signers=var_43)
    var_45 = var_44.iter_unsigners()
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = None
    var_49 = module_0.Serializer(var_1, var_48)
    var_50 = var_49.iter_unsigners()
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = {var_8: var_9}
    var_54 = {var_8: var_19}
    var_55 = [var_54]
    var_56 = module_0.Serializer(var_1, signer_kwargs=var_53, fallback_signers=var_55)
    var_57 = var_56.iter_unsigners()
    var_58 = list(var_57)



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'some bytes'
    var_5 = b'test'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'test'
    var_9 = b'not json'
    var_10 = var_1.load_payload(var_9)



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test-payload'
    var_5 = module_0.Serializer(var_0)
    var_6 = b'{"test": "data"}'
    var_7 = b'invalid json'
    var_8 = var_5.load_payload(var_7)
    var_9 = b''
    var_10 = var_5.load_payload(var_9)
    var_11 = b'\xff\xfe'
    var_12 = var_5.load_payload(var_11)



# Parsed testcases at query #53
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == 'serialized_data'



# Parsed testcases at query #54
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '"hello"'
    var_3 = module_0.loads(var_2)
    assert var_3 == 'hello'
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
    var_12 = 'false'
    var_13 = module_0.loads(var_12)
    assert var_13 is False
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)



# Parsed testcases at query #55
#--------------------------


import json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'[1, 2, 3]'
    var_5 = module_0.loads(var_4)
    var_6 = b'{}'
    var_7 = module_0.loads(var_6)
    var_8 = b'{"a": [1, 2, {"b": 3}]}'
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #56
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = 'invalid payload'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #57
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_payload'
    var_1 = module_0.loads(var_0)
    var_2 = '42'
    var_3 = module_0.loads(var_2)
    assert var_3 == 42
    var_4 = 'abc'
    var_5 = module_0.loads(var_4)
    var_6 = b'hello'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None



# Parsed testcases at query #58
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol supports loads method'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = '"hello"'
    var_8 = module_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = 'null'
    var_10 = module_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = 'false'
    var_14 = module_0.loads(var_13)
    assert var_14 is False
    var_15 = '{}'
    var_16 = module_0.loads(var_15)
    var_17 = '{"a": {"b": [1, 2, 3]}}'
    var_18 = module_0.loads(var_17)



# Parsed testcases at query #59
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
    var_6 = {}
    var_7 = module_0.dumps(var_6)
    assert var_7 == '{}'
    var_8 = None
    var_9 = module_0.dumps(var_8)
    assert var_9 == 'null'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'



# Parsed testcases at query #60
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.loads works correctly with different payload types.'
    var_1 = 'TestSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, payload: json.loads(payload)
    var_6 = lambda self, obj: json.dumps(obj)
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = '{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = ()
    var_11 = lambda self, payload: json.loads(payload.decode())
    var_12 = lambda self, obj: json.dumps(obj).encode()
    var_13 = {var_3: var_11, var_4: var_12}
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)
    var_16 = ()
    var_17 = lambda self, payload: int(payload)
    var_18 = lambda self, obj: str(obj)
    var_19 = {var_3: var_17, var_4: var_18}
    var_20 = '42'
    var_21 = module_0.loads(var_20)
    assert var_21 == 42
    var_22 = '[1, 2, 3]'
    var_23 = module_0.loads(var_22)
    var_24 = 'null'
    var_25 = module_0.loads(var_24)
    assert var_25 is None



# Parsed testcases at query #61
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '{"number": 42}'
    var_3 = module_0.loads(var_2)
    var_4 = '{}'
    var_5 = module_0.loads(var_4)
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = '123'
    var_9 = module_0.loads(var_8)
    assert var_9 == 123



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)
    var_7 = b'hello'
    var_8 = b'invalid json'
    var_9 = var_1.load_payload(var_8)
    var_10 = b'{"test": 123}'
    var_11 = var_1.load_payload(var_10)
    assert var_11 == b'raw bytes'
    var_12 = b'raw bytes'
    var_13 = b'not json'
    var_14 = var_1.load_payload(var_13)



# Parsed testcases at query #63
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
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'false'
    var_11 = module_0.loads(var_10)
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = '{}'
    var_15 = module_0.loads(var_14)
    var_16 = '[]'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #64
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = '{"numbers": [1, 2, 3], "nested": {"a": 1}}'
    var_7 = module_0.loads(var_6)
    var_8 = '{}'
    var_9 = module_0.loads(var_8)
    var_10 = '[1, 2, 3]'
    var_11 = module_0.loads(var_10)



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'salt'
    var_5 = b'test_bytes'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b''
    var_9 = var_1.load_payload(var_8)
    var_10 = b'null'
    var_11 = var_1.load_payload(var_10)
    var_12 = b'42'
    var_13 = var_1.load_payload(var_12)
    assert var_13 == 42
    var_14 = b'[1, 2, 3]'
    var_15 = var_1.load_payload(var_14)
    var_16 = b'true'
    var_17 = var_1.load_payload(var_16)
    assert var_17 is True
    var_18 = b'null'
    var_19 = var_1.load_payload(var_18)
    assert var_19 is None
    var_20 = b'"hello"'
    var_21 = var_1.load_payload(var_20)
    assert var_21 == 'hello'



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'{"key": "value"}'
    var_6 = b'test_data'
    var_7 = b'test'
    var_8 = b'invalid json'
    var_9 = var_2.load_payload(var_8)
    var_10 = b''
    var_11 = var_2.load_payload(var_10)
    var_12 = b'test_bytes'
    var_13 = b'not json'
    var_14 = var_2.load_payload(var_13)
    var_15 = b'hello'
    var_16 = b'binary_data'
    var_17 = b'test'



# Parsed testcases at query #67
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value", "number": 42}'
    var_6 = module_0.loads(var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3]'
    var_12 = 'hello'
    var_13 = module_0.dumps(var_12)
    assert var_13 == '"hello"'



# Parsed testcases at query #68
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
    var_8 = module_0.Serializer(var_0)
    var_9 = {var_2: var_3}
    var_10 = 'custom-salt'
    var_11 = var_8.dumps(var_9, var_10)
    var_12 = {var_2: var_3}
    var_13 = var_8.dumps(var_12, var_10)
    var_14 = module_0.Serializer(var_0)
    var_15 = 'value1'
    var_16 = {var_2: var_15}
    var_17 = var_14.dumps(var_16)
    var_18 = 'value2'
    var_19 = {var_2: var_18}
    var_20 = var_14.dumps(var_19)
    var_21 = module_0.Serializer(var_0)
    var_22 = 'test'
    var_23 = 'number'
    var_24 = 'data'
    var_25 = 42
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = var_21.dumps(var_26)
    var_28 = var_21.loads(var_27)
    var_29 = 'indent'
    var_30 = 2
    var_31 = {var_29: var_30}
    var_32 = {var_2: var_3}
    var_33 = var_21.dumps(var_32)
    var_34 = module_0.Serializer(var_0)
    var_35 = {}
    var_36 = var_34.dumps(var_35)
    var_37 = module_0.Serializer(var_0)
    var_38 = 1
    var_39 = 3
    var_40 = [var_38, var_30, var_39]
    var_41 = var_37.dumps(var_40)



# Parsed testcases at query #69
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol defines loads method correctly.'
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
    var_13 = 'invalid json'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #70
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {}
    var_6 = module_0.dumps(var_5)
    var_7 = {var_1: var_2}
    var_8 = module_0.dumps(var_7)
    assert var_8 == b'{"key": "value"}'
    var_9 = {}
    var_10 = module_0.dumps(var_9)



# Parsed testcases at query #71
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test _PDataSerializer loads method protocol conformance.'
    var_1 = 'hello'
    var_2 = module_0.loads(var_1)
    assert var_2 == 'HELLO'
    var_3 = 'test'
    var_4 = module_0.loads(var_3)
    var_5 = b'hello'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'HELLO'
    var_7 = b'test'
    var_8 = module_0.loads(var_7)
    var_9 = '{"key": "value"}'
    var_10 = module_0.loads(var_9)
    var_11 = module_0.loads(var_3)
    assert var_11 == 'test'



# Parsed testcases at query #72
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 'test data'
    var_12 = var_1.dumps(var_11)
    var_13 = var_1.loads(var_12)
    assert var_13 == 'test data'
    var_14 = 'sort_keys'
    var_15 = 'separators'
    var_16 = True
    var_17 = ','
    var_18 = ':'
    var_19 = (var_17, var_18)
    var_20 = {var_14: var_16, var_15: var_19}
    var_21 = module_0.Serializer(var_0, serializer_kwargs=var_20)
    var_22 = 'b'
    var_23 = 'a'
    var_24 = 2
    var_25 = {var_22: var_24, var_23: var_16}
    var_26 = var_21.dumps(var_25)
    var_27 = 3
    var_28 = [var_16, var_24, var_27]
    var_29 = var_1.dumps(var_28)
    var_30 = var_1.loads(var_29)
    var_31 = 'simple string'
    var_32 = var_1.dumps(var_31)
    var_33 = var_1.loads(var_32)
    assert var_33 == 'simple string'
    var_34 = None
    var_35 = var_1.dumps(var_34)
    var_36 = var_1.loads(var_35)
    assert var_36 is None
    var_37 = 42
    var_38 = var_1.dumps(var_37)
    var_39 = var_1.loads(var_38)
    assert var_39 == 42
    var_40 = var_1.dumps(var_16)
    var_41 = var_1.loads(var_40)
    assert var_41 is True
    var_42 = module_0.Serializer(var_0)
    var_43 = 'test'
    var_44 = var_42.dumps(var_43)
    var_45 = 123
    var_46 = var_42.dumps(var_45)
    var_47 = var_42.dumps(var_34)
    var_48 = 'old-key'
    var_49 = 'new-key'
    var_50 = [var_48, var_49]
    var_51 = module_0.Serializer(var_50)
    var_52 = var_51.dumps(var_43)
    var_53 = var_51.loads(var_52)
    assert var_53 == 'test'



# Parsed testcases at query #73
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
    var_8 = 123
    var_9 = module_0.dumps(var_8)
    assert var_9 == '123'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    assert var_14 == '[1, 2, 3]'



# Parsed testcases at query #74
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test loads method of _PDataSerializer protocol.'
    var_1 = {}
    var_2 = module_0.dumps(var_1)
    var_3 = var_2.__class__
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.loads(var_6)
    var_8 = '42'
    var_9 = module_0.loads(var_8)
    assert var_9 == 42
    var_10 = '["a", "b", "c"]'
    var_11 = module_0.loads(var_10)
    var_12 = 'null'
    var_13 = module_0.loads(var_12)
    assert var_13 is None
    var_14 = '{invalid: json}'
    var_15 = module_0.loads(var_14)
    var_16 = ''
    var_17 = module_0.loads(var_16)
    var_18 = 'hello'
    var_19 = module_0.loads(var_18)
    assert var_19 == 'HELLO'
    var_20 = 'test'
    var_21 = module_0.loads(var_20)
    assert var_21 == 4
    var_22 = module_0.loads(var_20)
    var_23 = b'test'
    var_24 = module_0.loads(var_23)



# Parsed testcases at query #75
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol loads method works correctly.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = 'invalid json'
    var_4 = module_0.loads(var_3)



# Parsed testcases at query #76
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer's loads method protocol is properly implemented."
    var_1 = 'hello'
    var_2 = module_0.loads(var_1)
    assert var_2 == 'HELLO'
    var_3 = b'hello'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'HELLO'
    var_5 = '{"key": "value"}'
    var_6 = module_0.loads(var_5)
    var_7 = 'int:42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'float:3.14'
    var_10 = module_0.loads(var_9)
    var_11 = 'list:a,b,c'
    var_12 = module_0.loads(var_11)
    var_13 = 'plain string'
    var_14 = module_0.loads(var_13)
    assert var_14 == 'plain string'



# Parsed testcases at query #77
#--------------------------


import json as module_0

def test_case_0():
    var_0 = b'test payload'
    var_1 = module_0.loads(var_0)
    assert var_1 == 'test payload'
    var_2 = '{"key": "value"}'
    var_3 = module_0.loads(var_2)
    var_4 = b'hello,world'
    var_5 = module_0.loads(var_4)



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'{}'
    var_6 = b'invalid json'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'test'
    var_9 = b''
    var_10 = var_1.load_payload(var_9)
    var_11 = None
    var_12 = var_1.load_payload(var_11)



# Parsed testcases at query #79
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



# Parsed testcases at query #80
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
    var_10 = {var_2: var_3}
    var_11 = b'custom-salt'
    var_12 = var_1.dumps(var_10, var_11)
    var_13 = var_1.loads(var_12, var_11)
    var_14 = 'sort_keys'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = module_0.Serializer(var_0, serializer_kwargs=var_16)
    var_18 = 'b'
    var_19 = 'a'
    var_20 = 2
    var_21 = {var_18: var_20, var_19: var_15}
    var_22 = var_17.dumps(var_21)
    var_23 = var_17.loads(var_22)
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Serializer(var_26)
    var_28 = 'data'
    var_29 = 'test'
    var_30 = {var_28: var_29}
    var_31 = var_27.dumps(var_30)
    var_32 = var_27.loads(var_31)
    var_33 = {}
    var_34 = var_1.dumps(var_33)
    var_35 = var_1.loads(var_34)
    var_36 = 3
    var_37 = [var_15, var_20, var_36]
    var_38 = var_1.dumps(var_37)
    var_39 = var_1.loads(var_38)



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = module_1.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_1.loads(var_3)
    var_5 = 'invalid json'
    var_6 = module_1.loads(var_5)
    var_7 = '{}'
    var_8 = module_1.loads(var_7)
    var_9 = '[1, 2, 3]'
    var_10 = module_1.loads(var_9)
    var_11 = 'null'
    var_12 = module_1.loads(var_11)
    assert var_12 is None
    var_13 = 'true'
    var_14 = module_1.loads(var_13)
    assert var_14 is True
    var_15 = 'false'
    var_16 = module_1.loads(var_15)
    assert var_16 is False
    var_17 = '42'
    var_18 = module_1.loads(var_17)
    assert var_18 == 42
    var_19 = '3.14'
    var_20 = module_1.loads(var_19)
    var_21 = '"hello"'
    var_22 = module_1.loads(var_21)
    assert var_22 == 'hello'
    var_23 = '"\\u0048\\u0065\\u006c\\u006c\\u006f"'
    var_24 = module_1.loads(var_23)
    assert var_24 == 'Hello'
    var_25 = '{"a": {"b": 1}, "c": [2, 3]}'
    var_26 = module_1.loads(var_25)
    var_27 = 'Infinity'
    var_28 = module_1.loads(var_27)
    var_29 = 'inf'
    var_30 = float(var_29)
    var_31 = '-Infinity'
    var_32 = module_1.loads(var_31)
    var_33 = '-inf'
    var_34 = float(var_33)
    var_35 = 'NaN'
    var_36 = module_1.loads(var_35)
    var_37 = ''
    var_38 = module_1.loads(var_37)
    var_39 = '  42  '
    var_40 = module_1.loads(var_39)
    assert var_40 == 42
    var_41 = module_1.loads(var_37)



# Parsed testcases at query #82
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
    var_26 = 'old-key'
    var_27 = 'new-key'
    var_28 = [var_26, var_27]
    var_29 = module_0.Serializer(var_28)
    var_30 = var_29.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = [var_26, var_27]
    var_34 = {var_11: var_12}
    var_35 = [var_34]
    var_36 = module_0.Serializer(var_33, fallback_signers=var_35)
    var_37 = var_36.iter_unsigners()
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = module_0.Serializer(var_0)
    var_41 = var_40.iter_unsigners()
    var_42 = list(var_41)
    var_43 = var_42[var_5]
    var_44 = var_43.secret_key



# Parsed testcases at query #83
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = {var_1: var_3, var_2: var_4}
    var_9 = module_0.dumps(var_8)
    var_10 = module_0.loads(var_9)
    var_11 = 'test_string'
    var_12 = module_0.loads(var_11)
    assert var_12 == 'test_string'
    var_13 = '{"test": 123}'
    var_14 = module_0.loads(var_13)
    var_15 = 'invalid json'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #84
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #85
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol is properly duck-typed for loads'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '{"number": 42}'
    var_4 = module_0.loads(var_3)
    var_5 = '{"nested": {"inner": "data"}}'
    var_6 = module_0.loads(var_5)
    var_7 = '{"test": true}'
    var_8 = module_0.loads(var_7)
    var_9 = b'{"key": "value"}'
    var_10 = module_0.loads(var_9)
    var_11 = 'invalid'
    var_12 = module_0.loads(var_11)



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'{"key": "value"}'
    var_6 = module_0.Serializer(var_1)
    var_7 = b'{"custom": "data"}'
    var_8 = module_0.Serializer(var_1)
    var_9 = b'invalid json'
    var_10 = var_8.load_payload(var_9)
    var_11 = module_0.Serializer(var_1)
    var_12 = b''
    var_13 = var_11.load_payload(var_12)
    var_14 = module_0.Serializer(var_1)
    var_15 = None
    var_16 = var_14.load_payload(var_15)
    var_17 = module_0.Serializer(var_16)
    var_18 = b'{"list": [1, 2, 3], "nested": {"a": 1}}'
    var_19 = var_17.load_payload(var_18)
    var_20 = module_0.Serializer(var_16)
    var_21 = '{"message": "héllo"}'
    var_22 = 'utf-8'
    var_23 = var_20.load_payload(var_18)
    assert var_23 == b'raw bytes data'
    var_24 = module_0.Serializer(var_16)
    var_25 = b'raw bytes data'
    var_26 = module_0.Serializer(var_16)
    var_27 = b'{}'
    var_28 = var_26.load_payload(var_27, var_15)
    var_29 = module_0.Serializer(var_28)
    var_30 = b'[1, 2, 3]'
    var_31 = var_29.load_payload(var_30)
    var_32 = module_0.Serializer(var_28)
    var_33 = b'"string"'
    var_34 = var_32.load_payload(var_33)
    assert var_34 == 'string'
    var_35 = b'42'
    var_36 = var_32.load_payload(var_35)
    assert var_36 == 42
    var_37 = b'true'
    var_38 = var_32.load_payload(var_37)
    assert var_38 is True
    var_39 = b'null'
    var_40 = var_32.load_payload(var_39)
    assert var_40 is None



# Parsed testcases at query #87
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{"key": "value", "number": 42}'



# Parsed testcases at query #88
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
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = {var_2: var_3}
    var_15 = var_1.dumps(var_14)
    var_16 = 'indent'
    var_17 = 2
    var_18 = {var_16: var_17}
    var_19 = {var_2: var_3}
    var_20 = module_1.dumps(var_19)
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Serializer(var_23)
    var_25 = 'test'
    var_26 = 'data'
    var_27 = {var_25: var_26}
    var_28 = var_24.dumps(var_27)
    var_29 = var_24.loads(var_28)
    var_30 = var_24.salt
    var_31 = var_24.loads(var_28, var_30)
    var_32 = 123
    var_33 = var_1.dumps(var_32)
    var_34 = module_1.dumps(var_32)
    var_35 = {}
    var_36 = var_1.dumps(var_35)
    var_37 = {}
    var_38 = module_1.dumps(var_37)



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = var_0.loads(var_3)
    var_5 = '42'
    var_6 = var_0.loads(var_5)
    assert var_6 == 42
    var_7 = '"hello"'
    var_8 = var_0.loads(var_7)
    assert var_8 == 'hello'
    var_9 = b'{"key": "value"}'
    var_10 = var_0.loads(var_9)
    var_11 = 'invalid json'
    var_12 = var_0.loads(var_11)
    var_13 = None
    var_14 = var_0.loads(var_13)
    var_15 = ''
    var_16 = var_0.loads(var_15)



# Parsed testcases at query #90
#--------------------------




# Parsed testcases at query #91
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == b'{"key": "value"}'
    var_4 = module_0.dumps(var_2)
    assert var_4 == '{"key": "value"}'
    var_5 = None
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'null'
    var_7 = 42
    var_8 = module_0.dumps(var_7)
    assert var_8 == b'42'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == b'[1, 2, 3]'
    var_14 = {}
    var_15 = module_0.dumps(var_14)
    var_16 = {}
    var_17 = module_0.dumps(var_16)



# Parsed testcases at query #92
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol correctly defines dumps method.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"
    var_5 = 123
    var_6 = module_0.dumps(var_5)
    assert var_6 == '123'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3]'
    var_12 = 'dumps'



# Parsed testcases at query #93
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
    var_12 = 'name:John'
    var_13 = module_0.loads(var_12)
    var_14 = ''
    var_15 = module_0.loads(var_14)
    assert var_15 == ''



# Parsed testcases at query #94
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.dumps serializes data correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    var_13 = module_0.loads(var_12)
    var_14 = 'hello'
    var_15 = module_0.dumps(var_14)
    assert var_15 == '"hello"'
    var_16 = 123
    var_17 = module_0.dumps(var_16)
    assert var_17 == '123'
    var_18 = True
    var_19 = module_0.dumps(var_18)
    assert var_19 == 'true'
    var_20 = None
    var_21 = module_0.dumps(var_20)
    assert var_21 == 'null'



# Parsed testcases at query #95
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = {}
    var_12 = var_1.dumps(var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = var_1.dumps(var_16)
    var_18 = var_1.loads(var_7)
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Serializer(var_21)
    var_23 = 'test'
    var_24 = 'data'
    var_25 = {var_23: var_24}
    var_26 = var_22.dumps(var_25)
    var_27 = var_22.loads(var_26)
    var_28 = 'sort_keys'
    var_29 = 'separators'
    var_30 = True
    var_31 = ','
    var_32 = ':'
    var_33 = (var_31, var_32)
    var_34 = {var_28: var_30, var_29: var_33}
    var_35 = module_0.Serializer(var_0, serializer_kwargs=var_34)
    var_36 = 'b'
    var_37 = 'a'
    var_38 = {var_36: var_14, var_37: var_30}
    var_39 = var_35.dumps(var_38)
    var_40 = 0
    var_41 = '.'
    var_42 = result_kwargs.rsplit(var_41, var_30)[var_40]
    var_43 = var_35.loads(var_39)
    var_44 = 'key_derivation'
    var_45 = 'hmac'
    var_46 = {var_44: var_45}
    var_47 = module_0.Serializer(var_0, signer_kwargs=var_46)
    var_48 = {var_23: var_24}
    var_49 = var_47.dumps(var_48)
    var_50 = var_47.loads(var_49)
    var_51 = {var_23: var_24}
    var_52 = module_1.dumps(var_51)
    var_53 = module_1.loads(var_52)



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test Serializer.load_payload method.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'test_payload'
    var_6 = module_0.Serializer(var_1)
    var_7 = b'test'
    var_8 = b'invalid json'
    var_9 = var_2.load_payload(var_8)
    var_10 = b'hello world'
    var_11 = b'test'



# Parsed testcases at query #97
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
    var_7 = 'key_derivation'
    var_8 = 'none'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = module_0.Serializer(var_0, fallback_signers=var_10)
    var_12 = var_11.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_13[var_5]
    var_16 = 1
    var_17 = var_13[var_16]
    var_18 = {var_7: var_8}
    var_19 = var_11.iter_unsigners()
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_20[var_16]
    var_23 = var_11.iter_unsigners()
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_24[var_16]
    var_27 = b'custom-salt'
    var_28 = module_0.Serializer(var_0, var_27)
    var_29 = var_28.iter_unsigners()
    var_30 = list(var_29)
    var_31 = b'default-salt'
    var_32 = module_0.Serializer(var_0, var_31)
    var_33 = b'override-salt'
    var_34 = var_32.iter_unsigners(var_33)
    var_35 = list(var_34)
    var_36 = 'old-key'
    var_37 = 'new-key'
    var_38 = [var_36, var_37]
    var_39 = module_0.Serializer(var_38)
    var_40 = var_39.iter_unsigners()
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = [var_36, var_37]
    var_44 = {var_7: var_8}
    var_45 = [var_44]
    var_46 = module_0.Serializer(var_43, fallback_signers=var_45)
    var_47 = var_46.iter_unsigners()
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 3
    var_50 = [var_36, var_37]
    var_51 = 'hmac'
    var_52 = {var_7: var_51}
    var_53 = var_46.iter_unsigners()
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 3
    var_56 = var_54[var_16]
    var_57 = 2
    var_58 = var_54[var_57]
    var_59 = [var_36, var_37]
    var_60 = var_46.iter_unsigners()
    var_61 = list(var_60)
    var_62 = len(var_61)
    assert var_62 == 3
    var_63 = var_61[var_16]
    var_64 = var_61[var_57]
    var_65 = []
    var_66 = module_0.Serializer(var_0, fallback_signers=var_65)
    var_67 = var_66.iter_unsigners()
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1
    var_70 = None
    var_71 = module_0.Serializer(var_0, var_70)
    var_72 = var_71.iter_unsigners()
    var_73 = list(var_72)
    var_74 = 'main-key'
    var_75 = {var_7: var_8}
    var_76 = {var_7: var_51}
    var_77 = var_71.iter_unsigners()
    var_78 = list(var_77)
    var_79 = len(var_78)
    assert var_79 == 4
    var_80 = var_78[var_57]
    var_81 = 3
    var_82 = var_78[var_81]
    var_83 = 'key1'
    var_84 = 'key2'
    var_85 = 'key3'
    var_86 = [var_83, var_84, var_85]
    var_87 = var_71.iter_unsigners()
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 4
    var_90 = {var_7: var_8}
    var_91 = var_71.iter_unsigners()
    var_92 = list(var_91)
    var_93 = {var_7: var_8}
    var_94 = {var_7: var_51}
    var_95 = [var_94]
    var_96 = module_0.Serializer(var_0, signer_kwargs=var_93, fallback_signers=var_95)
    var_97 = var_96.iter_unsigners()
    var_98 = list(var_97)



# Parsed testcases at query #98
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
    var_8 = module_0.Serializer(var_0, var_7)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'key_derivation'
    var_13 = 'none'
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
    var_24 = 1
    var_25 = var_22[var_24]
    var_26 = var_16.iter_unsigners()
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = var_27[var_24]
    var_30 = 'old-key'
    var_31 = 'new-key'
    var_32 = [var_30, var_31]
    var_33 = module_0.Serializer(var_32)
    var_34 = var_33.iter_unsigners()
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = [var_30, var_31]
    var_38 = {var_12: var_13}
    var_39 = [var_38]
    var_40 = module_0.Serializer(var_37, fallback_signers=var_39)
    var_41 = var_40.iter_unsigners()
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 3
    var_44 = b'default-salt'
    var_45 = module_0.Serializer(var_0, var_44)
    var_46 = b'override-salt'
    var_47 = var_45.iter_unsigners(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1



# Parsed testcases at query #99
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'test'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"test"'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = True
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'true'
    var_18 = 'a'
    var_19 = {var_18: var_16}
    var_20 = module_0.dumps(var_19)
    var_21 = 'dumps should return str type'



# Parsed testcases at query #100
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = 'test'
    var_11 = module_0.dumps(var_10)
    assert var_11 == '"test"'
    var_12 = 42
    var_13 = module_0.dumps(var_12)
    assert var_13 == '42'
    var_14 = None
    var_15 = module_0.dumps(var_14)
    assert var_15 == 'null'
    var_16 = True
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'true'
    var_18 = False
    var_19 = module_0.dumps(var_18)
    assert var_19 == 'false'
    var_20 = 'data'
    var_21 = {var_10: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = 'string'
    var_24 = 'number'
    var_25 = 'list'
    var_26 = 'nested'
    var_27 = 'hello'
    var_28 = 123
    var_29 = [var_16, var_6, var_7]
    var_30 = {var_1: var_2}
    var_31 = {var_23: var_27, var_24: var_28, var_25: var_29, var_26: var_30}
    var_32 = module_0.dumps(var_31)
    var_33 = module_0.loads(var_32)



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test that dumps returns correct type and can be loaded back.'
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
    var_15 = var_12.loads(var_14)
    var_16 = var_2.loads(var_6)
    var_17 = 'key1'
    var_18 = module_0.Serializer(var_17)
    var_19 = 'key2'
    var_20 = module_0.Serializer(var_19)
    var_21 = 'test'
    var_22 = 'data'
    var_23 = {var_21: var_22}



# Parsed testcases at query #102
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.dumps works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = module_0.loads(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = module_0.dumps(var_13)
    var_15 = module_0.loads(var_14)
    var_16 = 123
    var_17 = module_0.dumps(var_16)
    var_18 = module_0.loads(var_17)
    var_19 = None
    var_20 = module_0.dumps(var_19)
    var_21 = module_0.loads(var_20)
    assert var_21 is None
    var_22 = True
    var_23 = module_0.dumps(var_22)
    var_24 = module_0.loads(var_23)
    assert var_24 is True
    var_25 = 'outer'
    var_26 = 'list'
    var_27 = 'inner'
    var_28 = [var_22, var_9, var_10]
    var_29 = 'test'
    var_30 = {var_27: var_28, var_3: var_29}
    var_31 = {var_11: var_22}
    var_32 = {var_12: var_9}
    var_33 = [var_31, var_32]
    var_34 = {var_25: var_30, var_26: var_33}
    var_35 = module_0.dumps(var_34)
    var_36 = module_0.loads(var_35)



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'{"name": "\\u00e9l\\u00e8ve"}'
    var_6 = var_2.load_payload(var_5)
    var_7 = b'{"count": 42, "price": 19.99}'
    var_8 = var_2.load_payload(var_7)
    var_9 = b'[1, 2, 3, "test"]'
    var_10 = var_2.load_payload(var_9)
    var_11 = b'{"active": true, "completed": false}'
    var_12 = var_2.load_payload(var_11)
    var_13 = b'{"data": null}'
    var_14 = var_2.load_payload(var_13)
    var_15 = b'{}'
    var_16 = var_2.load_payload(var_15)
    var_17 = b'[]'
    var_18 = var_2.load_payload(var_17)
    var_19 = b'invalid json'
    var_20 = var_2.load_payload(var_19)
    var_21 = b''
    var_22 = var_2.load_payload(var_21)
    var_23 = b'{"key": "value"}extra'
    var_24 = var_2.load_payload(var_23)
    var_25 = b'test bytes payload'
    var_26 = b'custom data'
    var_27 = b'test'
    var_28 = b'{"outer": {"inner": "value"}, "list": [1, {"nested": True}]}'
    var_29 = var_2.load_payload(var_28)
    var_30 = b'{"text": "Line 1\\nLine 2\\tTabbed"}'
    var_31 = var_2.load_payload(var_30)
    var_32 = b'{"unicode": "\\u0048\\u0065\\u006c\\u006c\\u006f"}'
    var_33 = var_2.load_payload(var_32)



# Parsed testcases at query #104
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer.loads correctly deserializes data.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '123'
    var_4 = module_0.loads(var_3)
    assert var_4 == 123
    var_5 = '"hello"'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = '[1, 2, 3]'
    var_12 = module_0.loads(var_11)
    var_13 = '{"a": [1, 2, {"b": 3}], "c": "d"}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #105
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
    var_12 = 'hello'
    var_13 = module_0.loads(var_12)
    assert var_13 == 'HELLO'



# Parsed testcases at query #106
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_0.loads(var_5)
    var_7 = 'string'
    var_8 = module_0.dumps(var_7)
    assert var_8 == '"string"'
    var_9 = 123
    var_10 = module_0.dumps(var_9)
    assert var_10 == '123'
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.dumps(var_14)
    assert var_15 == '[1, 2, 3]'
    var_16 = None
    var_17 = module_0.dumps(var_16)
    assert var_17 == 'null'



# Parsed testcases at query #107
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol accepts valid serializer implementations.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '{"test": 123}'
    var_6 = module_0.loads(var_5)
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = module_0.loads(var_9)
    assert var_10 is True
    var_11 = '42'
    var_12 = module_0.loads(var_11)
    assert var_12 == 42
    var_13 = '"string"'
    var_14 = module_0.loads(var_13)
    assert var_14 == 'string'
    var_15 = '[1, 2, 3]'
    var_16 = module_0.loads(var_15)



# Parsed testcases at query #108
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
    var_14 = '{"a": [1, 2, {"b": 3}]}'
    var_15 = module_0.loads(var_14)
    var_16 = '{}'
    var_17 = module_0.loads(var_16)
    var_18 = '[]'
    var_19 = module_0.loads(var_18)



# Parsed testcases at query #109
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works correctly."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = '[1, 2, 3]'
    var_6 = module_0.loads(var_5)
    var_7 = 'null'
    var_8 = module_0.loads(var_7)
    assert var_8 is None
    var_9 = '42'
    var_10 = module_0.loads(var_9)
    assert var_10 == 42
    var_11 = 'true'
    var_12 = module_0.loads(var_11)
    assert var_12 is True
    var_13 = '{}'
    var_14 = module_0.loads(var_13)



# Parsed testcases at query #110
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
    var_10 = module_0.Serializer(var_1)
    var_11 = {var_3: var_4}
    var_12 = 'custom-salt'
    var_13 = var_10.dumps(var_11, var_12)
    var_14 = 'other-value'
    var_15 = {var_3: var_14}
    var_16 = var_2.dumps(var_15)
    var_17 = {}
    var_18 = var_2.dumps(var_17)
    var_19 = var_2.loads(var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = var_2.dumps(var_23)
    var_25 = var_2.loads(var_24)
    var_26 = None
    var_27 = var_2.dumps(var_26)
    var_28 = var_2.loads(var_27)
    assert var_28 is None



# Parsed testcases at query #111
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol implementers can dumps data.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'
    var_7 = {var_1: var_2}
    var_8 = module_0.dumps(var_7)
    assert var_8 == '{"key": "value"}'



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = var_1.load_payload(var_2)
    var_4 = b'test-data'
    var_5 = b'custom-data'
    var_6 = b'{"key": "value"}'
    var_7 = b'invalid json'
    var_8 = var_1.load_payload(var_7)
    var_9 = b''
    var_10 = var_1.load_payload(var_9)
    var_11 = None
    var_12 = var_1.load_payload(var_11)
    var_13 = b'{"special": "\\u00e9\\u00f1\\u00fc"}'
    var_14 = var_1.load_payload(var_13)
    var_15 = b'{"num": 42}'
    var_16 = var_1.load_payload(var_15)
    var_17 = b'{"flag": true}'
    var_18 = var_1.load_payload(var_17)
    var_19 = b'{"value": null}'
    var_20 = var_1.load_payload(var_19)
    var_21 = b'{"nested": {"inner": "value"}}'
    var_22 = var_1.load_payload(var_21)
    var_23 = b'{"items": [1, 2, 3]}'
    var_24 = var_1.load_payload(var_23)



# Parsed testcases at query #113
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test _PDataSerializer protocol's loads method behavior."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = module_0.loads(var_3)
    var_5 = 'a,b,c'
    var_6 = module_0.loads(var_5)
    var_7 = 'invalid json'
    var_8 = module_0.loads(var_7)



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test iter_unsigners method of Serializer class.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'digest_method'
    var_9 = 'sha256'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_1, fallback_signers=var_11)
    var_13 = var_12.iter_unsigners()
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key_derivation'
    var_17 = 'hmac'
    var_18 = {var_16: var_17}
    var_19 = var_12.iter_unsigners()
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_12.iter_unsigners()
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'key1'
    var_26 = 'key2'
    var_27 = 'key3'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.Serializer(var_28)
    var_30 = var_29.iter_unsigners()
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = [var_25, var_26]
    var_34 = {var_8: var_9}
    var_35 = [var_34]
    var_36 = module_0.Serializer(var_33, fallback_signers=var_35)
    var_37 = var_36.iter_unsigners()
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = module_0.Serializer(var_1)
    var_41 = b'custom-salt'
    var_42 = var_40.iter_unsigners(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = var_43[var_6]
    var_46 = []
    var_47 = module_0.Serializer(var_1, fallback_signers=var_46)
    var_48 = var_47.iter_unsigners()
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = {var_8: var_9}
    var_52 = [var_51]
    var_53 = module_0.Serializer(var_1, fallback_signers=var_52)
    var_54 = var_53.iter_unsigners()
    var_55 = '__next__'
    var_56 = hasattr(var_54, var_55)
    var_57 = '__iter__'
    var_58 = hasattr(var_54, var_57)
    var_59 = [var_25, var_26, var_27]
    var_60 = {var_8: var_9}
    var_61 = [var_60]
    var_62 = module_0.Serializer(var_59, fallback_signers=var_61)
    var_63 = var_62.iter_unsigners()
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 4



# Parsed testcases at query #115
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
    var_12 = {var_2: var_3}
    var_13 = var_11.dumps(var_12)
    var_14 = var_11.loads(var_13, var_10)
    var_15 = 'sort_keys'
    var_16 = True
    var_17 = {var_15: var_16}
    var_18 = module_0.Serializer(var_0, serializer_kwargs=var_17)
    var_19 = 'b'
    var_20 = 'a'
    var_21 = 2
    var_22 = {var_19: var_21, var_20: var_16}
    var_23 = var_18.dumps(var_22)
    var_24 = var_18.loads(var_23)
    var_25 = 'key1'
    var_26 = module_0.Serializer(var_25)
    var_27 = 'key2'
    var_28 = module_0.Serializer(var_27)
    var_29 = 'test'
    var_30 = 'data'
    var_31 = {var_29: var_30}
    var_32 = var_26.dumps(var_31)
    var_33 = var_28.dumps(var_31)
    var_34 = module_0.Serializer(var_0)
    var_35 = {}
    var_36 = var_34.dumps(var_35)
    var_37 = var_34.loads(var_36)
    var_38 = 3
    var_39 = [var_16, var_21, var_38]
    var_40 = var_34.dumps(var_39)
    var_41 = var_34.loads(var_40)
    var_42 = 'hello'
    var_43 = var_34.dumps(var_42)
    var_44 = var_34.loads(var_43)
    assert var_44 == 'hello'
    var_45 = {var_2: var_3}
    var_46 = var_34.dumps(var_45)
    var_47 = {var_2: var_3}
    var_48 = var_34.dumps(var_47)
    var_49 = 'old-key'
    var_50 = 'new-key'
    var_51 = [var_49, var_50]
    var_52 = module_0.Serializer(var_51)
    var_53 = {var_2: var_3}
    var_54 = var_52.dumps(var_53)
    var_55 = var_52.loads(var_54)
    var_56 = [var_49]
    var_57 = module_0.Serializer(var_56)
    var_58 = var_57.loads(var_54)



# Parsed testcases at query #116
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works with different payload types."
    var_1 = 'StrSerializer'
    var_2 = ()
    var_3 = 'loads'
    var_4 = 'dumps'
    var_5 = lambda self, payload: json.loads(payload)
    var_6 = lambda self, obj: json.dumps(obj)
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = '{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = 'BytesSerializer'
    var_11 = ()
    var_12 = lambda self, payload: json.loads(payload.decode())
    var_13 = lambda self, obj: json.dumps(obj).encode()
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = b'{"key": "value"}'
    var_16 = module_0.loads(var_15)
    var_17 = '"string"'
    var_18 = module_0.loads(var_17)
    var_19 = '123'
    var_20 = module_0.loads(var_19)
    var_21 = 'null'
    var_22 = module_0.loads(var_21)
    var_23 = None



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = var_2.loads(var_6)
    var_8 = module_1.dumps(var_5)
    var_9 = module_1.loads(var_8)
    var_10 = b'custom-salt'
    var_11 = var_2.dumps(var_5, var_10)
    var_12 = var_2.loads(var_11, var_10)
    var_13 = b'salt1'
    var_14 = var_2.dumps(var_5, var_13)
    var_15 = b'salt2'
    var_16 = var_2.dumps(var_5, var_15)
    var_17 = 'sort_keys'
    var_18 = 'separators'
    var_19 = True
    var_20 = ','
    var_21 = ':'
    var_22 = (var_20, var_21)
    var_23 = {var_17: var_19, var_18: var_22}
    var_24 = module_0.Serializer(var_0, var_1, serializer_kwargs=var_23)
    var_25 = var_24.dumps(var_5)
    var_26 = var_24.loads(var_25)
    var_27 = {var_3: var_4}
    var_28 = 2
    var_29 = 3
    var_30 = [var_19, var_28, var_29]
    var_31 = 'simple string'
    var_32 = 42
    var_33 = None
    var_34 = 'nested'
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_19, var_28, var_29]
    var_38 = {var_35: var_19, var_36: var_37}
    var_39 = {var_34: var_38}
    var_40 = [var_27, var_30, var_31, var_32, var_33, var_39]
    var_41 = var_2.dumps(var_5)
    var_42 = var_2.dumps(var_5)
    var_43 = {}
    var_44 = var_2.dumps(var_43)
    var_45 = var_2.loads(var_44)



# Parsed testcases at query #118
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)
    assert var_6 == b'{"key": "value"}'



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method with various configurations.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'test-data'
    var_8 = var_2.dumps(var_7)
    var_9 = var_2.loads(var_8)
    assert var_9 == 'test-data'
    var_10 = 'test-key'
    var_11 = {var_3: var_4}
    var_12 = module_1.dumps(var_11)
    var_13 = module_1.loads(var_12)
    var_14 = b'custom-salt'
    var_15 = module_0.Serializer(var_10, var_14)
    var_16 = var_15.dumps(var_7)
    var_17 = var_15.loads(var_16, var_14)
    assert var_17 == 'test-data'
    var_18 = 'key1'
    var_19 = module_0.Serializer(var_18)
    var_20 = 'key2'
    var_21 = module_0.Serializer(var_20)
    var_22 = 'test'
    var_23 = var_19.dumps(var_22)
    var_24 = var_21.dumps(var_22)
    var_25 = 'indent'
    var_26 = 2
    var_27 = {var_25: var_26}
    var_28 = {var_3: var_4}
    var_29 = module_1.dumps(var_28)
    var_30 = module_1.loads(var_29)
    var_31 = 'string'
    var_32 = 'number'
    var_33 = 'list'
    var_34 = 'nested'
    var_35 = 'hello'
    var_36 = 42
    var_37 = 1
    var_38 = 3
    var_39 = [var_37, var_26, var_38]
    var_40 = 'a'
    var_41 = 'b'
    var_42 = {var_40: var_37, var_41: var_26}
    var_43 = {var_31: var_35, var_32: var_36, var_33: var_39, var_34: var_42}
    var_44 = var_2.dumps(var_43)
    var_45 = var_2.loads(var_44)
    var_46 = {}
    var_47 = var_2.dumps(var_46)
    var_48 = var_2.loads(var_47)
    var_49 = None
    var_50 = var_2.dumps(var_49)
    var_51 = var_2.loads(var_50)
    assert var_51 is None
    var_52 = 'same-data'
    var_53 = var_2.dumps(var_52)
    var_54 = var_2.dumps(var_52)
    var_55 = len(var_53)
    var_56 = len(var_54)



# Parsed testcases at query #120
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    assert var_5 == 42
    var_6 = '"hello"'
    var_7 = module_0.loads(var_6)
    assert var_7 == 'hello'
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = b'{"key": "value"}'
    var_13 = module_0.loads(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.loads(var_14)
    var_16 = 'invalid json'
    var_17 = module_0.loads(var_16)
    var_18 = ''
    var_19 = module_0.loads(var_18)



# Parsed testcases at query #121
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol dumps method works correctly.'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'list'
    var_4 = 'value'
    var_5 = 42
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_1: var_4, var_2: var_5, var_3: var_9}
    var_11 = module_0.dumps(var_10)
    var_12 = module_0.loads(var_11)
    var_13 = 'hello'
    var_14 = module_0.dumps(var_13)
    assert var_14 == '"hello"'
    var_15 = 123
    var_16 = module_0.dumps(var_15)
    assert var_16 == '123'
    var_17 = True
    var_18 = module_0.dumps(var_17)
    assert var_18 == 'true'
    var_19 = None
    var_20 = module_0.dumps(var_19)
    assert var_20 == 'null'
    var_21 = {}
    var_22 = module_0.dumps(var_21)
    assert var_22 == '{}'
    var_23 = []
    var_24 = module_0.dumps(var_23)
    assert var_24 == '[]'
    var_25 = module_0.dumps(var_0)



# Parsed testcases at query #122
#--------------------------


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'
    var_2 = b'invalid json'
    var_3 = b''



# Parsed testcases at query #123
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



# Parsed testcases at query #124
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
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = module_1.dumps(var_10)
    var_12 = module_1.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = module_0.Serializer(var_0, var_13)
    var_15 = 'test data'
    var_16 = var_14.dumps(var_15)
    var_17 = var_14.loads(var_16)
    var_18 = 'sort_keys'
    var_19 = 'separators'
    var_20 = True
    var_21 = ','
    var_22 = ':'
    var_23 = (var_21, var_22)
    var_24 = {var_18: var_20, var_19: var_23}
    var_25 = module_0.Serializer(var_0, serializer_kwargs=var_24)
    var_26 = 'b'
    var_27 = 'a'
    var_28 = {var_26: var_8, var_27: var_20}
    var_29 = var_25.dumps(var_28)
    var_30 = var_25.loads(var_29)
    var_31 = b'bytes-key'
    var_32 = module_0.Serializer(var_31)
    var_33 = 'test'
    var_34 = var_32.dumps(var_33)
    var_35 = var_32.loads(var_34)
    var_36 = 'old-key'
    var_37 = 'new-key'
    var_38 = [var_36, var_37]
    var_39 = module_0.Serializer(var_38)
    var_40 = 'test data'
    var_41 = var_39.dumps(var_40)
    var_42 = var_39.loads(var_41)
    var_43 = 'secret'
    var_44 = module_0.Serializer(var_43)
    var_45 = {var_27: var_20}
    var_46 = var_44.dumps(var_45)
    var_47 = {var_27: var_8}
    var_48 = var_44.dumps(var_47)
    var_49 = 'salt1'
    var_50 = module_0.Serializer(var_43, var_49)
    var_51 = 'salt2'
    var_52 = module_0.Serializer(var_43, var_51)
    var_53 = 'test'
    var_54 = 'data'
    var_55 = {var_53: var_54}
    var_56 = var_50.dumps(var_55)
    var_57 = var_52.dumps(var_55)
    var_58 = var_50.loads(var_56)
    var_59 = var_52.loads(var_57)



# Parsed testcases at query #125
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "{'test': 'data'}"
    var_4 = {var_0: var_1}
    var_5 = module_0.dumps(var_4)
    assert var_5 == b"{'test': 'data'}"



# Parsed testcases at query #126
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



# Parsed testcases at query #127
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
    var_7 = var_1.loads(var_5)
    var_8 = {var_2: var_3}
    var_9 = module_1.dumps(var_8)
    var_10 = module_1.loads(var_9)
    var_11 = {var_2: var_3}
    var_12 = 'custom-salt'
    var_13 = var_1.dumps(var_11, var_12)
    var_14 = var_1.loads(var_13, var_12)
    var_15 = 123
    var_16 = var_1.dumps(var_15)
    var_17 = 'string'
    var_18 = var_1.dumps(var_17)
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = var_1.dumps(var_22)
    var_24 = None
    var_25 = var_1.dumps(var_24)
    var_26 = 'sort_keys'
    var_27 = True
    var_28 = {var_26: var_27}
    var_29 = module_0.Serializer(var_0, serializer_kwargs=var_28)
    var_30 = 'b'
    var_31 = 'a'
    var_32 = {var_30: var_27, var_31: var_20}
    var_33 = var_29.dumps(var_32)
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = 'test'
    var_39 = 'data'
    var_40 = {var_38: var_39}
    var_41 = var_37.dumps(var_40)
    var_42 = var_37.loads(var_41)



# Parsed testcases at query #128
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == "{'key': 'value'}"
    var_5 = 42
    var_6 = module_0.dumps(var_5)
    assert var_6 == '42'
    var_7 = 'hello'
    var_8 = module_0.dumps(var_7)
    assert var_8 == 'hello'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.dumps(var_12)
    assert var_13 == '[1, 2, 3]'
    var_14 = 'test'
    var_15 = 'data'
    var_16 = {var_14: var_15}
    var_17 = module_0.dumps(var_16)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test Serializer.dumps method with various configurations.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = 'eyJrZXkiOiAidmFsdWUifQ'
    var_8 = {var_3: var_4}
    var_9 = module_1.dumps(var_8)
    var_10 = {var_3: var_4}
    var_11 = 'custom-salt'
    var_12 = var_2.dumps(var_10, var_11)
    var_13 = 'data'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = var_2.dumps(var_15)
    var_17 = {var_13: var_14}
    var_18 = var_2.dumps(var_17)
    var_19 = 'sort_keys'
    var_20 = 'separators'
    var_21 = True
    var_22 = ','
    var_23 = ':'
    var_24 = (var_22, var_23)
    var_25 = {var_19: var_21, var_20: var_24}
    var_26 = module_0.Serializer(var_1, serializer_kwargs=var_25)
    var_27 = 'b'
    var_28 = 'a'
    var_29 = 2
    var_30 = {var_27: var_29, var_28: var_21}
    var_31 = var_26.dumps(var_30)
    var_32 = 'string'
    var_33 = var_2.dumps(var_32)
    var_34 = 123
    var_35 = var_2.dumps(var_34)
    var_36 = 3
    var_37 = [var_21, var_29, var_36]
    var_38 = var_2.dumps(var_37)
    var_39 = None
    var_40 = var_2.dumps(var_39)



# Parsed testcases at query #130
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol implementations can dumps objects.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = {var_1: var_2}
    var_6 = module_0.dumps(var_5)



# Parsed testcases at query #131
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test payload'
    var_1 = module_0.loads(var_0)
    var_2 = '{"name": "test", "value": 42}'
    var_3 = module_0.loads(var_2)
    var_4 = b'hello bytes'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'hello bytes'
    var_6 = '123'
    var_7 = module_0.loads(var_6)
    assert var_7 == 123
    var_8 = None
    var_9 = module_0.loads(var_8)
    var_10 = ''
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = 'non-empty'
    var_13 = module_0.loads(var_12)
    assert var_13 == 'non-empty'



# Parsed testcases at query #132
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '42'
    var_5 = module_0.loads(var_4)
    assert var_5 == 42
    var_6 = 'true'
    var_7 = module_0.loads(var_6)
    assert var_7 is True
    var_8 = 'null'
    var_9 = module_0.loads(var_8)
    assert var_9 is None
    var_10 = '"hello"'
    var_11 = module_0.loads(var_10)
    assert var_11 == 'hello'
    var_12 = b'apple,banana,cherry'
    var_13 = module_0.loads(var_12)
    var_14 = '{}'
    var_15 = module_0.loads(var_14)
    var_16 = '{"a": {"b": [1, 2, 3]}}'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #133
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.loads(var_0)
    var_2 = b'world'
    var_3 = module_0.loads(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.loads(var_4)
    var_6 = ''
    var_7 = module_0.loads(var_6)
    var_8 = b''
    var_9 = module_0.loads(var_8)



# Parsed testcases at query #134
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
    var_12 = 'invalid json'
    var_13 = module_0.loads(var_12)



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test Serializer.load_payload method.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = b'some bytes data'
    var_6 = module_0.Serializer(var_1)
    var_7 = b'custom data'
    var_8 = b'invalid json'
    var_9 = var_2.load_payload(var_8)
    var_10 = b''
    var_11 = var_2.load_payload(var_10)
    var_12 = None
    var_13 = var_2.load_payload(var_12)
    var_14 = b'hello world'



# Parsed testcases at query #136
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
    var_20 = module_0.Serializer(var_0, serializer_kwargs=var_19)
    var_21 = 'b'
    var_22 = 'a'
    var_23 = 2
    var_24 = {var_21: var_23, var_22: var_15}
    var_25 = var_20.dumps(var_24)
    var_26 = var_1.loads(var_25)
    var_27 = {}
    var_28 = var_1.dumps(var_27)
    var_29 = var_1.loads(var_28)



# Parsed testcases at query #137
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.dumps(var_8)
    assert var_9 == '[1, 2, 3]'
    var_10 = None
    var_11 = module_0.dumps(var_10)
    assert var_11 == 'null'
    var_12 = 'hello'
    var_13 = module_0.dumps(var_12)
    assert var_13 == '"hello"'
    var_14 = 'nested'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = [var_5, var_6, var_7]
    var_18 = {var_15: var_5, var_16: var_17}
    var_19 = {var_14: var_18, var_2: var_10}
    var_20 = module_0.dumps(var_19)
    var_21 = module_0.loads(var_20)



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '42'
    var_4 = var_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = var_0.loads(var_5)
    var_7 = 'null'
    var_8 = var_0.loads(var_7)
    assert var_8 is None
    var_9 = 'true'
    var_10 = var_0.loads(var_9)
    assert var_10 is True
    var_11 = '3.14'
    var_12 = var_0.loads(var_11)
    var_13 = '""'
    var_14 = var_0.loads(var_13)
    assert var_14 == ''



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = module_0.Serializer(var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_1.dumps(var_7)
    var_9 = b'invalid json'
    var_10 = var_2.load_payload(var_9)
    var_11 = b''
    var_12 = var_2.load_payload(var_11)
    var_13 = b'{"key": "value"}'
    var_14 = b'test'
    var_15 = 'not bytes'
    var_16 = var_2.load_payload(var_15)
    var_17 = b'{"a": 1}'
    var_18 = None
    var_19 = var_2.load_payload(var_17, var_18)



# Parsed testcases at query #140
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'my-salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_0, var_1, fallback_signers=var_11)
    var_13 = var_12.iter_unsigners()
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'digest_method'
    var_17 = 'sha256'
    var_18 = {var_16: var_17}
    var_19 = var_12.iter_unsigners()
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_20[var_6]
    var_23 = 1
    var_24 = var_20[var_23]
    var_25 = var_12.iter_unsigners()
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_26[var_6]
    var_29 = var_26[var_23]
    var_30 = b'custom-salt'
    var_31 = var_12.iter_unsigners(var_30)
    var_32 = list(var_31)
    var_33 = b'old-key'
    var_34 = b'newer-key'
    var_35 = b'newest-key'
    var_36 = [var_33, var_34, var_35]
    var_37 = module_0.Serializer(var_36, var_1)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 3



# Parsed testcases at query #141
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer protocol conforming objects can dump data.'
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
    var_9 = 42
    var_10 = module_0.dumps(var_9)
    assert var_10 == '42'
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.dumps(var_14)
    assert var_15 == '[1, 2, 3]'



# Parsed testcases at query #142
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
    var_12 = 'key_derivation'
    var_13 = 'hmac'
    var_14 = {var_12: var_13}
    var_15 = [var_14]
    var_16 = module_0.Serializer(var_0, fallback_signers=var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_18[var_5]
    var_21 = 1
    var_22 = var_18[var_21]
    var_23 = {var_12: var_13}
    var_24 = var_16.iter_unsigners()
    var_25 = list(var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = var_25[var_5]
    var_28 = var_25[var_21]
    var_29 = var_16.iter_unsigners()
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_30[var_5]
    var_33 = var_30[var_21]
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Serializer(var_36)
    var_38 = var_37.iter_unsigners()
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = [var_34, var_35]
    var_42 = {var_12: var_13}
    var_43 = [var_42]
    var_44 = module_0.Serializer(var_41, fallback_signers=var_43)
    var_45 = var_44.iter_unsigners()
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = module_0.Serializer(var_0)
    var_49 = var_48.iter_unsigners(var_7)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = None
    var_53 = module_0.Serializer(var_0, var_52)
    var_54 = var_53.iter_unsigners()
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = []
    var_58 = module_0.Serializer(var_0, fallback_signers=var_57)
    var_59 = var_58.iter_unsigners()
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 1



# Parsed testcases at query #143
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
    var_9 = 'test'
    var_10 = module_0.dumps(var_9)
    assert var_10 == '"test"'
    var_11 = 42
    var_12 = module_0.dumps(var_11)
    assert var_12 == '42'
    var_13 = True
    var_14 = module_0.dumps(var_13)
    assert var_14 == 'true'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'
    var_17 = {}
    var_18 = module_0.dumps(var_17)
    assert var_18 == '{}'
    var_19 = []
    var_20 = module_0.dumps(var_19)
    assert var_20 == '[]'
    var_21 = 'a'
    var_22 = 'b'
    var_23 = [var_13, var_5, var_6]
    var_24 = {var_22: var_23}
    var_25 = {var_21: var_24}
    var_26 = module_0.dumps(var_25)
    assert var_26 == '{"a": {"b": [1, 2, 3]}}'



# Parsed testcases at query #144
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
    var_6 = 'test_data'
    var_7 = var_1.dumps(var_6)
    var_8 = var_1.loads(var_7)
    assert var_8 == 'test_data'
    var_9 = {var_2: var_3}
    var_10 = module_1.dumps(var_9)
    var_11 = 'sort_keys'
    var_12 = 'separators'
    var_13 = True
    var_14 = ','
    var_15 = ':'
    var_16 = (var_14, var_15)
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = module_0.Serializer(var_0, serializer_kwargs=var_17)
    var_19 = 'b'
    var_20 = 'a'
    var_21 = 2
    var_22 = {var_19: var_21, var_20: var_13}
    var_23 = var_18.dumps(var_22)
    var_24 = 'data1'
    var_25 = var_1.dumps(var_24)
    var_26 = 'data2'
    var_27 = var_1.dumps(var_26)
    var_28 = 'test'
    var_29 = 'custom_salt'
    var_30 = var_1.dumps(var_28, var_29)
    var_31 = var_1.loads(var_30)
    var_32 = 'old_key'
    var_33 = 'new_key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Serializer(var_34)
    var_36 = var_35.dumps(var_6)
    var_37 = var_35.loads(var_36)
    assert var_37 == 'test_data'
    var_38 = 123
    var_39 = var_1.dumps(var_38)
    var_40 = module_1.dumps(var_38)
    var_41 = {}
    var_42 = var_1.dumps(var_41)
    var_43 = var_1.loads(var_42)
    var_44 = None
    var_45 = var_1.dumps(var_44)
    var_46 = var_1.loads(var_45)
    assert var_46 is None



# Parsed testcases at query #145
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
    var_10 = b'custom-salt'
    var_11 = var_1.dumps(var_7, var_10)
    var_12 = 'sort_keys'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = 'b'
    var_16 = 'a'
    var_17 = 2
    var_18 = {var_15: var_17, var_16: var_13}
    var_19 = module_1.dumps(var_18)
    var_20 = module_1.loads(var_19)
    var_21 = {}
    var_22 = var_1.dumps(var_21)
    var_23 = var_1.loads(var_22)



# Parsed testcases at query #146
#--------------------------


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.loads(var_2)
    var_4 = '"string"'
    var_5 = module_0.loads(var_4)
    assert var_5 == 'string'
    var_6 = '42'
    var_7 = module_0.loads(var_6)
    assert var_7 == 42
    var_8 = 'true'
    var_9 = module_0.loads(var_8)
    assert var_9 is True
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = 'utf-8'
    var_13 = b'{"key": "value"}'
    var_14 = module_0.loads(var_13)
    var_15 = b'[1, 2, 3]'
    var_16 = module_0.loads(var_15)
    var_17 = 'hello'
    var_18 = module_0.loads(var_17)
    assert var_18 == 'HELLO'



# Parsed testcases at query #147
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
    var_16 = 'test_data'
    var_17 = module_0.loads(var_16)



# Parsed testcases at query #148
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's dumps method works correctly."
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dumps(var_5)
    assert var_6 == '{"key": "value", "number": 42}'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 'test'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.dumps(var_11)
    assert var_12 == '[1, 2, 3, "test"]'
    var_13 = 'hello'
    var_14 = module_0.dumps(var_13)
    assert var_14 == '"hello"'
    var_15 = None
    var_16 = module_0.dumps(var_15)
    assert var_16 == 'null'



# Parsed testcases at query #149
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
    var_8 = {var_2: var_3}
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_8, var_9)
    var_11 = 'different-secret'
    var_12 = module_0.Serializer(var_11)
    var_13 = {var_2: var_3}
    var_14 = var_12.dumps(var_13)
    var_15 = 123
    var_16 = var_1.dumps(var_15)
    var_17 = 'string'
    var_18 = var_1.dumps(var_17)
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = var_1.dumps(var_22)
    var_24 = None
    var_25 = var_1.dumps(var_24)



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'Test iter_unsigners method returns correct signers.'
    var_1 = b'test-secret-key'
    var_2 = b'test-salt'
    var_3 = 'digest_method'
    var_4 = 'sha256'
    var_5 = {var_3: var_4}
    var_6 = 'sha512'
    var_7 = {var_3: var_6}
    var_8 = 'sha384'
    var_9 = {var_3: var_8}
    var_10 = 1
    var_11 = 0
    var_12 = b'custom-salt'
    var_13 = len(var_3)
    var_14 = [var_7]
    var_15 = []
    var_16 = module_0.Serializer(var_1, var_2, fallback_signers=var_15)
    var_17 = var_16.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = module_0.Serializer(var_1, var_2)
    var_21 = var_20.iter_unsigners()
    var_22 = list(var_21)
    var_23 = len(var_22)
    var_24 = var_20.secret_keys
    var_25 = len(var_24)
    var_26 = var_20.default_fallback_signers
    var_27 = len(var_26)
    var_28 = var_25 * var_27
    var_29 = var_10 + var_28



# Parsed testcases at query #151
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '[1, 2, 3]'
    var_4 = var_0.loads(var_3)
    var_5 = '"hello"'
    var_6 = var_0.loads(var_5)
    assert var_6 == 'hello'
    var_7 = '42'
    var_8 = var_0.loads(var_7)
    assert var_8 == 42
    var_9 = 'null'
    var_10 = var_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = var_0.loads(var_11)
    assert var_12 is True
    var_13 = '{}'
    var_14 = var_0.loads(var_13)
    var_15 = 'hello'
    var_16 = var_0.loads(var_15)
    assert var_16 == 'HELLO'



# Parsed testcases at query #152
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
    var_10 = {var_2: var_3}
    var_11 = module_1.dumps(var_10)
    var_12 = module_1.loads(var_11)
    var_13 = {}
    var_14 = var_1.dumps(var_13)
    var_15 = var_1.loads(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_1.dumps(var_19)
    var_21 = var_1.loads(var_20)
    var_22 = None
    var_23 = var_1.dumps(var_22)
    var_24 = var_1.loads(var_23)
    assert var_24 is None
    var_25 = 'indent'
    var_26 = {var_25: var_17}
    var_27 = {var_2: var_3}
    var_28 = module_1.dumps(var_27)



# Parsed testcases at query #153
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
    var_7 = None
    var_8 = var_1.dumps(var_4, var_7)
    var_9 = 'custom-salt'
    var_10 = var_1.dumps(var_4, var_9)
    var_11 = var_1.dumps(var_4)
    var_12 = var_1.loads(var_11)
    var_13 = 'sort_keys'
    var_14 = True
    var_15 = {var_13: var_14}
    var_16 = module_0.Serializer(var_0, serializer_kwargs=var_15)
    var_17 = 'b'
    var_18 = 'a'
    var_19 = 2
    var_20 = {var_17: var_14, var_18: var_19}
    var_21 = var_16.dumps(var_20)
    var_22 = var_16.loads(var_21)



# Parsed testcases at query #154
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.dumps(var_4)
    assert var_5 == '{"key": "value", "number": 42}'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 'hello'
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = module_0.dumps(var_10)
    assert var_11 == '[1, 2, 3, "hello"]'
    var_12 = None
    var_13 = module_0.dumps(var_12)
    assert var_13 == 'null'
    var_14 = 'test_string'
    var_15 = module_0.dumps(var_14)
    assert var_15 == '"test_string"'
    var_16 = 123
    var_17 = module_0.dumps(var_16)
    assert var_17 == '123'
    var_18 = True
    var_19 = module_0.dumps(var_18)
    assert var_19 == 'true'



# Parsed testcases at query #155
#--------------------------


import json as module_0

def test_case_0():
    var_0 = "Test that _PDataSerializer protocol's loads method works with different implementations."
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = 'hello'
    var_4 = module_0.loads(var_3)
    assert var_4 == 'HELLO'
    var_5 = b'hello'
    var_6 = module_0.loads(var_5)
    assert var_6 == 'HELLO'
    var_7 = '42'
    var_8 = module_0.loads(var_7)
    assert var_8 == 42
    var_9 = '[1, 2, 3]'
    var_10 = module_0.loads(var_9)
    var_11 = '{"a": {"b": [1, 2]}}'
    var_12 = module_0.loads(var_11)
    var_13 = 'null'
    var_14 = module_0.loads(var_13)
    assert var_14 is None
    var_15 = 'true'
    var_16 = module_0.loads(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = module_0.loads(var_17)
    assert var_18 is False
    var_19 = 'invalid json'
    var_20 = module_0.loads(var_19)
    var_21 = '{broken}'
    var_22 = module_0.loads(var_21)



# Parsed testcases at query #156
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
    var_20 = module_0.Serializer(var_0, serializer_kwargs=var_19)
    var_21 = 'b'
    var_22 = 'a'
    var_23 = 2
    var_24 = {var_21: var_23, var_22: var_15}
    var_25 = var_20.dumps(var_24)
    var_26 = 'old-key'
    var_27 = 'new-key'
    var_28 = [var_26, var_27]
    var_29 = module_0.Serializer(var_28)
    var_30 = 'test data'
    var_31 = var_29.dumps(var_30)
    var_32 = var_29.loads(var_31)
    assert var_32 == 'test data'
    var_33 = 42
    var_34 = var_1.dumps(var_33)
    var_35 = None
    var_36 = var_1.dumps(var_35)
    var_37 = var_1.loads(var_36)
    assert var_37 is None
    var_38 = 3
    var_39 = [var_15, var_23, var_38]
    var_40 = var_1.dumps(var_39)
    var_41 = var_1.loads(var_40)
    var_42 = 'hello'
    var_43 = var_1.dumps(var_42)
    var_44 = var_1.loads(var_43)
    assert var_44 == 'hello'
    var_45 = 'key1'
    var_46 = module_0.Serializer(var_45)
    var_47 = 'key2'
    var_48 = module_0.Serializer(var_47)
    var_49 = 'test'
    var_50 = {var_49: var_15}
    var_51 = var_46.dumps(var_50)
    var_52 = var_48.dumps(var_50)



# Parsed testcases at query #157
#--------------------------


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'Test load_payload method of Serializer class.'
    var_1 = 'test-secret'
    var_2 = module_0.Serializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_1.dumps(var_5)
    var_7 = 'utf-8'
    var_8 = "{'key': 'value'}"
    var_9 = b'hello'
    var_10 = module_0.Serializer(var_1)
    var_11 = b'custom:data'
    var_12 = module_0.Serializer(var_1)
    var_13 = b'invalid json'
    var_14 = var_12.load_payload(var_13)
    var_15 = module_0.Serializer(var_14)
    var_16 = b''
    var_17 = var_15.load_payload(var_16)
    var_18 = module_0.Serializer(var_17)
    var_19 = 42
    var_20 = module_1.dumps(var_19)
    var_21 = var_18.load_payload(var_11)
    assert var_21 == 42
    var_22 = module_0.Serializer(var_17)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_1.dumps(var_26)
    var_28 = var_22.load_payload(var_11)
    var_29 = module_0.Serializer(var_17)
    var_30 = None
    var_31 = module_1.dumps(var_30)
    var_32 = var_29.load_payload(var_11)
    assert var_32 is None
    var_33 = module_0.Serializer(var_17)
    var_34 = True
    var_35 = module_1.dumps(var_34)
    var_36 = var_33.load_payload(var_11)
    assert var_36 is True
    var_37 = module_0.Serializer(var_17)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = 'c'
    var_41 = 'd'
    var_42 = {var_40: var_41}
    var_43 = [var_34, var_24, var_42]
    var_44 = {var_39: var_43}
    var_45 = {var_38: var_44}
    var_46 = module_1.dumps(var_45)
    var_47 = var_37.load_payload(var_11)
    var_48 = module_0.Serializer(var_17)
    var_49 = 'text'
    var_50 = 'héllo wörld 🎉'
    var_51 = {var_49: var_50}
    var_52 = module_1.dumps(var_51)
    var_53 = var_48.load_payload(var_11)
    var_54 = module_0.Serializer(var_17)
    var_55 = 'x'
    var_56 = 10000
    var_57 = var_55 * var_56
    var_58 = {var_3: var_57}
    var_59 = module_1.dumps(var_58)
    var_60 = var_54.load_payload(var_11)
    var_61 = module_0.Serializer(var_17)
    var_62 = 'message'
    var_63 = 'こんにちは世界'
    var_64 = {var_62: var_63}
    var_65 = module_1.dumps(var_64)
    var_66 = var_61.load_payload(var_11)
    var_67 = b'test payload'
    var_68 = var_61.load_payload(var_67)
    assert var_68 == 'loaded: test payload'



# Parsed testcases at query #158
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
    var_8 = b'{"key": "value"}'
    var_9 = module_0.loads(var_8)
    var_10 = 'null'
    var_11 = module_0.loads(var_10)
    assert var_11 is None
    var_12 = 'invalid json'
    var_13 = module_0.loads(var_12)



# Parsed testcases at query #159
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'Test that _PDataSerializer loads method works correctly with string and bytes payloads.'
    var_1 = '{"key": "value"}'
    var_2 = module_0.loads(var_1)
    var_3 = '42'
    var_4 = module_0.loads(var_3)
    assert var_4 == 42
    var_5 = 'null'
    var_6 = module_0.loads(var_5)
    assert var_6 is None
    var_7 = b'{"key": "value"}'
    var_8 = module_0.loads(var_7)
    var_9 = b'42'
    var_10 = module_0.loads(var_9)
    assert var_10 == 42
    var_11 = b'null'
    var_12 = module_0.loads(var_11)
    assert var_12 is None
    var_13 = '{}'
    var_14 = module_0.loads(var_13)
    var_15 = b'{}'
    var_16 = module_0.loads(var_15)
    var_17 = 'nested'
    var_18 = 'list'
    var_19 = 'bool'
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = True
    var_25 = {var_18: var_23, var_19: var_24}
    var_26 = {var_17: var_25}
    var_27 = module_0.dumps(var_26)
    var_28 = module_0.loads(var_27)
    var_29 = module_0.dumps(var_26)
    var_30 = 'utf-8'



# Parsed testcases at query #160
#--------------------------


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = module_0._PDataSerializer()
    var_1 = '{"key": "value"}'
    var_2 = var_0.loads(var_1)
    var_3 = '42'
    var_4 = var_0.loads(var_3)
    assert var_4 == 42
    var_5 = '[1, 2, 3]'
    var_6 = var_0.loads(var_5)
    var_7 = b'{"key": "value"}'
    var_8 = var_0.loads(var_7)
    var_9 = 'null'
    var_10 = var_0.loads(var_9)
    assert var_10 is None
    var_11 = 'true'
    var_12 = var_0.loads(var_11)
    assert var_12 is True
    assert var_12 == 'HELLO'
    var_13 = 'hello'
    var_14 = 'invalid json'
    var_15 = var_0.loads(var_14)



