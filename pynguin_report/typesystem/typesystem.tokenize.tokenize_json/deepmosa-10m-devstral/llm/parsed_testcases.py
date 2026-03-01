####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/15 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 2/20 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": null}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": true}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": false}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": 123}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": 123.45}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": []}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": {}}'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = bool(not ('n' == 'x' and 'x'[:4] == 'null'))
    assert var_0 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 23/32 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/25 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = True
    var_17 = {}
    var_18 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = True
    var_17 = {}
    var_18 = {}
    var_19 = 2
    var_20 = (var_18, var_19)
    var_21 = lambda x, y, z, w, c: var_20
    var_22 = '{"key": 1}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 2
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = True
    var_17 = {}
    var_18 = '[]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = {}
    var_17 = '123.45'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = True
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = lambda s, e: var_4
    var_6 = '{"key": "value"}'
    var_7 = (var_6, var_1)
    var_8 = True
    var_9 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = lambda s, e: var_4
    var_6 = '{"key1": "value1", "key2": "value2"}'
    var_7 = (var_6, var_1)
    var_8 = True
    var_9 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = lambda s, e: var_4
    var_6 = '{ "key" : "value" }'
    var_7 = (var_6, var_1)
    var_8 = True
    var_9 = {}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 11/12 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = var_1.string
    assert var_8 == 'null'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = var_1.string
    assert var_8 == 'true'
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = False
    var_12 = 4
    var_13 = module_1.ScalarToken(var_11, var_11, var_12, var_9)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True
    var_15 = var_10.value
    assert var_15 is False
    var_16 = var_10.string
    assert var_16 == 'false'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = var_1.string
    assert var_8 == '42'
    var_9 = '3.14'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 3.14
    var_12 = 3
    var_13 = module_1.ScalarToken(var_11, var_3, var_12, var_9)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True
    var_15 = var_10.value
    var_16 = bool(var_10.value == 3.14)
    assert var_16 is True
    var_17 = var_10.string
    assert var_17 == '3.14'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = var_1.string
    assert var_8 == '"hello"'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_json_with_valid_json. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 13/17 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 8
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = 6
    var_9 = 7
    var_10 = lambda s, i: (ScalarToken(var_6 if i == var_5 else var_7, i, i + var_8, s), i + var_9)
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 9
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = 6
    var_9 = 7
    var_10 = lambda s, i: (ScalarToken(var_6 if i == var_5 else var_7, i, i + var_8, s), i + var_9)
    var_11 = module_0._TokenizingJSONObject(var_3, var_4, var_10, var_1, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting value'

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = 17
    var_9 = lambda s, i: (_TokenizingJSONObject((s, i), var_4, lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7), var_1, var_0)[var_2], i + var_8)
    var_10 = 'outer'
    var_11 = len(var_0)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    var_5 = bool(var_4 != '')
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_index_error_handling_in_tokenizing_json_object. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = 'value'
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, var_0), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = False
    var_8 = len(var_0)
    var_9 = bool(var_1 == var_8)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/23 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '"test"'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = '[1]'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 11/13 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, i: (ScalarToken(var_5, i, i, s), i + var_4)
    var_7 = module_0._TokenizingJSONObject(var_3, var_4, var_6, var_1, var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 7
    var_10 = 6
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2
    var_23 = [var_15]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'a'
    var_26 = [var_19]
    var_27 = var_1.lookup_key(var_26)
    var_28 = var_27.value
    assert var_28 == 'b'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Expecting'



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    assert var_4 == ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/7 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = lambda s, i, strict: ('value', i + 6)
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '{"key": "value"}'
    var_7 = 1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '{"key": null}'
    var_7 = 1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '{"key": true}'
    var_7 = 1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '{"key": false}'
    var_7 = 1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = lambda x: float(x)
    var_3 = lambda x: int(x)
    var_4 = None
    var_5 = False
    var_6 = '{"key": 123}'
    var_7 = 1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = lambda x: float(x)
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '{"key": 123.45}'
    var_7 = 1

def test_case_0():
    var_0 = lambda s, i, strict: ('key', i + 4)
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = False
    var_6 = '{"key": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = lambda x, y: ([1, 2, 3], 7)
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = False
    var_6 = '[1, 2, 3]'
    var_7 = 0



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True
    var_14 = var_4.string
    assert var_14 == 'null'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_json_raises_parse_error_on_invalid_json. Retrieved 3/5 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '.'
    var_3 = bool(var_1)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True
    var_14 = var_4.string
    assert var_14 == 'null'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_list. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/24 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"test": "value"}'
    var_19 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 4
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": "value"}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = 1
    var_9 = 0
    var_10 = ''
    var_11 = module_0.ScalarToken(var_8, var_9, var_9, var_10)
    var_12 = [var_11]
    var_13 = 3
    var_14 = (var_12, var_13)
    var_15 = lambda x, y: var_14
    var_16 = (var_10, var_8)
    var_17 = lambda x, y, z: var_16
    var_18 = False
    var_19 = {}
    var_20 = '[1, 2, 3]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 5/8 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 5/8 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}

def test_case_0():
    var_0 = '{"a": 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"a": 42, "b": 43}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{ "a" : 42 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 42'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a" 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 42 "b": 43}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    var_5 = bool(var_4 != '')
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 20/21 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 18/19 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = 1
    var_6 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_5)
    var_7 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 8
    var_11 = 13
    var_12 = module_0.ScalarToken(var_4, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 8
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s) if e == var_4 else ScalarToken(var_7, e, e + var_6, s), e + var_8)
    var_10 = {}
    var_11 = 'key1'
    var_12 = 'key2'
    var_13 = 13
    var_14 = module_0.ScalarToken(var_5, var_4, var_13, var_0)
    var_15 = 22
    var_16 = 27
    var_17 = module_0.ScalarToken(var_7, var_15, var_16, var_0)
    var_18 = {var_11: var_14, var_12: var_17}
    var_19 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_4, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value",}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = 1
    var_6 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_5)
    var_7 = {}
    var_8 = module_0._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'inner'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 17
    var_8 = 18
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = 'outer'
    var_12 = {var_4: var_5}
    var_13 = 9
    var_14 = 26
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = {var_11: var_15}
    var_17 = len(var_0)



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = var_4.start.line_no
    assert var_7 == 1
    var_8 = var_4.end.line_no
    assert var_8 == 1
    var_9 = var_4.string
    assert var_9 == 'null'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_empty_string_raises_stop_iteration. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_string_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_null_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_true_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_false_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_integer_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_float_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_object_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_array_value. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0

def test_case_0():
    var_0 = '"test"'
    var_1 = '"test"'
    var_2 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 'true'
    var_2 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 'false'
    var_2 = 0

def test_case_0():
    var_0 = '123'
    var_1 = '123'
    var_2 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = '123.45'
    var_2 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"}'
    var_2 = 0

def test_case_0():
    var_0 = '["value"]'
    var_1 = '["value"]'
    var_2 = 0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test'
    var_2 = 0



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_index_error_predicate_false. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 10
    var_2 = 'Match'
    var_3 = ()
    var_4 = 'end'
    var_5 = lambda s, end: type(var_2, var_3, {var_4: lambda self: end})()
    var_6 = ' '
    var_7 = 1
    var_8 = var_1 + var_7
    var_9 = 1
    var_10 = var_8 + var_9
    var_11 = var_5(var_0, var_10)
    var_12 = bool(True)
    assert var_12 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 20/21 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 14/15 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'value'
    var_2 = 8
    var_3 = 13
    var_4 = 14
    var_5 = lambda s, e: (ScalarToken(var_1, var_2, var_3, s), var_4)
    var_6 = '{"key": "value"}'
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = False
    var_10 = {}
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_13 = {var_11: var_12}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 8
    var_2 = 'value1'
    var_3 = 13
    var_4 = 14
    var_5 = 'value2'
    var_6 = 24
    var_7 = 29
    var_8 = 30
    var_9 = lambda s, e: (ScalarToken(var_2, var_1, var_3, s), var_4) if e == var_1 else (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = '{"key1": "value1", "key2": "value2"}'
    var_11 = 0
    var_12 = (var_10, var_11)
    var_13 = False
    var_14 = {}
    var_15 = 'key1'
    var_16 = 'key2'
    var_17 = module_0.ScalarToken(var_2, var_1, var_3, var_0)
    var_18 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_19 = {var_15: var_17, var_16: var_18}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 'value'
    var_2 = 10
    var_3 = 15
    var_4 = 16
    var_5 = lambda s, e: (ScalarToken(var_1, var_2, var_3, s), var_4)
    var_6 = '{"key":  "value"}'
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = False
    var_10 = {}
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_13 = {var_11: var_12}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Expecting value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 22/28 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": "value"}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = 'value'
    var_9 = 0
    var_10 = 4
    var_11 = '["value"]'
    var_12 = module_0.ScalarToken(var_8, var_9, var_10, var_11)
    var_13 = [var_12]
    var_14 = 8
    var_15 = (var_13, var_14)
    var_16 = lambda x, y: var_15
    var_17 = 7
    var_18 = (var_8, var_17)
    var_19 = lambda x, y, z: var_18
    var_20 = False
    var_21 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0._make_scanner(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 5/11 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False
    var_3 = {}
    var_4 = ''



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/19 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/14 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '"test"'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = 0

def test_case_0():
    var_0 = ''
    var_1 = 0



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_index_error_in_whitespace_skip. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = var_0[var_1 + 1]
    var_4 = 1
    var_5 = var_1 + var_4



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    assert var_5 == 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_tokenize_json_with_valid_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_handles_string_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_dict_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_list_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_null_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_true_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_false_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_handles_number_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/14 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 1

def test_case_0():
    var_0 = '["test"]'
    var_1 = 1

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_list. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/24 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 5
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"test": "value"}'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 3
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"key": "value"}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = 1
    var_9 = 0
    var_10 = ''
    var_11 = module_0.ScalarToken(var_8, var_9, var_9, var_10)
    var_12 = [var_11]
    var_13 = 3
    var_14 = lambda x, y: (var_12, y + var_13)
    var_15 = (var_10, var_9)
    var_16 = lambda x, y, z: var_15
    var_17 = False
    var_18 = {}
    var_19 = '[1, 2, 3]'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 20/21 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 15/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = 1
    var_6 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_5)
    var_7 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = False
    var_6 = 'value'
    var_7 = 4
    var_8 = 5
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8)
    var_10 = 'key'
    var_11 = 8
    var_12 = 11
    var_13 = module_0.ScalarToken(var_6, var_11, var_12, var_0)
    var_14 = {var_10: var_13}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = '{"key1": "value1", "key2": "value2"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = False
    var_6 = 8
    var_7 = 'value1'
    var_8 = 6
    var_9 = 'value2'
    var_10 = 7
    var_11 = lambda s, e: (ScalarToken(var_7, e, e + var_8, s) if e == var_6 else ScalarToken(var_9, e, e + var_8, s), e + var_10)
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 13
    var_15 = module_0.ScalarToken(var_7, var_6, var_14, var_0)
    var_16 = 23
    var_17 = 28
    var_18 = module_0.ScalarToken(var_9, var_16, var_17, var_0)
    var_19 = {var_12: var_15, var_13: var_18}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = {}
    var_2 = '{"key":  "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = False
    var_6 = 'value'
    var_7 = 4
    var_8 = 5
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8)
    var_10 = 'key'
    var_11 = 10
    var_12 = 13
    var_13 = module_0.ScalarToken(var_6, var_11, var_12, var_0)
    var_14 = {var_10: var_13}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 8
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s) if e == var_4 else ScalarToken(var_7, e, e + var_6, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_scalar_token_creation_for_string. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 24/25 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 12
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 13
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = 'key'
    var_13 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_14 = {var_12: var_13}
    var_15 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 13
    var_7 = module_0.ScalarToken(var_5, var_4, var_6)
    var_8 = 14
    var_9 = (var_7, var_8)
    var_10 = 'value2'
    var_11 = 23
    var_12 = 28
    var_13 = module_0.ScalarToken(var_10, var_11, var_12)
    var_14 = 29
    var_15 = (var_13, var_14)
    var_16 = lambda s, e: var_9 if e == var_4 else var_15
    var_17 = {}
    var_18 = 'key1'
    var_19 = 'key2'
    var_20 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_21 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 9
    var_6 = 14
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 15
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = 'key'
    var_13 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_14 = {var_12: var_13}
    var_15 = len(var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{123'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" 123'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key":'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 13
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 14
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_10, var_11, var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value. Retrieved 10/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_values. Retrieved 15/31 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/13 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 13
    var_7 = 14
    var_8 = lambda s, e: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 9
    var_7 = 15
    var_8 = 'value2'
    var_9 = 25
    var_10 = 31
    var_11 = 16
    var_12 = 32
    var_13 = lambda s, e: (ScalarToken(var_5, var_6, var_7, s) if e == var_4 else ScalarToken(var_8, var_9, var_10, s), var_11 if e == var_4 else var_12)
    var_14 = {}

def test_case_0():
    var_0 = '  {  "key"  :  "value"  }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 16
    var_6 = 21
    var_7 = 22
    var_8 = lambda s, e: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 13
    var_7 = 14
    var_8 = lambda s, e: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 9
    var_7 = 15
    var_8 = 'value2'
    var_9 = 25
    var_10 = 31
    var_11 = 16
    var_12 = 32
    var_13 = lambda s, e: (ScalarToken(var_5, var_6, var_7, s) if e == var_4 else ScalarToken(var_8, var_9, var_10, s), var_11 if e == var_4 else var_12)
    var_14 = {}
    var_15 = module_0._TokenizingJSONObject(var_2, var_3, var_13, var_14, var_0)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 12
    var_7 = 13
    var_8 = lambda s, e: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = bool(not ('null' == 'n' and 'null'[:4] == 'null'))
    assert var_0 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 5/8 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 5/10 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 5/8 statements.
# Partially parsed test__TokenizingJSONObject_missing_closing_brace. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (Token(var_4, e, e, s), e + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"a": 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"a": 42, "b": 3.14}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{ "a" : 42 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"a": 42'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting'

def test_case_0():
    var_0 = '{"a" 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 42 "b": 3.14}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{"a": {"b": 42}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 3/8 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/8 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 2/8 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test content'

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = len(var_0)

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '[]'
    var_3 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = 'test'
    var_1 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_tokenizing_json_object. Retrieved 9/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = True
    var_8 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 6/7 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = var_1.string
    assert var_8 == 'null'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = var_1.string
    assert var_8 == 'true'
    var_9 = module_2.Position(var_2, var_2, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 4
    var_13 = module_2.Position(var_2, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = bool(var_1 == var_4)
    assert var_5 is True
    var_6 = var_1.value
    assert var_6 is False
    var_7 = var_1.string
    assert var_7 == 'false'
    var_8 = 1
    var_9 = module_2.Position(var_8, var_8, var_2)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 5
    var_13 = module_2.Position(var_8, var_12, var_3)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = var_1.string
    assert var_8 == '42'
    var_9 = module_2.Position(var_4, var_4, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 2
    var_13 = module_2.Position(var_4, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = var_1.string
    assert var_8 == '"hello"'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 7
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2
    var_23 = [var_15]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'a'
    var_26 = [var_19]
    var_27 = var_1.lookup_key(var_26)
    var_28 = var_27.value
    assert var_28 == 'b'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3
    var_18 = [var_4]
    var_19 = var_1.lookup_key(var_18)
    var_20 = var_19.value
    assert var_20 == 'a'
    var_21 = [var_13]
    var_22 = var_1.lookup_key(var_21)
    var_23 = var_22.value
    assert var_23 == 'b'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 10
    var_4 = 9
    var_5 = module_1.Position(var_2, var_3, var_4)



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    assert var_4 == ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_nextchar_not_quote_raises_error. Retrieved 11/12 statements.


def test_case_0():
    var_0 = '{"key": "value", }'
    var_1 = 11
    var_2 = {}
    var_3 = 'value'
    var_4 = 8
    var_5 = 12
    var_6 = 13
    var_7 = lambda s, end: (ScalarToken(var_3, var_4, var_5, s), var_6)
    var_8 = (var_0, var_1)
    var_9 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_json_simple_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 10/11 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 11/12 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 7
    var_10 = 6
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, "two", false]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two', False])
    assert var_3 is True
    var_4 = var_1._value
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 15
    var_12 = 14
    var_13 = module_1.Position(var_6, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": "two"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 'two'})
    assert var_3 is True
    var_4 = var_1._value
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 17
    var_12 = 16
    var_13 = module_1.Position(var_6, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"key": "value"}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'nested': {'key': 'value'}})
    assert var_3 is True
    var_4 = 'list'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'nested'
    var_10 = 'key'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 'value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Expecting'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 5/20 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/16 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = None
    var_3 = module_0.Token(var_2, var_1, var_1, var_0)
    var_4 = (var_3, var_1)

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = ''
    var_1 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 28/29 statements.
# Partially parsed test_tokenize_json_nested. Retrieved 21/22 statements.
# Partially parsed test_tokenize_json_bytes. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = var_1.string
    assert var_8 == 'null'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = var_1.string
    assert var_8 == 'true'
    var_9 = module_2.Position(var_2, var_2, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 4
    var_13 = module_2.Position(var_2, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = bool(var_1 == var_4)
    assert var_5 is True
    var_6 = var_1.value
    assert var_6 is False
    var_7 = var_1.string
    assert var_7 == 'false'
    var_8 = 1
    var_9 = module_2.Position(var_8, var_8, var_2)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 5
    var_13 = module_2.Position(var_8, var_12, var_3)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = var_1.string
    assert var_8 == '42'
    var_9 = module_2.Position(var_4, var_4, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 2
    var_13 = module_2.Position(var_4, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 3.14
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    var_8 = bool(var_1.value == 3.14)
    assert var_8 is True
    var_9 = var_1.string
    assert var_9 == '3.14'
    var_10 = 1
    var_11 = module_2.Position(var_10, var_10, var_3)
    var_12 = var_1.start
    var_13 = bool(var_1.start == var_11)
    assert var_13 is True
    var_14 = 4
    var_15 = module_2.Position(var_10, var_14, var_4)
    var_16 = var_1.end
    var_17 = bool(var_1.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = var_1.string
    assert var_8 == '"hello"'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 7
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = module_2.ScalarToken(var_5, var_5, var_5, var_0)
    var_18 = bool(var_16 == var_17)
    assert var_18 is True
    var_19 = [var_5]
    var_20 = var_1.lookup(var_19)
    var_21 = 2
    var_22 = 3
    var_23 = module_2.ScalarToken(var_21, var_22, var_22, var_0)
    var_24 = bool(var_20 == var_23)
    assert var_24 is True
    var_25 = [var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = 5
    var_28 = module_2.ScalarToken(var_22, var_27, var_27, var_0)
    var_29 = bool(var_26 == var_28)
    assert var_29 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = 3
    var_19 = module_2.ScalarToken(var_15, var_5, var_18, var_0)
    var_20 = bool(var_17 == var_19)
    assert var_20 is True
    var_21 = [var_15]
    var_22 = var_1.lookup(var_21)
    var_23 = 5
    var_24 = module_2.ScalarToken(var_5, var_23, var_23, var_0)
    var_25 = bool(var_22 == var_24)
    assert var_25 is True
    var_26 = 'b'
    var_27 = [var_26]
    var_28 = var_1.lookup_key(var_27)
    var_29 = 8
    var_30 = 10
    var_31 = module_2.ScalarToken(var_26, var_29, var_30, var_0)
    var_32 = bool(var_28 == var_31)
    assert var_32 is True
    var_33 = [var_26]
    var_34 = var_1.lookup(var_33)
    var_35 = 2
    var_36 = 12
    var_37 = module_2.ScalarToken(var_35, var_36, var_36, var_0)
    var_38 = bool(var_34 == var_37)
    assert var_38 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = 1
    var_9 = 7
    var_10 = module_1.ScalarToken(var_8, var_9, var_9, var_0)
    var_11 = bool(var_7 == var_10)
    assert var_11 is True
    var_12 = [var_4, var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = 2
    var_15 = 9
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_0)
    var_17 = bool(var_13 == var_16)
    assert var_17 is True
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_18, var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = 3
    var_23 = 20
    var_24 = module_1.ScalarToken(var_22, var_23, var_23, var_0)
    var_25 = bool(var_21 == var_24)
    assert var_25 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1}'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a":}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_index_error_in_whitespace_handling. Retrieved 10/14 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = 8
    var_2 = '\\s*'
    var_3 = lambda s, end: re.match(var_2, s[end:])
    var_4 = ' \t\n\r'
    var_5 = 1
    var_6 = var_1 + var_5
    var_7 = 1
    var_8 = var_6 + var_7
    var_9 = var_3(var_0, var_8)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_values. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 7/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 10
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, i: (ScalarToken(var_5 if i < var_4 else var_6, i, i + var_7, s), i + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '  {  "key"  :  "value"  }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 10
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, i: (ScalarToken(var_5 if i < var_4 else var_6, i, i + var_7, s), i + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'outer'
    var_6 = len(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_index_error_in_whitespace_skip. Retrieved 11/12 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = {}
    var_5 = 'value'
    var_6 = lambda s, end: (ScalarToken(var_5, end, end, s), end)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/14 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = ''
    assert var_3 == ''



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = ''
    var_5 = var_0[var_3]
    var_6 = bool(var_5 != '')
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 10
    var_2 = var_0[var_1]



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = ''
    assert var_3 == ''



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_tokenize_json_raises_parse_error_on_invalid_json. Retrieved 3/7 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '.'
    var_3 = bool(var_1)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 10
    var_2 = 'Match'
    var_3 = ()
    var_4 = 'end'
    var_5 = lambda s, end: type(var_2, var_3, {var_4: lambda self: end})()
    var_6 = ' '
    var_7 = 1
    var_8 = var_1 + var_7
    var_9 = 1
    var_10 = var_8 + var_9
    var_11 = var_5(var_0, var_10)
    var_12 = bool(True)
    assert var_12 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = ''
    assert var_3 == ''



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_tokenize_json_with_invalid_json. Retrieved 3/7 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '.'
    var_3 = bool(var_1)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    assert var_3 == ''



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 7/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 9/17 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 7/13 statements.
# Partially parsed test__TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, i: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key'
    var_6 = module_0.ScalarToken(var_5, var_1, var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key1'
    var_6 = module_0.ScalarToken(var_5, var_1, var_1)
    var_7 = 'key2'
    var_8 = module_0.ScalarToken(var_7, var_1, var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key'
    var_6 = module_0.ScalarToken(var_5, var_1, var_1)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, i: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, i: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, i: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

def test_case_0():
    var_0 = '{"key":'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 7/10 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = ''



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 10
    var_2 = 'MockMatch'
    var_3 = ()
    var_4 = 'end'
    var_5 = 1
    var_6 = lambda s, end: type(var_2, var_3, {var_4: lambda self: end + var_5})()
    var_7 = ' '
    var_8 = bool(not (var_0[var_1] in var_7 and var_0[var_1 + 1] in var_7))
    assert var_8 is True



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 10
    var_4 = 9
    var_5 = module_1.Position(var_2, var_3, var_4)



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_line_61_predicate_false. Retrieved 8/11 statements.


import re as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    var_4 = 1
    var_5 = var_1 + var_4
    var_6 = module_0.match(var_0, var_5)
    var_7 = var_0[var_1]
    assert var_7 == ''



