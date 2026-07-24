####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_list. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/24 statements.


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
    var_17 = '{"key": "test"}'
    var_18 = '"test"'

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
    var_13 = 2
    var_14 = lambda x, y: (var_12, y + var_13)
    var_15 = (var_10, var_9)
    var_16 = lambda x, y, z: var_15
    var_17 = False
    var_18 = {}
    var_19 = '[1]'

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
    var_16 = {}
    var_17 = '123.45'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 3/21 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_tokenizing_json_object. Retrieved 1/11 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 12/18 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/13 statements.


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
    var_0 = '{"key": 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key'
    var_6 = 42
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = {var_5: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 42, "key2": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 42
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_6, var_6, var_0)
    var_8 = 'value'
    var_9 = 17
    var_10 = 21
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 42 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key'
    var_6 = 42
    var_7 = 8
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = {var_5: var_8}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": 42'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": 42 "key2": 43}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{123: 42}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 42
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested. Retrieved 12/13 statements.
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
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 7
    var_6 = 6
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 17
    var_6 = 16
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": {"b": [1, 2]}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = 1
    var_9 = [var_2, var_3, var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 6/7 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 11/16 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 10
    var_2 = 'match'
    var_3 = ()
    var_4 = 'end'
    var_5 = lambda s, end: type(var_0, var_1, {var_2: lambda self: end})()
    var_6 = ' \t\n\r'
    var_7 = 1
    var_8 = var_1 + var_7
    var_9 = 1
    var_10 = var_8 + var_9



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 5
    var_3 = lambda s, i: (ScalarToken(var_0, i, i + var_1, s), i + var_2)
    var_4 = '{"key": "value"}'
    var_5 = 0
    var_6 = (var_4, var_5)
    var_7 = True
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_6, var_7, var_3, var_8, var_4)
    var_10 = var_9[var_5]
    var_11 = len(var_10)
    assert var_11 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 5
    var_3 = lambda s, i: (ScalarToken(var_0, i, i + var_1, s), i + var_2)
    var_4 = '{"k1": "v1", "k2": "v2"}'
    var_5 = 0
    var_6 = (var_4, var_5)
    var_7 = True
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_6, var_7, var_3, var_8, var_4)
    var_10 = var_9[var_5]
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 5
    var_3 = lambda s, i: (ScalarToken(var_0, i, i + var_1, s), i + var_2)
    var_4 = '{ "key" : "value" }'
    var_5 = 0
    var_6 = (var_4, var_5)
    var_7 = True
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_6, var_7, var_3, var_8, var_4)
    var_10 = var_9[var_5]
    var_11 = len(var_10)
    assert var_11 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = 17
    var_5 = lambda s, i: (ScalarToken(var_2, i, i + var_3, s), i + var_4)
    var_6 = '{"outer": {"nested": "value"}}'
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = True
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_8, var_9, var_5, var_10, var_6)
    var_12 = var_11[var_7]
    var_13 = len(var_12)
    assert var_13 == 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 17/18 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 25/26 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 17/18 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, end: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 8
    var_7 = 12
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key": "value"}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = 13
    var_14 = (var_8, var_13)
    var_15 = lambda s, end: var_14
    var_16 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value1'
    var_6 = 8
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 17
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 24
    var_15 = 28
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = '{"key1": "value1", "key2": "value2"}'
    var_18 = 0
    var_19 = (var_17, var_18)
    var_20 = True
    var_21 = 14
    var_22 = 29
    var_23 = lambda s, end: (var_8 if end == var_6 else var_16, var_21 if end == var_6 else var_22)
    var_24 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 10
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key":  "value"}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = 15
    var_14 = (var_8, var_13)
    var_15 = lambda s, end: var_14
    var_16 = {}

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
    var_7 = lambda s, end: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, end: var_6
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
    var_7 = lambda s, end: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

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
    var_7 = lambda s, end: var_6
    var_8 = {}
    var_9 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 6/7 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 5/6 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 20/22 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_missing_colon. Retrieved 6/8 statements.
# Partially parsed test__TokenizingJSONObject_missing_comma. Retrieved 6/8 statements.
# Partially parsed test__TokenizingJSONObject_missing_closing_brace. Retrieved 6/8 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = var_0
    var_2 = 'value'
    var_3 = 8
    var_4 = 13
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_1)
    var_6 = 14
    var_7 = (var_5, var_6)
    var_8 = lambda s, end: var_7
    var_9 = 0
    var_10 = (var_0, var_9)
    var_11 = True
    var_12 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = var_0
    var_2 = 8
    var_3 = 'value1'
    var_4 = 9
    var_5 = 15
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_1)
    var_7 = 16
    var_8 = (var_6, var_7)
    var_9 = 'value2'
    var_10 = 25
    var_11 = 31
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_1)
    var_13 = 32
    var_14 = (var_12, var_13)
    var_15 = lambda s, end: var_8 if end == var_2 else var_14
    var_16 = 0
    var_17 = (var_0, var_16)
    var_18 = True
    var_19 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '  { "key" : "value" }  '
    var_1 = var_0
    var_2 = 'value'
    var_3 = 14
    var_4 = 19
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_1)
    var_6 = 20
    var_7 = (var_5, var_6)
    var_8 = lambda s, end: var_7
    var_9 = 0
    var_10 = (var_0, var_9)
    var_11 = True
    var_12 = {}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 13/14 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 18/19 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 13/14 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

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
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_12 = {var_10: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 9
    var_5 = 'value1'
    var_6 = 15
    var_7 = 'value2'
    var_8 = 25
    var_9 = 31
    var_10 = 7
    var_11 = lambda s, e: (ScalarToken(var_5, var_4, var_6, s) if e == var_4 else ScalarToken(var_7, var_8, var_9, s), e + var_10)
    var_12 = {}
    var_13 = 'key1'
    var_14 = 'key2'
    var_15 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_16 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_17 = {var_13: var_15, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 10
    var_6 = 15
    var_7 = 16
    var_8 = lambda s, e: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_12 = {var_10: var_11}

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

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 9
    var_5 = 'value1'
    var_6 = 15
    var_7 = 'value2'
    var_8 = 25
    var_9 = 31
    var_10 = 7
    var_11 = lambda s, e: (ScalarToken(var_5, var_4, var_6, s) if e == var_4 else ScalarToken(var_7, var_8, var_9, s), e + var_10)
    var_12 = {}
    var_13 = module_0._TokenizingJSONObject(var_2, var_3, var_11, var_12, var_0)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = '{"key" : value}'
    var_1 = 7



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 19/20 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 16/17 statements.
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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 4
    var_9 = module_2.Position(var_6, var_8, var_4)

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
    var_6 = module_2.Position(var_2, var_2, var_3)
    var_7 = 4
    var_8 = module_2.Position(var_2, var_7, var_4)
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = False
    var_12 = module_1.ScalarToken(var_11, var_11, var_7, var_9)
    var_13 = module_2.Position(var_2, var_2, var_11)
    var_14 = 5
    var_15 = module_2.Position(var_2, var_14, var_7)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = '3.14'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 3.14
    var_9 = 3
    var_10 = module_1.ScalarToken(var_8, var_3, var_9, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 14
    var_6 = 13
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup_key(var_9)
    var_11 = var_10.value
    assert var_11 == 'a'
    var_12 = [var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 1
    var_15 = 'b'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_json_array. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_object. Retrieved 19/20 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_whitespace. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 4
    var_9 = module_2.Position(var_6, var_8, var_4)

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
    var_6 = module_2.Position(var_2, var_2, var_3)
    var_7 = 4
    var_8 = module_2.Position(var_2, var_7, var_4)
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = False
    var_12 = module_1.ScalarToken(var_11, var_11, var_7, var_9)
    var_13 = module_2.Position(var_2, var_2, var_11)
    var_14 = 5
    var_15 = module_2.Position(var_2, var_14, var_7)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = '3.14'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 3.14
    var_9 = 3
    var_10 = module_1.ScalarToken(var_8, var_3, var_9, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup_key(var_9)
    var_11 = var_10.value
    assert var_11 == 'a'
    var_12 = [var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 1
    var_15 = 'b'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '  {  "a"  :  1  }  '
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test'

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
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 'test'
    var_2 = 0



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    var_5 = ''
    assert var_5 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_IndexError_handling_in_whitespace_skipping. Retrieved 18/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0
    var_5 = {}
    var_6 = None
    var_7 = 0
    var_8 = lambda s, end: (ScalarToken(var_6, var_7, var_7, s), end)
    var_9 = 'MockPattern'
    var_10 = ()
    var_11 = 'match'
    var_12 = ' \t\n\r'
    var_13 = (var_0, var_7)
    var_14 = False
    var_15 = 'key'
    var_16 = module_0.ScalarToken(var_6, var_14, var_14, var_0)
    var_17 = {var_15: var_16}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 12/13 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0
    var_5 = {}
    var_6 = None
    var_7 = lambda s, end: (ScalarToken(var_6, end, end, s), end + var_2)
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = False
    var_11 = len(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_handles_string_token. Retrieved 20/28 statements.
# Partially parsed test_make_scanner_handles_dict_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_handles_list_token. Retrieved 20/28 statements.
# Partially parsed test_make_scanner_handles_null_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_handles_true_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_handles_false_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_handles_number_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_handles_float_token. Retrieved 19/27 statements.


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
    var_18 = '{"test": "value"}'
    var_19 = 1

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
    var_18 = '{"test": "value"}'

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
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '[]'
    var_19 = '[1, 2, 3]'

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
    var_18 = 'null'

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
    var_18 = 'true'

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
    var_18 = 'false'

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
    var_18 = '123'

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
    var_18 = '123.45'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_index_error_in_whitespace_optimization. Retrieved 18/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0
    var_5 = {}
    var_6 = 'value'
    var_7 = lambda s, end: (ScalarToken(var_6, end, end, s), end + var_2)
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = False
    var_11 = 'key'
    var_12 = len(var_0)
    var_13 = var_12 - var_2
    var_14 = len(var_0)
    var_15 = var_14 - var_2
    var_16 = module_0.ScalarToken(var_6, var_13, var_15, var_0)
    var_17 = {var_11: var_16}



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_index_error_handling_in_tokenizing_json_object. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = var_0
    var_3 = {}
    var_4 = 'value'
    var_5 = lambda s, end: (ScalarToken(var_4, end, end, var_2), end)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = len(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 10/19 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 10/19 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 8/15 statements.


def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = ''

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = '"test"'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = 'false'
    var_7 = 0

import re as module_0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: float(x)
    var_3 = lambda self, x: int(x)
    var_4 = {}
    var_5 = False
    var_6 = '(-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)'
    var_7 = module_0.compile(var_6)
    var_8 = '123'
    var_9 = 0

import re as module_0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: float(x)
    var_3 = lambda self, x: int(x)
    var_4 = {}
    var_5 = False
    var_6 = '(-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)'
    var_7 = module_0.compile(var_6)
    var_8 = '123.45'
    var_9 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 2)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = '[]'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = {}
    var_5 = False
    var_6 = '{}'
    var_7 = 0



# Parsed testcases at query #2
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
    var_1 = 1

def test_case_0():
    var_0 = '{"test": null}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": true}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": false}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": 123}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": 123.45}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": []}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": {}}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_values. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_nested_objects. Retrieved 10/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_index_error_raises_stop_iteration. Retrieved 3/7 statements.


def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_IndexError_handling_in_whitespace_skipping. Retrieved 9/10 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = var_0
    var_3 = {}
    var_4 = 'value'
    var_5 = lambda s, end: (ScalarToken(var_4, end, end, var_2), end)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key":'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_2, var_6, var_4)
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = {var_5: var_5}
    var_12 = (var_0, var_3)
    var_13 = False
    var_14 = None
    var_15 = lambda s, end: (var_14, end)
    var_16 = module_1._TokenizingJSONObject(var_12, var_13, var_15, var_11, var_4)



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0._make_scanner(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 13/14 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 19/20 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 13/14 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 17/18 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, end: (ScalarToken(var_4, var_1, var_1, s), end)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 13
    var_7 = 14
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_12 = {var_10: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 9
    var_5 = 'value1'
    var_6 = 15
    var_7 = 16
    var_8 = 'value2'
    var_9 = 25
    var_10 = 31
    var_11 = 32
    var_12 = lambda s, end: (ScalarToken(var_5, var_4, var_6, s), var_7) if end == var_4 else (ScalarToken(var_8, var_9, var_10, s), var_11)
    var_13 = {}
    var_14 = 'key1'
    var_15 = 'key2'
    var_16 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_17 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_18 = {var_14: var_16, var_15: var_17}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '  {  "key"  :  "value"  }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 14
    var_6 = 19
    var_7 = 20
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_12 = {var_10: var_11}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 13
    var_7 = 14
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)

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
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 9
    var_5 = 'value1'
    var_6 = 15
    var_7 = 16
    var_8 = 'value2'
    var_9 = 25
    var_10 = 31
    var_11 = 32
    var_12 = lambda s, end: (ScalarToken(var_5, var_4, var_6, s), var_7) if end == var_4 else (ScalarToken(var_8, var_9, var_10, s), var_11)
    var_13 = {}
    var_14 = module_0._TokenizingJSONObject(var_2, var_3, var_12, var_13, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = '{'
    var_5 = 'value'
    var_6 = 20
    var_7 = 25
    var_8 = 26
    var_9 = {}
    var_10 = lambda s, end: _TokenizingJSONObject((s, end), var_3, lambda s, end: (ScalarToken(var_5, var_6, var_7, s), var_8), var_9, s) if s[end] == var_4 else (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_11 = {}
    var_12 = 'outer'
    var_13 = 'inner'
    var_14 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 16/17 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = var_0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'value'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8)
    var_10 = 'key'
    var_11 = 8
    var_12 = 12
    var_13 = module_0.ScalarToken(var_6, var_11, var_12, var_1)
    var_14 = {var_10: var_13}
    var_15 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = var_0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'value1'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8)
    var_10 = 'key1'
    var_11 = 8
    var_12 = 13
    var_13 = module_0.ScalarToken(var_6, var_11, var_12, var_1)
    var_14 = {var_10: var_13}
    var_15 = len(var_0)



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    assert var_3 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_line_72_predicate_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = '"key2"'
    var_2 = len(var_1)
    var_3 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 28/29 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 25/26 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 4
    var_9 = module_2.Position(var_6, var_8, var_4)

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
    var_6 = module_2.Position(var_2, var_2, var_3)
    var_7 = 4
    var_8 = module_2.Position(var_2, var_7, var_4)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = 1
    var_6 = module_2.Position(var_5, var_5, var_2)
    var_7 = 5
    var_8 = module_2.Position(var_5, var_7, var_3)

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
    var_6 = module_2.Position(var_4, var_4, var_3)
    var_7 = 2
    var_8 = module_2.Position(var_4, var_7, var_4)

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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 4
    var_9 = module_2.Position(var_6, var_8, var_4)

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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 7
    var_9 = module_2.Position(var_6, var_8, var_4)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = module_2.ScalarToken(var_2, var_2, var_2, var_0)
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = 2
    var_14 = 3
    var_15 = module_2.ScalarToken(var_13, var_14, var_14, var_0)
    var_16 = [var_13]
    var_17 = var_1.lookup(var_16)
    var_18 = 5
    var_19 = module_2.ScalarToken(var_14, var_18, var_18, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = 6
    var_12 = module_2.ScalarToken(var_2, var_11, var_11, var_0)
    var_13 = 'b'
    var_14 = [var_13]
    var_15 = var_1.lookup(var_14)
    var_16 = 2
    var_17 = 13
    var_18 = module_2.ScalarToken(var_16, var_17, var_17, var_0)
    var_19 = [var_8]
    var_20 = var_1.lookup_key(var_19)
    var_21 = 3
    var_22 = module_2.ScalarToken(var_8, var_2, var_21, var_0)
    var_23 = [var_13]
    var_24 = var_1.lookup_key(var_23)
    var_25 = 9
    var_26 = 11
    var_27 = module_2.ScalarToken(var_13, var_25, var_26, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = 1
    var_7 = 7
    var_8 = module_1.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = [var_2, var_6]
    var_10 = var_1.lookup(var_9)
    var_11 = 2
    var_12 = 9
    var_13 = module_1.ScalarToken(var_11, var_12, var_12, var_0)
    var_14 = 'b'
    var_15 = 'c'
    var_16 = [var_14, var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = 3
    var_19 = 20
    var_20 = module_1.ScalarToken(var_18, var_19, var_19, var_0)
    var_21 = [var_14, var_15]
    var_22 = var_1.lookup_key(var_21)
    var_23 = 18
    var_24 = module_1.ScalarToken(var_15, var_23, var_19, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1,}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


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
    var_6 = 1
    var_7 = module_2.Position(var_6, var_6, var_3)
    var_8 = 4
    var_9 = module_2.Position(var_6, var_8, var_4)

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
    var_6 = module_2.Position(var_2, var_2, var_3)
    var_7 = 4
    var_8 = module_2.Position(var_2, var_7, var_4)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = 1
    var_6 = module_2.Position(var_5, var_5, var_2)
    var_7 = 5
    var_8 = module_2.Position(var_5, var_7, var_3)

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
    var_6 = module_2.Position(var_4, var_4, var_3)
    var_7 = 2
    var_8 = module_2.Position(var_4, var_7, var_4)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 7
    var_6 = 6
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup_key(var_9)
    var_11 = var_10.value
    assert var_11 == 'a'
    var_12 = [var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 1
    var_15 = 'b'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'b'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1,}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 10/12 statements.


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
    var_5 = 'value'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
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
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, i: (ScalarToken(var_5, i, i + var_6, s), i + var_7)
    var_9 = len(var_0)



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0._make_scanner(var_0, var_1)



