####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_with_null. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_number. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 6/14 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = '123'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = '12.34'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('parsed', idx + 8)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = '"string"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan_once: ([], args[1])
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = lambda self, s, idx, strict: ('', idx)
    var_5 = lambda self, args, scan_once: ([], args[1])
    var_6 = 'null'
    var_7 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 7
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 11
    var_13 = 12
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 15
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_10, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 13
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
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
    var_0 = '[1, "two", false]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two', False])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, "two", false]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 18
    var_11 = 17
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"inner": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'inner'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = var_1.value
    var_12 = bool(var_1.value == var_10)
    assert var_12 is True
    var_13 = var_1.string
    assert var_13 == '{"list": [1, 2], "nested": {"inner": true}}'
    var_14 = 0
    var_15 = module_1.Position(var_8, var_8, var_14)
    var_16 = var_1.start
    var_17 = bool(var_1.start == var_15)
    assert var_17 is True
    var_18 = 44
    var_19 = 43
    var_20 = module_1.Position(var_8, var_18, var_19)
    var_21 = var_1.end
    var_22 = bool(var_1.end == var_20)
    assert var_22 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/15 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/15 statements.
# Partially parsed test_make_scanner_empty_object. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_empty_array. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_nested_structure. Retrieved 7/21 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = lambda self, idx_scan: ([], idx_scan[1])
    var_3 = {}
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('value', idx + 7)
    var_2 = {}
    var_3 = '{"key": ["item"]}'
    var_4 = 0
    var_5 = 'key'
    var_6 = 'key'
    var_7 = len(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/22 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/23 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/23 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 14/21 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 7
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 2
    var_15 = 14
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_0)
    var_17 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = len(var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = [var_2]
    var_4 = (var_0, var_2)
    var_5 = True
    var_6 = 'key'
    var_7 = 5
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'value2'
    var_10 = 23
    var_11 = 28
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = len(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
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
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [{"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [{'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [{"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 21
    var_11 = 20
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": }'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 10
    var_11 = module_1.Position(var_9, var_4, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/19 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/22 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/20 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_invalid_key. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, idx: (ScalarToken(var_5, idx, idx, var_0), idx + var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 6
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 2
    var_15 = 13
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_json_does_not_raise_parse_error_on_valid_json. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = []
    var_2 = 'content'
    var_3 = {var_2: var_0}
    var_4 = module_0._TokenizingDecoder(*var_1, **var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_stop_iteration_on_empty_string. Retrieved 6/12 statements.
# Partially parsed test_make_scanner_memo_cleared_after_scan. Retrieved 6/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '42'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '3.14'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = lambda self, string_idx, scan_once: ([], 2)
    var_3 = {}
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = ''
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '"test"'
    var_7 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 4/15 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 4/15 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 4/15 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 4/15 statements.
# Partially parsed test_make_scanner_scalar_number_integer. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_number_float. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 5/16 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 4/14 statements.
# Partially parsed test_make_scanner_stop_iteration_on_empty_string. Retrieved 4/14 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid_char. Retrieved 4/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '42'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '3.14'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = len(var_2)

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '["item"]'
    var_3 = 0
    var_4 = len(var_2)

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = ''
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 7
    var_10 = 9
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ' { "key" : 123 } '
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 10
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 5
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 9
    var_13 = 10
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 5/35 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
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
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [{"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [{'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [{"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 21
    var_11 = 20
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'"bytes"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'bytes'
    var_3 = var_1.string
    assert var_3 == '"bytes"'
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

def test_case_0():
    var_0 = '{"invalid": }'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Expecting value'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_number. Retrieved 8/19 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 6/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = lambda self, string_idx, scan_once: ([ScalarToken(1, 2, 2, '[1]')], 3)
    var_3 = {}
    var_4 = '[1]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

import re as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '(-?\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?)'
    var_5 = module_0.compile(var_4)
    var_6 = '42'
    var_7 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '"test"'
    var_7 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 1/12 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 12/31 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = ''
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/19 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/30 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/20 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/15 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    assert var_2 == 2
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 8
    var_11 = 9
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 6
    var_14 = module_0.ScalarToken(var_5, var_13, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_14, var_12: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 10
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 18/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 10
    var_2 = 14
    var_3 = '{"key":"value"}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = {}
    var_9 = '{"key":"value"}'
    var_10 = 0
    var_11 = (var_3, var_10)
    var_12 = True
    var_13 = 'key'
    var_14 = module_0.ScalarToken(var_0, var_1, var_2, var_9)
    var_15 = {var_13: var_14}
    var_16 = 16
    var_17 = (var_15, var_16)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 4/20 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_stop_iteration_on_empty_string. Retrieved 4/16 statements.
# Partially parsed test_make_scanner_memo_cleared_after_scan. Retrieved 4/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '42'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '3.14'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '[1, 2]'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = ''
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = '"test"'
    var_5 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_list. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_dict. Retrieved 6/20 statements.
# Partially parsed test_make_scanner_empty_string. Retrieved 6/14 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 6/14 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '42'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '3.14'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = lambda self, string_idx, scan_once: ([ScalarToken(1, 1, 1, ''), ScalarToken(2, 3, 3, '')], 5)
    var_3 = {}
    var_4 = '[1, 2]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = ''
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '"test"'
    var_7 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_json_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_list_simple. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_dict_simple. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_invalid_bytes. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

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
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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
    var_4 = var_1.string
    assert var_4 == '[1, "two", false]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 18
    var_11 = 17
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"inner": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'inner'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = var_1.value
    var_12 = bool(var_1.value == var_10)
    assert var_12 is True
    var_13 = var_1.string
    assert var_13 == '{"list": [1, 2], "nested": {"inner": true}}'
    var_14 = 0
    var_15 = module_1.Position(var_8, var_8, var_14)
    var_16 = var_1.start
    var_17 = bool(var_1.start == var_15)
    assert var_17 is True
    var_18 = 44
    var_19 = 43
    var_20 = module_1.Position(var_8, var_18, var_19)
    var_21 = var_1.end
    var_22 = bool(var_1.end == var_20)
    assert var_22 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"unclosed": '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"invalid": \x80}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'invalid': ''})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n    1,\n    2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n    1,\n    2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 13
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "a": 1\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = var_1.start
    var_6 = bool(var_1.start == var_4)
    assert var_6 is True
    var_7 = 3
    var_8 = 9
    var_9 = module_1.Position(var_7, var_2, var_8)
    var_10 = var_1.end
    var_11 = bool(var_1.end == var_9)
    assert var_11 is True
    var_12 = 'a'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = 2
    var_16 = 6
    var_17 = 7
    var_18 = module_1.Position(var_15, var_16, var_17)
    var_19 = var_14.start
    var_20 = bool(var_14.start == var_18)
    assert var_20 is True
    var_21 = module_1.Position(var_15, var_16, var_17)
    var_22 = var_14.end
    var_23 = bool(var_14.end == var_21)
    assert var_23 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_tokenizing_json_object. Retrieved 19/26 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
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
    var_17 = ''
    var_18 = '{}'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 6/27 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_false. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = len(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_false. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = lambda s, end, strict: (var_0, end + var_1)
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_false. Retrieved 13/22 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 5
    var_3 = lambda s, idx: (ScalarToken(var_0, idx, idx + var_1, s), idx + var_2)
    var_4 = '{"key": "value"}'
    var_5 = {}
    var_6 = 0
    var_7 = (var_4, var_6)
    var_8 = True
    var_9 = module_0._TokenizingJSONObject(var_7, var_8, var_3, var_5, var_4)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_9[var_6]
    var_12 = var_9[0]
    var_13 = bool(var_9[0] == {'key': 'value'})
    assert var_13 is True
    var_14 = len(var_4)
    var_15 = var_9[1]
    var_16 = bool(var_9[1] == var_14)
    assert var_16 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_false. Retrieved 6/34 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = ' \t\n\r'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_whitespace_handling. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = '"a"'
    var_6 = '"b"'
    var_7 = 1
    var_8 = 2
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = [var_2]
    var_4 = (var_0, var_2)
    var_5 = True
    var_6 = len(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_with_null. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_with_integer. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_nested_structure. Retrieved 7/20 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid. Retrieved 4/12 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '42'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '3.14'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: (string[idx:idx + 5], idx + 6)
    var_2 = {}
    var_3 = '"hello"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, args, strict, scan_once, memo, content: ({}, args[1])
    var_2 = {}
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, args, scan_once: ([], args[1])
    var_2 = {}
    var_3 = '[]'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, args, strict, scan_once, memo, content: ({'key': ScalarToken('value', 0, 0, content)}, args[1] + 10)
    var_2 = lambda self, string, idx, strict: (string[idx:idx + 5], idx + 6)
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0
    var_6 = '[1]'

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key":'
    var_2 = len(var_1)
    var_3 = ' \t\n\r'
    var_4 = var_0[var_2]
    var_5 = var_4 in var_3
    assert var_5 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_list. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_list_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_dict. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_dict_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_with_unicode. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_with_unicode. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'   '
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
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"inner": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'inner'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = var_1.value
    var_12 = bool(var_1.value == var_10)
    assert var_12 is True
    var_13 = var_1.string
    assert var_13 == '{"list": [1, 2], "nested": {"inner": true}}'
    var_14 = 0
    var_15 = module_1.Position(var_8, var_8, var_14)
    var_16 = var_1.start
    var_17 = bool(var_1.start == var_15)
    assert var_17 is True
    var_18 = 46
    var_19 = 45
    var_20 = module_1.Position(var_8, var_18, var_19)
    var_21 = var_1.end
    var_22 = bool(var_1.end == var_20)
    assert var_22 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Expecting property name enclosed in double quotes'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "name": "John",\n  "age": 30\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'name': 'John', 'age': 30})
    assert var_3 is True
    var_4 = var_1.string
    var_5 = bool(var_1.string == var_0)
    assert var_5 is True
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 4
    var_12 = 30
    var_13 = module_1.Position(var_11, var_6, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"café"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'café'
    var_3 = var_1.string
    assert var_3 == '"café"'
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
    var_0 = b'"caf\xc3\xa9"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'café'
    var_3 = var_1.string
    assert var_3 == '"café"'
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_json_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_number. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_lookup. Retrieved 10/11 statements.
# Partially parsed test_tokenize_json_lookup_key. Retrieved 11/12 statements.


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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [1, {"b": 2}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, {'b': 2}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [1, {"b": 2}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 21
    var_11 = 20
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'"hello"'
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

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [1, 2]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 2
    var_7 = var_5.string
    assert var_7 == '2'
    var_8 = 12
    var_9 = 11
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_5.start
    var_12 = bool(var_5.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_3, var_8, var_9)
    var_14 = var_5.end
    var_15 = bool(var_5.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == '"key"'
    var_7 = 1
    var_8 = 2
    var_9 = module_1.Position(var_7, var_8, var_7)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 6
    var_13 = 5
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 15/24 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = 'value'
    var_5 = 8
    var_6 = 14
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 15
    var_9 = (var_7, var_8)
    var_10 = (var_0, var_1)
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_10, var_5, var_6, var_0)
    var_13 = {var_11: var_12}



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 2/23 statements.


def test_case_0():
    var_0 = ''
    var_1 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_parse_object_not_equal_to_TokenizingJSONObject. Retrieved 6/8 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = ''
    var_3 = module_0._make_scanner(var_1, var_2)
    var_4 = var_3.__closure__[var_0]
    var_5 = var_4.cell_contents



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 7/18 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/14 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = len(var_0)

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 7/28 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 0
    var_6 = len(var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 7
    var_2 = ' \t\n\r'
    var_3 = var_0[var_1]
    var_4 = var_3 in var_2
    assert var_4 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_equal_to_TokenizingJSONObject. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_tokenizing_json_object_predicate_at_line_61_false. Retrieved 11/42 statements.


import re as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = '\\s*'
    var_3 = module_0.compile(var_2)
    var_4 = ' \t\n\r'
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = True
    var_8 = 0
    var_9 = len(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_parse_object_not_used_when_not_dict. Retrieved 18/24 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
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
    var_17 = '[]'



