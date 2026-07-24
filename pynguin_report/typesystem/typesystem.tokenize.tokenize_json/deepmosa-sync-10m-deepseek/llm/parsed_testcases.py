####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 7/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 7/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 7/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 7/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_number. Retrieved 7/17 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 7/14 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 6/14 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 7/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '"test"'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'null'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'true'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'false'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '42'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('key', idx + 5)
    var_2 = lambda self, args, strict, scan_once, memo, content: ({ScalarToken('key', 1, 5, content): ScalarToken('value', 7, 13, content)}, 14)
    var_3 = None
    var_4 = {}
    var_5 = '{"key": "value"}'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '[1, 2]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = None
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '"test"'
    var_8 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'invalid'
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 16/21 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
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
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 2
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = 9
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 6
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 13
    var_15 = module_0.ScalarToken(var_6, var_14, var_14, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 10
    var_11 = 16
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/23 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/19 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)
    var_8 = {}

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
    var_12 = {var_7: var_11}

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
    var_8 = 7
    var_9 = module_0.ScalarToken(var_4, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 12
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_14 = 2
    var_15 = 16
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_0)
    var_17 = {var_7: var_9, var_13: var_16}

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
    var_10 = 12
    var_11 = 18
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

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 6/21 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 7/22 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 6/21 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

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
    assert var_2 == 2
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
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = len(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_TokenizingJSONObject_raises_error_when_nextchar_is_not_double_quote_after_comma_and_whitespace. Retrieved 6/32 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value",  "key2": "value2"}'
    var_2 = '{"key": "value",  "key2": "value2"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 16/18 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 11/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 11/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 11/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 8/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 8/14 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 19/21 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 24/26 statements.


import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'memo'
    var_6 = 'hello'
    var_7 = lambda s, idx, strict: (var_6, len(s))
    var_8 = True
    var_9 = {}
    var_10 = {var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1._make_scanner(var_14, var_0)
    var_16 = 0
    var_17 = len(var_0)

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'memo'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_1._make_scanner(var_9, var_0)
    var_11 = 0
    var_12 = len(var_0)

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'memo'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_1._make_scanner(var_9, var_0)
    var_11 = 0
    var_12 = len(var_0)

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'memo'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_1._make_scanner(var_9, var_0)
    var_11 = 0
    var_12 = len(var_0)

def test_case_0():
    var_0 = '42'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_int'
    var_4 = 'memo'
    var_5 = {}
    var_6 = 0
    var_7 = len(var_0)

def test_case_0():
    var_0 = '3.14'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_float'
    var_4 = 'memo'
    var_5 = {}
    var_6 = 0
    var_7 = len(var_0)

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = 'value'
    var_5 = 8
    var_6 = 14
    var_7 = lambda string_idx, strict, scan_once, memo, content: ({ScalarToken(var_1, var_2, var_3, content): ScalarToken(var_4, var_5, var_6, content)}, len(content))
    var_8 = 'Context'
    var_9 = ()
    var_10 = 'parse_object'
    var_11 = 'memo'
    var_12 = {}
    var_13 = {var_10: var_7, var_11: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_1._make_scanner(var_17, var_0)
    var_19 = 0
    var_20 = len(var_0)

import typesystem.tokenize.tokens as module_0
import builtins as module_1
import typesystem.tokenize.tokenize_json as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 1
    var_2 = module_0.ScalarToken(var_1, var_1, var_1, var_0)
    var_3 = 2
    var_4 = 4
    var_5 = module_0.ScalarToken(var_3, var_4, var_4, var_0)
    var_6 = 3
    var_7 = 7
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = [var_2, var_5, var_8]
    var_10 = len(var_0)
    var_11 = (var_9, var_10)
    var_12 = lambda string_idx, scan_once: var_11
    var_13 = 'Context'
    var_14 = ()
    var_15 = 'parse_array'
    var_16 = 'memo'
    var_17 = {}
    var_18 = {var_15: var_12, var_16: var_17}
    var_19 = [var_13, var_14, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = module_2._make_scanner(var_22, var_0)
    var_24 = 0
    var_25 = len(var_0)

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '"test"'
    var_1 = {}
    var_2 = 'Context'
    var_3 = ()
    var_4 = 'parse_string'
    var_5 = 'strict'
    var_6 = 'memo'
    var_7 = 'test'
    var_8 = lambda s, idx, strict: (var_7, len(s))
    var_9 = True
    var_10 = {var_4: var_8, var_5: var_9, var_6: var_1}
    var_11 = [var_2, var_3, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1._make_scanner(var_14, var_0)
    var_16 = 0
    var_17 = var_15(var_0, var_16)
    var_18 = bool(var_1 == {})
    assert var_18 is True

import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'memo'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = module_1._make_scanner(var_9, var_0)
    var_11 = 0
    var_12 = var_10(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/24 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/14 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

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
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

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
    var_9 = 6
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 10
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 14
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_scanner_with_null. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_integer. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 4/17 statements.


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
    var_1 = {}
    var_2 = '{}'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '[]'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"hello"'
    var_3 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 13/17 statements.


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
    var_9 = 7
    var_10 = 9
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
    var_9 = 6
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 10
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 15
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_10, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{  "key"  :  42  }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 42
    var_10 = 12
    var_11 = 13
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
    var_9 = 8
    var_10 = 13
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = 'key'
    var_14 = bool('key' in var_1)
    assert var_14 is True
    var_15 = var_1['key']
    assert var_15 == 'key'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_integer. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_empty_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_list_with_elements. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_empty_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_dict_with_elements. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_lookup_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_lookup_dict. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_lookup_key. Retrieved 5/6 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 4

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 4

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 5

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 4

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 7

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, true, null]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, True, None])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, true, null]'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 16

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 16

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"inner": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'nested': {'inner': True}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"list": [1, 2], "nested": {"inner": true}}'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"bytes"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'bytes'
    var_3 = var_1.string
    assert var_3 == '"bytes"'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 7

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 4
    var_8 = var_1.end.column_no
    assert var_8 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[10, 20, 30]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 20
    var_6 = var_4.string
    assert var_6 == '20'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 2
    var_6 = var_4.string
    assert var_6 == '2'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"x": 100}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'x'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'x'
    var_6 = var_4.string
    assert var_6 == '"x"'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_json_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_list_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_dict_with_items. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline_content. Retrieved 8/9 statements.


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
    var_0 = '   \n\t  '
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
    var_0 = '{"key": "value", "num": 42}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value', 'num': 42})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value", "num": 42}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 28
    var_11 = 27
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
    var_18 = 48
    var_19 = 47
    var_20 = module_1.Position(var_8, var_18, var_19)
    var_21 = var_1.end
    var_22 = bool(var_1.end == var_20)
    assert var_22 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": "bytes"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 'bytes'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": "bytes"}'
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

def test_case_0():
    var_0 = '{"unclosed":'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "key": "value"\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
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
    var_11 = 3
    var_12 = 20
    var_13 = module_1.Position(var_11, var_6, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #12
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
    var_0 = '{"a": [1, {"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, {'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [1, {"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 24
    var_11 = 23
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
    var_0 = '{invalid}'
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
    var_12 = 33
    var_13 = module_1.Position(var_11, var_6, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/15 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/15 statements.
# Partially parsed test_make_scanner_list. Retrieved 5/16 statements.
# Partially parsed test_make_scanner_dict. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 5/13 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.


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
    var_3 = '[1, 2]'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '{"key": "value"}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = ''
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '"test"'
    var_6 = 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/24 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/14 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_scan_once_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

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
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

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
    var_9 = 6
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 10
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 14
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'nullx'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_TokenizingJSONObject. Retrieved 3/10 statements.


def test_case_0():
    var_0 = True
    var_1 = '{"key": "value"}'
    var_2 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '{}'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 7/21 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = len(var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 12/26 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = ' \t\n\r'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_scalar_token_null. Retrieved 6/15 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, *args: ([], 0)
    var_3 = lambda self, string, idx, strict: ('', idx)
    var_4 = 'null'
    var_5 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_nested_scan_once. Retrieved 7/14 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)
    var_8 = {}

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
    var_0 = '{"key": [1, 2]}'
    var_1 = {}
    var_2 = 0
    assert var_2 == 1
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = len(var_0)

def test_case_0():
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_0 = '[1, true, null]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, True, None])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, true, null]'
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
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'nested': {'inner': True}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 45
    var_10 = 44
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

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
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 9
    var_11 = module_1.Position(var_9, var_4, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 19/28 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 15/19 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 7/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_on_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
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
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 6
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 10
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 14
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_10, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
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
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key":'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 16/25 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon_raises_error. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma_raises_error. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote_on_key_raises_error. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_raises_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 12
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = module_0.ScalarToken(var_6, var_5, var_5, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = module_0.ScalarToken(var_8, var_9, var_9, var_0)
    var_11 = 6
    var_12 = module_0.ScalarToken(var_5, var_11, var_11, var_0)
    var_13 = 2
    var_14 = 13
    var_15 = module_0.ScalarToken(var_13, var_14, var_14, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 4
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_dict. Retrieved 6/18 statements.
# Partially parsed test_make_scanner_list. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 5/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '{"key": "value"}'
    var_4 = 0
    var_5 = len(var_3)

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '["item1", "item2"]'
    var_4 = 0
    var_5 = len(var_3)

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '"test"'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'invalid'
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is True
    var_7 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_8 = 'NULL'
    var_9 = module_0.ScalarToken(var_0, var_1, var_2, var_8)
    var_10 = var_7 == var_9
    assert var_10 is False



# Parsed testcases at query #6
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
    var_0 = '   \n\t  '
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
    var_0 = '{"a": [1, {"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, {'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [1, {"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 24
    var_11 = 23
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"x": 5}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'x': 5})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"x": 5}'
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

def test_case_0():
    var_0 = '{"unclosed":'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Expecting'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"invalid": \x80}'
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 1/12 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_TokenizingJSONObject_raises_error_when_nextchar_not_double_quote_at_line_72. Retrieved 8/21 statements.


def test_case_0():
    var_0 = '{"key": "value", "another": "value"}'
    var_1 = '{"key": "value", "another": "value"}'
    var_2 = '{"key": "value", '
    var_3 = len(var_2)
    var_4 = {}
    var_5 = ' \t\n\r'
    var_6 = (var_1, var_3)
    var_7 = True
    var_8 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 8/14 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 16/28 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 8/13 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 7/15 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = 0
    var_6 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = 0
    var_5 = (var_0, var_1)
    var_6 = 'a'
    var_7 = 1
    var_8 = 3
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_0)
    var_10 = 'b'
    var_11 = 9
    var_12 = 11
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_14 = len(var_0)

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = 0
    var_6 = len(var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = 0
    var_5 = (var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_nextchar_not_double_quote_raises_error. Retrieved 10/31 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value", "another": 123}'
    var_2 = '{"key": "value", "another": 123}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = '{"key": "value", another: 123}'
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_scan_once_does_not_raise_stop_iteration. Retrieved 11/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 7
    var_4 = 13
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = 14
    var_7 = (var_5, var_6)
    var_8 = 1
    var_9 = (var_0, var_8)
    var_10 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_scalar_token_null_value. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'null'
    var_9 = 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 16/23 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/13 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 2
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 7
    var_9 = module_0.ScalarToken(var_4, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 11
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_14 = 14
    var_15 = module_0.ScalarToken(var_6, var_14, var_14, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 13
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

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

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 4/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_false. Retrieved 13/24 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'nullx'
    var_9 = 0
    var_10 = 'xnull'
    var_11 = 'xnull'
    var_12 = 0
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.value
    var_6 = None
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_53_evaluates_to_false. Retrieved 7/14 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = (var_0, var_1)
    var_5 = 'key'
    var_6 = len(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_with_null. Retrieved 10/18 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 10/18 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 10/18 statements.
# Partially parsed test_make_scanner_with_integer. Retrieved 10/18 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 10/18 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 14/22 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 15/23 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 15/23 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 12/20 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'null'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'true'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'false'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = '42'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = '3.14'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = 'parse_string'
    var_7 = True
    var_8 = {}
    var_9 = 'hello'
    var_10 = 7
    var_11 = lambda s, idx, strict: (var_9, idx + var_10)
    var_12 = '"hello"'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = 'parse_object'
    var_7 = True
    var_8 = {}
    var_9 = {}
    var_10 = 2
    var_11 = (var_9, var_10)
    var_12 = lambda *args: var_11
    var_13 = '{}'
    var_14 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = 'parse_array'
    var_7 = True
    var_8 = {}
    var_9 = []
    var_10 = 2
    var_11 = (var_9, var_10)
    var_12 = lambda *args: var_11
    var_13 = '[]'
    var_14 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'null'
    var_11 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'invalid'
    var_9 = 0
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_false. Retrieved 6/26 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = ''
    var_6 = 'key'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 14/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 21/28 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 15/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
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
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 6
    var_10 = 7
    var_11 = module_0.ScalarToken(var_5, var_9, var_10, var_0)
    var_12 = 'b'
    var_13 = 10
    var_14 = 11
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = 14
    var_17 = 15
    var_18 = module_0.ScalarToken(var_7, var_16, var_17, var_0)
    var_19 = {var_8: var_11, var_15: var_18}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 11
    var_11 = 13
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 19/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 1
    var_2 = {}
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 14
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 15
    var_9 = (var_7, var_8)
    var_10 = ' \t\n\r'
    var_11 = 1
    var_12 = lambda s, idx: Mock(end=lambda : idx + var_11) if s[idx] in var_10 else Mock(end=lambda : idx)
    var_13 = ' \t\n\r'
    var_14 = (var_0, var_1)
    var_15 = 'key'
    var_16 = module_0.ScalarToken(var_14, var_5, var_6, var_0)
    var_17 = {var_15: var_16}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 6/22 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 14/30 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 6/22 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 7/24 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

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
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = len(var_0)

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
    var_0 = '{"key": "value", "key": "other"}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'key'
    var_7 = bool('key' in var_1)
    assert var_7 is True
    var_8 = len(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_TokenizingJSONObject. Retrieved 1/7 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.value
    var_6 = None
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_TokenizingJSONObject. Retrieved 1/11 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 13/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'null'
    var_9 = 0
    var_10 = None
    var_11 = 3
    var_12 = module_0.ScalarToken(var_10, var_9, var_11, var_8)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 6/17 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 5/20 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 5/21 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 7/15 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'true'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'false'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'null'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = '42'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = '3.14'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = '{"key":"value"}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = '["item1","item2"]'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = None
    var_6 = '"test"'
    var_7 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'invalid'
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 6/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 6/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)
    var_8 = {}

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
    var_5 = len(var_0)

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
    var_6 = 'key'
    var_7 = bool('key' in var_1)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is True



