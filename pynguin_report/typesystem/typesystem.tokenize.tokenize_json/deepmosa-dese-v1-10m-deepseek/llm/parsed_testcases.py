####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_string. Retrieved 3/11 statements.
# Partially parsed test_make_scanner_null. Retrieved 3/9 statements.
# Partially parsed test_make_scanner_true. Retrieved 3/9 statements.
# Partially parsed test_make_scanner_false. Retrieved 3/9 statements.
# Partially parsed test_make_scanner_number. Retrieved 3/13 statements.


def test_case_0():
    var_0 = ''
    var_1 = '"test"'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 'true'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 'false'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = '123'
    var_2 = 0



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
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
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 5/6 statements.


import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = ''
    var_2 = module_1._make_scanner(var_0, var_1)
    var_3 = ''
    var_4 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 8/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = None
    var_2 = 0
    var_3 = len(var_0)
    var_4 = 1
    var_5 = var_3 - var_4
    var_6 = module_0.ScalarToken(var_1, var_2, var_5, var_0)
    var_7 = len(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_TokenizingJSONObject_single_key_value_pair. Retrieved 27/43 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_value_pairs. Retrieved 46/78 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (var_4, e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, var_0), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = 'key'
    var_11 = var_9[var_1][var_10]
    var_12 = var_9[var_1][var_10]
    var_13 = var_9[var_1][var_10]
    var_14 = var_9[var_1][var_10]
    var_15 = var_9[var_1][var_10]
    var_16 = var_9[var_1][var_10]
    var_17 = var_9[var_1][var_10]
    var_18 = var_9[var_1][var_10]
    var_19 = var_9[var_1][var_4]
    var_20 = var_9[var_1][var_4]
    var_21 = var_9[var_1][var_4]
    var_22 = var_9[var_1][var_4]
    var_23 = var_9[var_1][var_4]
    var_24 = var_9[var_1][var_4]
    var_25 = var_9[var_1][var_4]
    var_26 = var_9[var_1][var_4]

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, var_0), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = 'key1'
    var_13 = var_11[var_1][var_12]
    var_14 = var_11[var_1][var_12]
    var_15 = var_11[var_1][var_12]
    var_16 = var_11[var_1][var_12]
    var_17 = var_11[var_1][var_12]
    var_18 = var_11[var_1][var_12]
    var_19 = var_11[var_1][var_12]
    var_20 = var_11[var_1][var_12]
    var_21 = var_11[var_1][var_5]
    var_22 = var_11[var_1][var_5]
    var_23 = var_11[var_1][var_5]
    var_24 = var_11[var_1][var_5]
    var_25 = var_11[var_1][var_5]
    var_26 = var_11[var_1][var_5]
    var_27 = var_11[var_1][var_5]
    var_28 = var_11[var_1][var_5]
    var_29 = 'key2'
    var_30 = var_11[var_1][var_29]
    var_31 = var_11[var_1][var_29]
    var_32 = var_11[var_1][var_29]
    var_33 = var_11[var_1][var_29]
    var_34 = var_11[var_1][var_29]
    var_35 = var_11[var_1][var_29]
    var_36 = var_11[var_1][var_29]
    var_37 = var_11[var_1][var_29]
    var_38 = var_11[var_1][var_6]
    var_39 = var_11[var_1][var_6]
    var_40 = var_11[var_1][var_6]
    var_41 = var_11[var_1][var_6]
    var_42 = var_11[var_1][var_6]
    var_43 = var_11[var_1][var_6]
    var_44 = var_11[var_1][var_6]
    var_45 = var_11[var_1][var_6]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 8/12 statements.


def test_case_0():
    var_0 = lambda *args: None
    var_1 = lambda *args: None
    var_2 = False
    var_3 = lambda *args: None
    var_4 = lambda *args: None
    var_5 = {}
    var_6 = 'null'
    var_7 = 0



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, end: (ScalarToken(var_5, end, end, var_1), end + var_4)
    var_7 = {}
    var_8 = module_0._TokenizingJSONObject(var_3, var_4, var_6, var_7, var_1)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)
    var_11 = 'key'
    var_12 = 7
    var_13 = 12
    var_14 = module_1.ScalarToken(var_5, var_12, var_13, var_1)
    var_15 = {var_11: var_14}
    var_16 = 14
    var_17 = (var_15, var_16)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, end: (ScalarToken(var_5 + str(end), end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)
    var_11 = 'key1'
    var_12 = 'key2'
    var_13 = 'value1'
    var_14 = 8
    var_15 = 14
    var_16 = module_1.ScalarToken(var_13, var_14, var_15, var_1)
    var_17 = 'value2'
    var_18 = 24
    var_19 = 30
    var_20 = module_1.ScalarToken(var_17, var_18, var_19, var_1)
    var_21 = {var_11: var_16, var_12: var_20}
    var_22 = 32
    var_23 = (var_21, var_22)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)
    var_11 = 'key'
    var_12 = 10
    var_13 = 16
    var_14 = module_1.ScalarToken(var_5, var_12, var_13, var_1)
    var_15 = {var_11: var_14}
    var_16 = 18
    var_17 = (var_15, var_16)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value", "key2": "value2"}'
    var_1 = var_0
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, end: (ScalarToken(var_5 + str(end), end, end + var_6, var_1), end + var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_9, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 20/25 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_array'
    var_4 = 'parse_string'
    var_5 = 'strict'
    var_6 = 'parse_float'
    var_7 = 'parse_int'
    var_8 = 'memo'
    var_9 = []
    var_10 = len(var_0)
    var_11 = (var_9, var_10)
    var_12 = lambda *args: var_11
    var_13 = 'value'
    var_14 = len(var_0)
    var_15 = (var_13, var_14)
    var_16 = lambda *args: var_15
    var_17 = True
    var_18 = {}
    var_19 = 0



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = None
    var_2 = 0
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__TokenizingJSONObject_trivial_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_object_with_whitespace. Retrieved 17/18 statements.
# Partially parsed test__TokenizingJSONObject_object_with_multiple_pairs. Retrieved 24/25 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, idx: (ScalarToken(var_4, idx, idx, var_0), idx + var_3)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, idx: (ScalarToken(var_4, idx, idx + var_5, var_0), idx + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 3
    var_11 = module_0.ScalarToken(var_9, var_3, var_10, var_0)
    var_12 = 7
    var_13 = 11
    var_14 = module_0.ScalarToken(var_4, var_12, var_13, var_0)
    var_15 = {var_11: var_14}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, idx: (ScalarToken(var_4, idx, idx + var_5, var_0), idx + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 2
    var_11 = 4
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 9
    var_14 = 13
    var_15 = module_0.ScalarToken(var_4, var_13, var_14, var_0)
    var_16 = {var_12: var_15}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, idx: (ScalarToken(var_5 if var_4 in s else var_6, idx, idx + var_7, var_0), idx + var_8)
    var_10 = {}
    var_11 = 4
    var_12 = module_0.ScalarToken(var_4, var_3, var_11, var_0)
    var_13 = 'key2'
    var_14 = 16
    var_15 = 19
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = 8
    var_18 = 13
    var_19 = module_0.ScalarToken(var_5, var_17, var_18, var_0)
    var_20 = 23
    var_21 = 28
    var_22 = module_0.ScalarToken(var_6, var_20, var_21, var_0)
    var_23 = {var_12: var_19, var_16: var_22}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, idx: (ScalarToken(var_4, idx, idx + var_5, var_0), idx + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, idx: (ScalarToken(var_5 if var_4 in s else var_6, idx, idx + var_7, var_0), idx + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, idx: (ScalarToken(var_4, idx, idx + var_5, var_0), idx + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_TokenizingJSONObject_handles_empty_string_at_position. Retrieved 7/8 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, end: (var_4, end)
    var_6 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_TokenizingJSONObject_raises_error_when_nextchar_is_not_quote. Retrieved 7/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value", 123}'
    var_2 = var_1
    var_3 = 13
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = str(var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 12/24 statements.


def test_case_0():
    var_0 = lambda *args: None
    var_1 = lambda *args: None
    var_2 = True
    var_3 = lambda *args: None
    var_4 = lambda *args: None
    var_5 = {}
    var_6 = '{"a": 1}'
    var_7 = {}
    var_8 = 0
    var_9 = 2
    var_10 = 3
    var_11 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__make_scanner_with_string. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_dict. Retrieved 6/16 statements.
# Partially parsed test__make_scanner_with_list. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_null. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_boolean. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 6/13 statements.


def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('test', 6)
    var_2 = True
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('key', 5)
    var_2 = True
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: (['item1', 'item2'], 12)
    var_1 = lambda self, *args: ('item1', 7)
    var_2 = True
    var_3 = {}
    var_4 = '["item1", "item2"]'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = '123'
    var_5 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_json_does_not_raise_json_decode_error. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0._TokenizingDecoder()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenizing_json_object_handles_index_error_in_whitespace_skipping. Retrieved 14/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = var_0
    var_2 = {}
    var_3 = 1
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 11
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = 13
    var_13 = (var_11, var_12)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_predicate_at_line_4_evaluates_to_false. Retrieved 7/15 statements.


def test_case_0():
    var_0 = lambda x, y: ([], 0)
    var_1 = lambda x, y, z: ('', 0)
    var_2 = False
    var_3 = {}
    var_4 = ''
    var_5 = ''
    var_6 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenizing_json_object_handles_index_error_at_line_48. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = {}
    var_5 = var_0
    var_6 = 'value'
    var_7 = 8
    var_8 = 14
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_5)
    var_10 = lambda s, end: (var_9, end + var_2)
    var_11 = 0
    var_12 = (var_0, var_11)
    var_13 = True
    var_14 = module_1._TokenizingJSONObject(var_12, var_13, var_10, var_4, var_5)
    var_15 = len(var_14)
    assert var_15 == 2



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = {}
    var_3 = ''
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = None
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = lambda x, y: var_8
    var_10 = module_0._TokenizingJSONObject(var_4, var_5, var_9, var_2, var_3)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_json_does_not_raise_json_decode_error. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0._TokenizingDecoder()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenizing_json_object_empty_string_handles_index_error. Retrieved 7/15 statements.


def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = ' \t\n\r'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenizing_json_object_handles_index_error_at_line_61. Retrieved 10/18 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = -1
    var_2 = var_0[:var_1]
    var_3 = {}
    var_4 = 1
    var_5 = (var_2, var_4)
    var_6 = True
    var_7 = 0
    var_8 = len(var_2)
    var_9 = var_8 + var_6



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_TokenizingJSONObject_predicate_at_line_61_evaluates_to_False. Retrieved 20/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = ''
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = (var_4, var_5)
    var_7 = 1
    var_8 = lambda : var_7
    var_9 = 'key'
    var_10 = (var_9, var_7)
    var_11 = {}
    var_12 = '{"key": "value"}'
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = True
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_9: var_16}
    var_18 = 6
    var_19 = (var_17, var_18)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 20/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, i: (var_5, i)
    var_7 = module_0._TokenizingJSONObject(var_3, var_4, var_6, var_1, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key1'
    var_6 = 6
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value1'
    var_9 = 10
    var_10 = 17
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 'key2'
    var_13 = 20
    var_14 = 25
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = 'value2'
    var_17 = 27
    var_18 = 34
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 7
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_json_with_valid_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_tokenizing_json_object_no_index_error. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 21/26 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_array'
    var_4 = 'parse_string'
    var_5 = 'strict'
    var_6 = 'parse_float'
    var_7 = 'parse_int'
    var_8 = 'memo'
    var_9 = None
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = lambda *args: var_11
    var_13 = 'value'
    var_14 = len(var_0)
    var_15 = 2
    var_16 = var_14 - var_15
    var_17 = (var_13, var_16)
    var_18 = lambda *args: var_17
    var_19 = True
    var_20 = {}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 8/14 statements.


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = {}
    var_5 = ''
    var_6 = ''
    var_7 = 0



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_line_4_predicate_evaluates_to_false. Retrieved 7/10 statements.


def test_case_0():
    var_0 = lambda *args: None
    var_1 = lambda *args: None
    var_2 = False
    var_3 = lambda *args: None
    var_4 = lambda *args: None
    var_5 = {}
    var_6 = ''



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_tokenizing_json_object_without_index_error. Retrieved 21/22 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 5
    var_9 = lambda s, end: (var_7, end + var_8)
    var_10 = None
    var_11 = lambda : var_10
    var_12 = lambda s, end: var_11
    var_13 = 9
    var_14 = '{"key": "value"}'
    var_15 = 1
    var_16 = (var_14, var_15)
    var_17 = True
    var_18 = {}
    var_19 = ' '
    var_20 = module_1._TokenizingJSONObject(var_16, var_17, var_9, var_18, var_0, var_12, var_19)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_tokenizing_json_object_index_error_handling. Retrieved 31/36 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {}
    var_9 = 'key: value'
    var_10 = (var_3, var_7)
    var_11 = [var_10]
    var_12 = var_11.append
    var_13 = var_8.setdefault
    var_14 = 9
    var_15 = 'key: value'
    var_16 = lambda s, end: end
    var_17 = ' \t\n\r'
    var_18 = lambda s, end: (var_7, end)
    var_19 = lambda s, end, strict: (var_0, end)
    var_20 = 'WHITESPACE'
    var_21 = ()
    var_22 = 'match'
    var_23 = ()
    var_24 = 'end'
    var_25 = lambda : var_14
    var_26 = {var_24: var_25}
    var_27 = ' \t\n\r'
    var_28 = (var_15, var_1)
    var_29 = False
    var_30 = module_1._TokenizingJSONObject(var_28, var_29, var_18, var_8, var_9, var_16, var_17)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_tokenizing_json_object_empty_string. Retrieved 9/10 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_0)
    var_6 = (var_5, var_1)
    var_7 = lambda s, e: var_6
    var_8 = {}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__make_scanner_with_string. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_dict. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_list. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_null. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_true. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_false. Retrieved 3/17 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 3/17 statements.


def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = '"test"'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({'key': 'value'}, args[0][1] + 2)
    var_1 = '{"key": "value"}'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = '["item"]'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = 'true'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = 'false'
    var_2 = 0

def test_case_0():
    var_0 = lambda *args: ({}, args[0][1] + 2)
    var_1 = '123'
    var_2 = 0



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda *args: var_0
    var_2 = 'content'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 15/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_value. Retrieved 21/22 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{}'
    var_6 = None
    var_7 = 1
    var_8 = lambda s, e: (ScalarToken(var_6, e, e, var_5), e + var_7)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{"key": "value"}'
    var_6 = 'value'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, var_5), e + var_8)
    var_10 = 'key'
    var_11 = 9
    var_12 = 14
    var_13 = module_0.ScalarToken(var_6, var_11, var_12, var_5)
    var_14 = {var_10: var_13}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{"key1": "value1", "key2": "value2"}'
    var_6 = 'value'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_6 + s[e], e, e + var_7, var_5), e + var_8)
    var_10 = 'key1'
    var_11 = 'key2'
    var_12 = 'value1'
    var_13 = 9
    var_14 = 15
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_5)
    var_16 = 'value2'
    var_17 = 25
    var_18 = 31
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_5)
    var_20 = {var_10: var_15, var_11: var_19}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value", "invalid"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{"key": "value", "invalid"}'
    var_6 = 'value'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, var_5), e + var_8)
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_4, var_5)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value", "key2":}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{"key": "value", "key2":}'
    var_6 = 'value'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, var_5), e + var_8)
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_4, var_5)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 21/26 statements.
# Partially parsed test_TokenizingJSONObject_invalid_json_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_invalid_json_missing_comma. Retrieved 5/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = {}
    var_6 = ''
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_6)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_3, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 12
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'key1'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_3, var_6, var_0)
    var_8 = 'key2'
    var_9 = 17
    var_10 = 20
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 'value1'
    var_13 = 9
    var_14 = 14
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = 'value2'
    var_17 = 25
    var_18 = 30
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_0)
    var_20 = {var_7: var_15, var_11: var_19}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_tokenizing_json_object_end_of_string. Retrieved 8/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (var_4, e)
    var_6 = {}
    var_7 = ''



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 6/15 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: float(x)
    var_2 = lambda x: int(x)
    var_3 = {}
    var_4 = '{}'
    var_5 = 0



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_make_scanner_with_string. Retrieved 17/21 statements.
# Partially parsed test_make_scanner_with_object. Retrieved 20/24 statements.
# Partially parsed test_make_scanner_with_array. Retrieved 18/22 statements.
# Partially parsed test_make_scanner_with_null. Retrieved 14/18 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 14/18 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 14/18 statements.
# Partially parsed test_make_scanner_with_number. Retrieved 16/20 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = 'test'
    var_10 = 6
    var_11 = lambda s, i, strict: (var_9, i + var_10)
    var_12 = None
    var_13 = {}
    var_14 = {var_2: var_8, var_3: var_11, var_4: var_12, var_5: var_12, var_6: var_12, var_7: var_13}
    var_15 = '"test"'
    var_16 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = 'parse_object'
    var_9 = True
    var_10 = None
    var_11 = {}
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 13
    var_16 = lambda s, i, strict, scan_once, memo, content: (var_14, i + var_15)
    var_17 = {var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_10, var_6: var_10, var_7: var_11, var_8: var_16}
    var_18 = '{"key":"value"}'
    var_19 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = None
    var_10 = 'item'
    var_11 = [var_10]
    var_12 = 7
    var_13 = lambda s, i, scan_once: (var_11, i + var_12)
    var_14 = {}
    var_15 = {var_2: var_8, var_3: var_9, var_4: var_13, var_5: var_9, var_6: var_9, var_7: var_14}
    var_16 = '["item"]'
    var_17 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = None
    var_10 = {}
    var_11 = {var_2: var_8, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'null'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = None
    var_10 = {}
    var_11 = {var_2: var_8, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'true'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = None
    var_10 = {}
    var_11 = {var_2: var_8, var_3: var_9, var_4: var_9, var_5: var_9, var_6: var_9, var_7: var_10}
    var_12 = 'false'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_string'
    var_4 = 'parse_array'
    var_5 = 'parse_int'
    var_6 = 'parse_float'
    var_7 = 'memo'
    var_8 = True
    var_9 = None
    var_10 = 42
    var_11 = lambda s: var_10
    var_12 = {}
    var_13 = {var_2: var_8, var_3: var_9, var_4: var_9, var_5: var_11, var_6: var_9, var_7: var_12}
    var_14 = '42'
    var_15 = 0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 6/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 6/16 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 6/13 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = '{}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, i: (var_5, i)
    var_7 = {}
    var_8 = module_0._TokenizingJSONObject(var_3, var_4, var_6, var_7, var_0)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = len(var_0)

def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = len(var_0)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_tokenizing_json_object_empty_object. Retrieved 6/11 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = '{}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = {}



# Parsed testcases at query #44
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'value'
    var_7 = 7
    var_8 = 12
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_0)
    var_10 = 13
    var_11 = (var_9, var_10)
    var_12 = lambda x, y: var_11
    var_13 = {}
    var_14 = module_1._TokenizingJSONObject(var_4, var_5, var_12, var_13, var_0)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_tokenize_json_scanner_handles_opening_brace. Retrieved 3/15 statements.


def test_case_0():
    var_0 = ''
    var_1 = '{'
    var_2 = 0



# Parsed testcases at query #46
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, end: (var_4, end)
    var_6 = {}
    var_7 = ''
    var_8 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_7)



# Parsed testcases at query #47
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #48
#--------------------------






####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__make_scanner_scalar_string. Retrieved 8/14 statements.
# Partially parsed test__make_scanner_scalar_true. Retrieved 8/12 statements.
# Partially parsed test__make_scanner_scalar_false. Retrieved 8/12 statements.
# Partially parsed test__make_scanner_scalar_null. Retrieved 8/12 statements.
# Partially parsed test__make_scanner_scalar_number. Retrieved 8/12 statements.
# Partially parsed test__make_scanner_list. Retrieved 7/13 statements.
# Partially parsed test__make_scanner_dict. Retrieved 8/15 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'test'
    var_6 = '"test"'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'false'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = lambda x: float(x)
    var_4 = lambda x: int(x)
    var_5 = {}
    var_6 = '123'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '[1,2,3]'
    var_6 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = '{"key":"value"}'
    var_7 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_scalar_token_null_condition_false. Retrieved 8/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = 'true'
    var_7 = module_0.ScalarToken(var_5, var_1, var_2, var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 15/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_value. Retrieved 21/22 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = False
    var_3 = None
    var_4 = 1
    var_5 = lambda s, end: (ScalarToken(var_3, end, end), end + var_4)
    var_6 = {}
    var_7 = var_0
    var_8 = (var_0, var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = False
    var_3 = 'value'
    var_4 = 4
    var_5 = 5
    var_6 = lambda s, end: (ScalarToken(var_3, end, end + var_4), end + var_5)
    var_7 = {}
    var_8 = var_0
    var_9 = (var_0, var_1)
    var_10 = 'key'
    var_11 = 7
    var_12 = 12
    var_13 = module_0.ScalarToken(var_3, var_11, var_12)
    var_14 = {var_10: var_13}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = False
    var_3 = 6
    var_4 = '"value1"'
    var_5 = 'value1'
    var_6 = 5
    var_7 = 'value2'
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6), end + var_3) if s[end:end + var_3] == var_4 else (ScalarToken(var_7, end, end + var_6), end + var_3)
    var_9 = {}
    var_10 = var_0
    var_11 = (var_0, var_1)
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 8
    var_15 = 14
    var_16 = module_0.ScalarToken(var_5, var_14, var_15)
    var_17 = 24
    var_18 = 30
    var_19 = module_0.ScalarToken(var_7, var_17, var_18)
    var_20 = {var_12: var_16, var_13: var_19}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = False
    var_3 = 'value'
    var_4 = 4
    var_5 = 5
    var_6 = lambda s, end: (ScalarToken(var_3, end, end + var_4), end + var_5)
    var_7 = {}
    var_8 = var_0
    var_9 = (var_0, var_1)
    var_10 = module_0._TokenizingJSONObject(var_9, var_2, var_6, var_7, var_8)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = False
    var_3 = 'value'
    var_4 = 4
    var_5 = 5
    var_6 = lambda s, end: (ScalarToken(var_3, end, end + var_4), end + var_5)
    var_7 = {}
    var_8 = var_0
    var_9 = (var_0, var_1)
    var_10 = module_0._TokenizingJSONObject(var_9, var_2, var_6, var_7, var_8)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = False
    var_3 = 6
    var_4 = '"value1"'
    var_5 = 'value1'
    var_6 = 5
    var_7 = 'value2'
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6), end + var_3) if s[end:end + var_3] == var_4 else (ScalarToken(var_7, end, end + var_6), end + var_3)
    var_9 = {}
    var_10 = var_0
    var_11 = (var_0, var_1)
    var_12 = module_0._TokenizingJSONObject(var_11, var_2, var_8, var_9, var_10)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = False
    var_3 = 'value'
    var_4 = 4
    var_5 = 5
    var_6 = lambda s, end: (ScalarToken(var_3, end, end + var_4), end + var_5)
    var_7 = {}
    var_8 = var_0
    var_9 = (var_0, var_1)
    var_10 = module_0._TokenizingJSONObject(var_9, var_2, var_6, var_7, var_8)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 16/21 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, i: (ScalarToken(var_6, i, i, s), i + var_5)
    var_8 = module_0._TokenizingJSONObject(var_4, var_5, var_7, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = 'value'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, i: (ScalarToken(var_6, i, i + var_7, s), i + var_8)
    var_10 = module_0._TokenizingJSONObject(var_4, var_5, var_9, var_1, var_0)
    var_11 = 'key'
    var_12 = 7
    var_13 = 12
    var_14 = module_1.ScalarToken(var_6, var_12, var_13, var_0)
    var_15 = {var_11: var_14}
    var_16 = 14
    var_17 = (var_15, var_16)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": {"nested": "value"}}'
    var_1 = {}
    var_2 = '{"key": {"nested": "value"}}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = 'key'
    var_7 = 'nested'
    var_8 = 'value'
    var_9 = 16
    var_10 = 21
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = {var_6: var_12}
    var_14 = 24
    var_15 = (var_13, var_14)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_handles_string. Retrieved 13/17 statements.
# Partially parsed test_make_scanner_handles_object. Retrieved 16/20 statements.
# Partially parsed test_make_scanner_handles_array. Retrieved 15/19 statements.
# Partially parsed test_make_scanner_handles_null. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_handles_true. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_handles_false. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_handles_number. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_string'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'test'
    var_6 = 6
    var_7 = lambda s, i, strict: (var_5, i + var_6)
    var_8 = True
    var_9 = {}
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = '"test"'
    var_12 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_object'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 13
    var_9 = (var_7, var_8)
    var_10 = lambda s, strict, scan_once, memo, content: var_9
    var_11 = True
    var_12 = {}
    var_13 = {var_2: var_10, var_3: var_11, var_4: var_12}
    var_14 = '{"key":"value"}'
    var_15 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'item'
    var_6 = [var_5]
    var_7 = 6
    var_8 = (var_6, var_7)
    var_9 = lambda s, scan_once: var_8
    var_10 = True
    var_11 = {}
    var_12 = {var_2: var_9, var_3: var_10, var_4: var_11}
    var_13 = '["item"]'
    var_14 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'null'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'true'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'false'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_float'
    var_3 = 'parse_int'
    var_4 = 'strict'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = '123'
    var_9 = 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_TokenizingJSONObject_single_key_value_pair. Retrieved 13/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_value_pairs. Retrieved 18/25 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = {}
    var_5 = 'key'
    var_6 = 'value'
    var_7 = 7
    var_8 = 13
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_0)
    var_10 = {var_5: var_9}
    var_11 = 15
    var_12 = (var_10, var_11)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = {}
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 'value1'
    var_8 = 9
    var_9 = 15
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = 'value2'
    var_12 = 26
    var_13 = 32
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = {var_5: var_10, var_6: var_14}
    var_16 = 34
    var_17 = (var_15, var_16)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, var_0), i + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = 'key'
    var_11 = 7
    var_12 = 13
    var_13 = module_1.ScalarToken(var_4, var_11, var_12, var_0)
    var_14 = {var_10: var_13}
    var_15 = 15
    var_16 = (var_14, var_15)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 6
    var_5 = 7
    var_6 = lambda s, i: (ScalarToken(f'value{i}', i, i + var_4, var_0), i + var_5)
    var_7 = {}
    var_8 = module_0._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = 'value1'
    var_12 = 8
    var_13 = 14
    var_14 = module_1.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 'value2'
    var_16 = 24
    var_17 = 30
    var_18 = module_1.ScalarToken(var_15, var_16, var_17, var_0)
    var_19 = {var_9: var_14, var_10: var_18}
    var_20 = 32
    var_21 = (var_19, var_20)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 6
    var_5 = 7
    var_6 = lambda s, i: (ScalarToken(f'value{i}', i, i + var_4, var_0), i + var_5)
    var_7 = {}
    var_8 = module_0._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__make_scanner_with_string. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_dict. Retrieved 6/16 statements.
# Partially parsed test__make_scanner_with_list. Retrieved 5/15 statements.
# Partially parsed test__make_scanner_with_null. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_true. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_false. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 6/13 statements.


def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('test', 6)
    var_2 = True
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('key', 5)
    var_2 = True
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('item', 6)
    var_1 = True
    var_2 = {}
    var_3 = '["item"]'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = '123'
    var_5 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenizing_json_object_ends_with_closing_brace. Retrieved 14/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = len(var_1)
    var_6 = 'key'
    var_7 = 4
    var_8 = module_0.ScalarToken(var_6, var_4, var_7, var_1)
    var_9 = 'value'
    var_10 = 7
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_1)
    var_13 = {var_8: var_12}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 6/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 13/23 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 6/13 statements.
# Partially parsed test__TokenizingJSONObject_invalid_missing_colon. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_invalid_missing_comma. Retrieved 5/11 statements.
# Partially parsed test__TokenizingJSONObject_invalid_key_not_string. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, i: (var_6, i)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key1'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'key2'
    var_9 = 18
    var_10 = 22
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = len(var_0)

def test_case_0():
    var_0 = ' { "key" : "value" } '
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = lambda *args: None
    var_1 = lambda *args: None
    var_2 = False
    var_3 = lambda *args: None
    var_4 = lambda *args: None
    var_5 = {}
    var_6 = ''
    var_7 = ''
    var_8 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 8/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 12/24 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 9/16 statements.
# Partially parsed test__TokenizingJSONObject_error_missing_colon. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_error_missing_comma. Retrieved 5/13 statements.
# Partially parsed test__TokenizingJSONObject_error_unquoted_key. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, i: (var_5, i)

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key1'
    var_6 = 6
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'key2'
    var_9 = 19
    var_10 = 24
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 7
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenizing_json_object_empty_string. Retrieved 5/8 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 20/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/13 statements.
# Partially parsed test_TokenizingJSONObject_unquoted_key. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, e: (var_5, e)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key1'
    var_6 = 6
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value1'
    var_9 = 10
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 'key2'
    var_13 = 20
    var_14 = 25
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = 'value2'
    var_17 = 26
    var_18 = 32
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 14
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 15/20 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'parse_array'
    var_6 = 'memo'
    var_7 = True
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = {}
    var_13 = 'test'
    var_14 = 'false'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 5/10 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 5/13 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 5/10 statements.
# Partially parsed test__TokenizingJSONObject_raises_on_missing_colon. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_raises_on_missing_comma. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (var_4, i)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1": 1, "key2": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1": 1 "key2": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 10/15 statements.


def test_case_0():
    var_0 = lambda *args: ('', 0)
    var_1 = lambda *args: ([], 0)
    var_2 = lambda *args: ({}, 0)
    var_3 = False
    var_4 = lambda x: float(x)
    var_5 = lambda x: int(x)
    var_6 = {}
    var_7 = ''
    var_8 = 'null'
    var_9 = 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 2/17 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__make_scanner_with_null. Retrieved 6/12 statements.
# Partially parsed test__make_scanner_with_true. Retrieved 6/12 statements.
# Partially parsed test__make_scanner_with_false. Retrieved 6/12 statements.
# Partially parsed test__make_scanner_with_string. Retrieved 6/12 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 6/12 statements.
# Partially parsed test__make_scanner_with_empty_dict. Retrieved 6/15 statements.
# Partially parsed test__make_scanner_with_empty_list. Retrieved 6/12 statements.


def test_case_0():
    var_0 = lambda self, *args: ([], 4)
    var_1 = lambda self, *args: ('', 4)
    var_2 = True
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 4)
    var_1 = lambda self, *args: ('', 4)
    var_2 = True
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 5)
    var_1 = lambda self, *args: ('', 5)
    var_2 = True
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 5)
    var_1 = lambda self, *args: ('test', 6)
    var_2 = True
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 3)
    var_1 = lambda self, *args: ('', 3)
    var_2 = True
    var_3 = {}
    var_4 = '123'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 2)
    var_2 = True
    var_3 = {}
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 2)
    var_2 = True
    var_3 = {}
    var_4 = '[]'
    var_5 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 5/10 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 5/13 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, idx: (var_4, idx)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, idx: (var_4, idx)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, idx: (var_4, idx)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, idx: (var_4, idx)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__make_scanner_with_null. Retrieved 5/13 statements.
# Partially parsed test__make_scanner_with_true. Retrieved 5/13 statements.
# Partially parsed test__make_scanner_with_false. Retrieved 5/13 statements.
# Partially parsed test__make_scanner_with_string. Retrieved 5/13 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 5/13 statements.
# Partially parsed test__make_scanner_with_empty_dict. Retrieved 5/16 statements.
# Partially parsed test__make_scanner_with_empty_list. Retrieved 5/13 statements.


def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('test', idx + 6)
    var_1 = False
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = '123'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = lambda self, string, idx, strict: ('', idx + 2)
    var_1 = False
    var_2 = {}
    var_3 = '[]'
    var_4 = 0



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = None
    var_4 = 0
    var_5 = (var_3, var_4)
    var_6 = lambda x, y: var_5
    var_7 = {var_2: var_6}
    var_8 = ''
    var_9 = 'null'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 19/23 statements.


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
    var_11 = lambda *args: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda *args: var_13
    var_15 = False
    var_16 = {}
    var_17 = ''
    var_18 = 'false'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_parse_object_not_assigned_to_TokenizingJSONObject. Retrieved 4/6 statements.


import builtins as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = '{}'
    var_2 = module_1._make_scanner(var_0, var_1)
    var_3 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = lambda *args: ([], 4)
    var_1 = lambda *args: ('test', 5)
    var_2 = True
    var_3 = lambda x: float(x)
    var_4 = lambda x: int(x)
    var_5 = {}
    var_6 = 'test'
    var_7 = '"test"'
    var_8 = 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__make_scanner_with_string. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_dict. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_list. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_boolean_true. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_boolean_false. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_null. Retrieved 6/13 statements.
# Partially parsed test__make_scanner_with_number. Retrieved 6/13 statements.


def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('test', 6)
    var_2 = True
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('key', 5)
    var_2 = True
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: (['item1', 'item2'], 14)
    var_1 = lambda self, *args: ('item1', 7)
    var_2 = True
    var_3 = {}
    var_4 = '["item1", "item2"]'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = lambda self, *args: ([], 2)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = '123'
    var_5 = 0



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = 'true'
    var_7 = module_0.ScalarToken(var_5, var_1, var_2, var_6)
    var_8 = var_4 == var_7
    assert var_8 is False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_scanner_creates_dict_token_for_opening_brace. Retrieved 9/14 statements.


def test_case_0():
    var_0 = lambda *args: ([], 1)
    var_1 = lambda *args: ('', 1)
    var_2 = True
    var_3 = lambda x: float(x)
    var_4 = lambda x: int(x)
    var_5 = {}
    var_6 = ''
    var_7 = '{'
    var_8 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_simple_object. Retrieved 6/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 13/23 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 8/15 statements.
# Partially parsed test__TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, i: (ScalarToken(var_6, i, i, s), i + var_5)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key1'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'key2'
    var_9 = 16
    var_10 = 19
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = len(var_0)

def test_case_0():
    var_0 = ' { "key" : "value" } '
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 0
    var_6 = len(var_0)
    var_7 = var_6 - var_4

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_scanner_predicate_evaluates_to_true. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = None
    var_9 = False
    var_10 = {}
    var_11 = 'null'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_multiple_keys. Retrieved 6/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{}'
    var_5 = (var_0, var_1)
    var_6 = None

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = (var_0, var_1)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{"key1": "value1", "key2": "value2"}'
    var_5 = (var_0, var_1)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{key: "value"}'
    var_5 = (var_0, var_1)
    var_6 = None
    var_7 = module_0._TokenizingJSONObject(var_5, var_2, var_6, var_3, var_4)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{"key" "value"}'
    var_5 = (var_0, var_1)
    var_6 = None
    var_7 = module_0._TokenizingJSONObject(var_5, var_2, var_6, var_3, var_4)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": value}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{"key": value}'
    var_5 = (var_0, var_1)
    var_6 = None
    var_7 = module_0._TokenizingJSONObject(var_5, var_2, var_6, var_3, var_4)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value" "key2": "value2"}'
    var_1 = 0
    var_2 = False
    var_3 = {}
    var_4 = '{"key": "value" "key2": "value2"}'
    var_5 = (var_0, var_1)
    var_6 = None
    var_7 = module_0._TokenizingJSONObject(var_5, var_2, var_6, var_3, var_4)



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'not None'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 18/21 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = lambda *args: var_8
    var_11 = False
    var_12 = lambda x: float(x)
    var_13 = lambda x: int(x)
    var_14 = {}
    var_15 = {var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_14}
    var_16 = ''
    var_17 = ''



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_scanner_string. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_dict. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_list. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_null. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_true. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_false. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_number. Retrieved 8/15 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'content'
    var_6 = '"test"'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'content'
    var_6 = '{"key": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'content'
    var_6 = '["item1", "item2"]'
    var_7 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'content'
    var_7 = 'null'
    var_8 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'content'
    var_7 = 'true'
    var_8 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = None
    var_5 = {}
    var_6 = 'content'
    var_7 = 'false'
    var_8 = 0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = None
    var_4 = {}
    var_5 = 'content'
    var_6 = '123'
    var_7 = 0



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'value'
    var_7 = lambda x, y: (ScalarToken(var_6, y, y, var_0), y)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_4, var_5, var_7, var_8, var_0)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = var_0
    var_2 = len(var_1)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = var_1[var_4]



