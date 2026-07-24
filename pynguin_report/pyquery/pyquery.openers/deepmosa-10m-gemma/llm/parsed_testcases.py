####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 10/24 statements.
# Partially parsed test_requests_get_with_data_encoding. Retrieved 15/30 statements.
# Partially parsed test_requests_http_error_raises. Retrieved 7/21 statements.
# Partially parsed test_requests_uses_session_if_provided. Retrieved 6/23 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'method'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_4: var_5, var_1: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'encoding'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'val'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = {var_3: var_5, var_0: var_8, var_4: var_9}
    var_11 = 'http://example.com'
    var_12 = module_0._requests(var_11, var_10)
    assert var_12 == 'encoded'
    var_13 = 'url'
    var_14 = 1
    var_15 = 'key=val'

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com/fail'
    var_6 = module_0._requests(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_url_opener_requests_get_success. Retrieved 6/15 statements.
# Partially parsed test_url_opener_requests_error. Retrieved 6/18 statements.
# Partially parsed test_query_logic_with_data_dict. Retrieved 9/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests.get'
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)
    assert var_5 == '<html>content</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests.get'
    var_1 = 'http://example.com/bad'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?key=value'

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'new'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = var_7[var_2]
    var_9 = module_0.urlencode(var_8)
    var_10 = 'http://example.com?existing=true&new=val'
    var_11 = '?'
    var_12 = var_0 + var_11
    var_13 = '&'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_9
    var_16 = bool(var_15 == var_10)
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_get_with_list_data. Retrieved 14/16 statements.
# Partially parsed test_query_get_with_existing_params. Retrieved 11/12 statements.
# Partially parsed test_query_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_post_encoding. Retrieved 7/9 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b'
    var_8 = 'http://example.com'
    var_9 = 'get'
    var_10 = 'k'
    var_11 = 'v'
    var_12 = {var_10: var_11}
    var_13 = {var_2: var_12}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=1'
    var_8 = 'get'
    var_9 = {var_3: var_4}
    var_10 = {var_2: var_9}
    var_11 = 'existing=1'
    var_12 = 'new=2'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_requests_success_get_with_params.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_get_method_with_dict_data. Retrieved 9/11 statements.
# Partially parsed test_query_post_method_with_dict_data. Retrieved 12/14 statements.
# Partially parsed test_query_get_method_with_existing_query_params. Retrieved 13/15 statements.
# Partially parsed test_query_get_method_with_trailing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_no_data_in_kwargs. Retrieved 3/4 statements.
# Partially parsed test_query_list_data_encoding. Retrieved 11/13 statements.
# Partially parsed test_query_case_insensitive_get. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None

import urllib.parse as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'param'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example/example.com?existing=true&new=param'
    var_8 = 'http://example.com?existing=true&new=param'
    var_9 = 'http://example.com?existing=true'
    var_10 = 'get'
    var_11 = {var_3: var_4}
    var_12 = {var_2: var_11}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

import urllib.parse as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_3, var_4]
    var_8 = module_0.urlencode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'id'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 9/25 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 8/23 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 7/25 statements.
# Partially parsed test_requests_uses_session. Retrieved 4/22 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'params'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_2: var_3, var_0: var_4}
    var_6 = 'http://example.com'
    var_7 = module_0._requests(var_6, var_5)
    assert var_7 == '<html>success</html>'
    var_8 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = module_0._requests(var_6, var_5)
    assert var_7 == '<html>utf8</html>'

import pyquery.openers as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0._requests(var_3, var_2)
    var_5 = bool(True)
    assert var_5 is True
    var_6 = 'HTTPError was not raised'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.AssertionError(*var_7, **var_8)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #7
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'param'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=true&new=param'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)
    var_12 = module_2._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_7, var_11))
    assert var_13 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None
    var_5 = module_0._query(var_0, var_1, var_2)
    var_6 = bool(var_5 == (var_3, var_4))
    assert var_6 is True

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_3, var_4]
    var_8 = module_0.urlencode(var_7)
    var_9 = 'http://example.com?'
    var_10 = var_9 + var_8
    var_11 = None
    var_12 = module_1._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_10, var_11))
    assert var_13 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?c=d'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 'auth'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'session'
    var_7 = 'get'
    var_8 = 10
    var_9 = 'http://example.com'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_url_opener_evaluates_predicate_to_false. Retrieved 3/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_urllib_get_with_data_params. Retrieved 12/17 statements.
# Partially parsed test_urllib_post_with_data_payload. Retrieved 15/20 statements.
# Partially parsed test_urllib_get_with_existing_query_params. Retrieved 12/17 statements.
# Partially parsed test_urllib_with_custom_timeout. Retrieved 10/14 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?key=value'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)
    var_11 = 10

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'POST'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com'
    var_9 = {var_4: var_5}
    var_10 = module_0.urlencode(var_9)
    var_11 = 'utf-8'
    var_12 = module_1.encode(var_11)
    var_13 = module_2._urllib(var_0, var_7)
    var_14 = 10

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'param'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?existing=true&new=param'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)
    var_11 = 10

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = None
    var_8 = module_0._urllib(var_0, var_5)
    var_9 = 5



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_requests_success_get. Retrieved 15/28 statements.
# Partially parsed test_requests_failure_raises_http_error. Retrieved 8/21 statements.
# Partially parsed test_requests_session_usage. Retrieved 8/22 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'headers'
    var_2 = 'timeout'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'encoding'
    var_7 = 'get'
    var_8 = 'key'
    var_9 = 'val'
    var_10 = {var_8: var_9}
    var_11 = 'utf-8'
    var_12 = {var_5: var_7, var_0: var_10, var_6: var_11}
    var_13 = 'http://example.com'
    var_14 = module_0._requests(var_13, var_12)
    assert var_14 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'get'
    var_5 = {var_3: var_4}
    var_6 = 'http://example.com'
    var_7 = module_0._requests(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 10
    var_3 = 'method'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 2
    var_7 = 'http://example.com'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_false_method_not_string. Retrieved 6/13 statements.
# Partially parsed test_predicate_false_method_not_get. Retrieved 5/10 statements.
# Partially parsed test_predicate_false_no_data. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 123

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 7/17 statements.
# Partially parsed test_requests_get_with_data_encoding. Retrieved 11/22 statements.
# Partially parsed test_requests_raises_http_error_on_404. Retrieved 5/14 statements.
# Partially parsed test_requests_uses_session_if_provided. Retrieved 4/17 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 'utf-8'
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)
    assert var_10 == 'content'
    var_11 = 'a=b'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_requests_success_get. Retrieved 7/15 statements.
# Partially parsed test_requests_error_raises_exception. Retrieved 5/14 statements.
# Partially parsed test_requests_with_session_and_encoding. Retrieved 6/15 statements.
# Partially parsed test_requests_with_data_in_get. Retrieved 9/16 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)
    assert var_8 == 'result'
    var_9 = 'key=value'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_not_string. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'name=test'
    var_4 = {var_2: var_3}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_query_predicate_false_by_method_not_string. Retrieved 5/9 statements.
# Partially parsed test_query_predicate_false_by_method_not_get. Retrieved 5/9 statements.
# Partially parsed test_query_predicate_false_by_no_data. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'name=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'name=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_requests_predicate_false_when_method_is_post. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'timeout'
    var_3 = 'post'
    var_4 = 5
    var_5 = 'http://example.com'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_requests_with_session_exists. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_get_with_dict_data. Retrieved 9/11 statements.
# Partially parsed test_query_get_with_existing_params. Retrieved 9/11 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 12/14 statements.
# Partially parsed test_query_no_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_list_data. Retrieved 12/14 statements.
# Partially parsed test_query_with_trailing_ampersand. Retrieved 9/11 statements.
# Partially parsed test_query_with_trailing_question_mark. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&key=value'
    var_8 = None

import urllib.parse as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'http://example/example.com?a=%5Ba%2C+b%5D'
    var_8 = [var_3, var_4]
    var_9 = module_0.urlencode(var_8)
    var_10 = 'http://example.com?'
    var_11 = var_10 + var_9

def test_case_0():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&key=value'
    var_8 = None

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_query_predicate_true. Retrieved 16/25 statements.


import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'post'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_9}
    var_11 = module_0.urlencode(var_10)
    var_12 = 'http://example.com'
    var_13 = 'post'
    var_14 = {var_3: var_4}
    var_15 = {var_2: var_14}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_urllib_get_with_data_params. Retrieved 11/16 statements.
# Partially parsed test_urllib_post_with_data_payload. Retrieved 12/17 statements.
# Partially parsed test_urllib_get_with_existing_query_params. Retrieved 11/16 statements.
# Partially parsed test_urllib_with_custom_timeout. Retrieved 9/13 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?key=value'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.urlencode(var_3)
    var_5 = 'utf-8'
    var_6 = module_1.encode(var_5)
    var_7 = 'method'
    var_8 = 'data'
    var_9 = 'POST'
    var_10 = {var_7: var_9, var_8: var_3}
    var_11 = module_2._urllib(var_0, var_10)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'param'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?existing=true&new=param'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._urllib(var_0, var_5)
    var_7 = None
    var_8 = 30



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_type. Retrieved 6/10 statements.
# Partially parsed test_query_predicate_false_due_to_method_name. Retrieved 5/6 statements.
# Partially parsed test_query_predicate_false_due_to_no_data. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    assert var_0 == 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'param=value'
    var_4 = {var_2: var_3}
    var_5 = 123

def test_case_0():
    var_0 = 'http://example.com'
    assert var_0 == 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'param=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_url_opener_requests_get_with_data. Retrieved 9/18 statements.
# Partially parsed test_url_opener_requests_error_raises_exception. Retrieved 5/15 statements.
# Partially parsed test_url_opener_urllib_path. Retrieved 6/11 statements.
# Partially parsed test_query_logic_url_encoding. Retrieved 11/19 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == 'success'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'urllib_content'
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)
    assert var_5 == 'urllib_content'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    var_9 = 'url'
    var_10 = 1
    var_11 = 'key=val'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_requests_success_get. Retrieved 7/20 statements.
# Partially parsed test_requests_error_raises_http_error. Retrieved 5/17 statements.
# Partially parsed test_requests_with_session. Retrieved 4/12 statements.
# Partially parsed test_requests_get_with_params_encoding. Retrieved 11/21 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'encoding'
    var_2 = 5
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = 'latin-1'
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == 'encoded'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status_code. Retrieved 8/27 statements.
# Partially parsed test_requests_predicate_evaluates_to_false_on_404. Retrieved 5/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = 'requests'
    var_3 = 'http://example.com'
    var_4 = 'method'
    var_5 = 'get'
    var_6 = {var_4: var_5}
    var_7 = module_0._requests(var_3, var_6)

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = 'Not Found'
    var_3 = {}
    var_4 = ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 1/23 statements.


def test_case_0():
    var_0 = 'requests'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status_code. Retrieved 6/21 statements.
# Partially parsed test_predicate_evaluates_to_false. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com/error'
    var_2 = 'session'
    var_3 = 'method'
    var_4 = 'get'
    var_5 = 'http://example.com/error'

import requests.api as module_0

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = 'Not Found'
    var_3 = {}
    var_4 = ''
    var_5 = 'session'
    var_6 = 'method'
    var_7 = 'get'
    var_8 = 'http://example.com'
    var_9 = 10
    var_10 = 'timeout'
    var_11 = {var_10: var_9}
    var_12 = module_0.get(var_8, **var_11)
    var_13 = bool(not 200 <= var_12.status_code < 300)
    assert var_13 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 9/19 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 5/12 statements.
# Partially parsed test_requests_http_error_raises. Retrieved 5/12 statements.
# Partially parsed test_requests_session_usage. Retrieved 4/16 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == '<html>success</html>'
    var_7 = 'http://example.com'
    var_8 = 5

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    assert var_4 == 'content'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/bad'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'session'
    var_2 = 'method'
    var_3 = 'get'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_not_get. Retrieved 7/11 statements.
# Partially parsed test_query_predicate_false_due_to_missing_data. Retrieved 3/4 statements.
# Partially parsed test_query_predicate_false_due_to_non_string_method. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = None
    var_2 = 'data'
    var_3 = 'param'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_requests_predicate_false_when_method_is_post. Retrieved 11/24 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 10
    var_3 = None
    var_4 = lambda url, method, kwargs: (url, var_3)
    var_5 = 'method'
    var_6 = 'post'
    var_7 = 5
    var_8 = {var_5: var_6, var_0: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == 'success'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_requests_with_session_evaluates_true. Retrieved 5/29 statements.


def test_case_0():
    var_0 = None
    var_1 = 'session'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = 'http://example.com'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = 'method'
    var_5 = 'session'
    var_6 = 'get'
    var_7 = 5
    var_8 = 'User-Agent'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = 'http://example.com'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_not_get. Retrieved 5/7 statements.
# Partially parsed test_query_predicate_false_due_to_missing_data. Retrieved 3/4 statements.
# Partially parsed test_query_predicate_false_due_to_no_data_in_kwargs. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'other'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_url_opener_evaluates_predicate_to_false. Retrieved 3/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 9/21 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = [var_0, var_1]
    var_3 = 'other'
    var_4 = 30
    var_5 = 'unused'
    var_6 = {var_0: var_4, var_3: var_5}
    var_7 = 'http://test.com'
    var_8 = module_0._requests(var_7, var_6)
    assert var_8 == 'success'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_requests_with_session. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_url_opener_evaluates_predicate_to_false. Retrieved 3/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 30
    var_6 = 'http://example.com'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 12/25 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = [var_0, var_1]
    var_3 = 'method'
    var_4 = 'get'
    var_5 = 10
    var_6 = 'User-Agent'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_4, var_0: var_5, var_1: var_8}
    var_10 = 'http://example.com'
    var_11 = module_0._requests(var_10, var_9)
    assert var_11 == 'success'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 17/38 statements.
# Partially parsed test_requests_get_encoding. Retrieved 10/29 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 10/38 statements.
# Partially parsed test_requests_with_session. Retrieved 7/29 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'headers'
    var_2 = 'timeout'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'get'
    var_7 = 10
    var_8 = 'User-Agent'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_5: var_6, var_2: var_7, var_1: var_10}
    var_12 = 'http://example.com'
    var_13 = module_0._requests(var_12, var_11)
    assert var_13 == '<html>success</html>'
    var_14 = 'http://example.com'
    var_15 = {var_8: var_9}
    var_16 = 'requests'

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)
    assert var_8 == 'content'
    var_9 = 'requests'

import pyquery.openers as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com/bad'
    var_6 = module_0._requests(var_5, var_4)
    var_7 = 'HTTPError was not raised'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.AssertionError(*var_8, **var_9)
    var_11 = 'requests'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'
    var_6 = 'requests'



