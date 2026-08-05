####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__urllib_with_get_method_and_data. Retrieved 9/11 statements.
# Partially parsed test__urllib_with_post_method_and_data. Retrieved 9/11 statements.
# Partially parsed test__urllib_with_timeout. Retrieved 5/6 statements.
# Partially parsed test__urllib_with_default_timeout. Retrieved 3/4 statements.


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
    var_8 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'POST'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'timeout'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = module_0._urllib(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0._urllib(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__requests_get_with_session. Retrieved 9/10 statements.
# Partially parsed test__requests_get_without_session. Retrieved 7/8 statements.
# Partially parsed test__requests_post_with_session. Retrieved 11/12 statements.
# Partially parsed test__requests_post_without_session. Retrieved 9/10 statements.
# Partially parsed test__requests_with_timeout. Retrieved 7/8 statements.
# Partially parsed test__requests_with_custom_encoding. Retrieved 7/8 statements.


import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = {var_1: var_4, var_2: var_0, var_3: var_5}
    var_7 = 'http://example.com'
    var_8 = module_1._requests(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'data'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_1: var_4, var_2: var_0, var_3: var_7}
    var_9 = 'http://example.com'
    var_10 = module_1._requests(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'post'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'latin-1'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com/404'
    var_4 = module_0._requests(var_3, var_2)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/12 statements.
# Partially parsed test_url_opener_with_requests_http_error. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/7 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 9/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == b'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == b'test'



# Parsed testcases at query #4
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 400
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = bool(not 200 <= var_8.status_code < 300)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_post_method_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'get'
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
    var_1 = 'post'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #6
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = bool(var_3 == (var_0, None))
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_question. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = bool(not 200 <= var_8.status_code < 300)
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/12 statements.
# Partially parsed test_url_opener_with_requests_http_error. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/7 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 9/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test html'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == 'test html'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test html'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == 'test html'



# Parsed testcases at query #11
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = bool(not 200 <= var_7.status_code < 300)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 404
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = bool(not 200 <= var_7.status_code < 300)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 5/9 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = 'get'



# Parsed testcases at query #15
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = bool(not 200 <= var_8.status_code < 300)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?key=value'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '?'
    var_8 = bool('?' in var_0)
    assert var_8 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param=value'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_false. Retrieved 3/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0._requests(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param=value'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '?'
    var_8 = bool('?' in var_0)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_requests_timeout. Retrieved 7/10 statements.
# Partially parsed test_url_opener_with_requests_encoding. Retrieved 7/10 statements.
# Partially parsed test_url_opener_with_requests_session. Retrieved 7/10 statements.
# Partially parsed test_url_opener_with_requests_http_error. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/7 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 5/7 statements.
# Partially parsed test_url_opener_with_urllib_timeout. Retrieved 7/9 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'post'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == 'test'

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_1.url_opener(var_1, var_5)
    assert var_6 == 'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == b'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'post'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == b'test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == b'test'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_with_data_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_string. Retrieved 5/6 statements.
# Partially parsed test_query_with_existing_query_string. Retrieved 7/8 statements.
# Partially parsed test_query_with_existing_query_string_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_non_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_case_insensitive_get. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

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

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 5/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = 'get'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__urllib_with_get_method_and_data. Retrieved 9/11 statements.
# Partially parsed test__urllib_with_post_method_and_data. Retrieved 9/11 statements.
# Partially parsed test__urllib_with_timeout. Retrieved 5/6 statements.
# Partially parsed test__urllib_with_no_data. Retrieved 3/5 statements.
# Partially parsed test__urllib_with_list_data. Retrieved 10/12 statements.
# Partially parsed test__urllib_with_tuple_data. Retrieved 10/12 statements.


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
    var_8 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'POST'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'timeout'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = module_0._urllib(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0._urllib(var_0, var_1)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = {var_1: var_3, var_2: var_7}
    var_9 = module_0._urllib(var_0, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = (var_4, var_5)
    var_7 = (var_6,)
    var_8 = {var_1: var_3, var_2: var_7}
    var_9 = module_0._urllib(var_0, var_8)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__requests_with_get_method. Retrieved 9/10 statements.
# Partially parsed test__requests_with_post_method. Retrieved 13/14 statements.
# Partially parsed test__requests_with_session. Retrieved 11/12 statements.
# Partially parsed test__requests_with_custom_encoding. Retrieved 9/10 statements.
# Partially parsed test__requests_with_timeout. Retrieved 7/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'timeout'
    var_5 = 'post'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = 10
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'encoding'
    var_5 = 'timeout'
    var_6 = 'get'
    var_7 = 'utf-8'
    var_8 = 10
    var_9 = {var_2: var_6, var_3: var_1, var_4: var_7, var_5: var_8}
    var_10 = module_1._requests(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'latin-1'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/12 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/7 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 9/11 statements.
# Partially parsed test_url_opener_with_requests_http_error. Retrieved 5/9 statements.
# Partially parsed test_url_opener_with_urllib_timeout. Retrieved 8/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'test response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == 'test response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == b'test response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    assert var_8 == b'test response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 1
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.url_opener(var_1, var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ending. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_not_in_kwargs. Retrieved 5/6 statements.
# Partially parsed test_query_with_non_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_uppercase_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_mixed_case_get_method. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 123
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GeT'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_requests_get_without_data. Retrieved 5/6 statements.
# Partially parsed test_requests_get_with_data. Retrieved 9/10 statements.
# Partially parsed test_requests_post. Retrieved 9/10 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_requests_with_session. Retrieved 7/8 statements.
# Partially parsed test_requests_with_timeout. Retrieved 7/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_1}
    var_6 = module_1._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__requests_with_get_method_and_no_data. Retrieved 5/6 statements.
# Partially parsed test__requests_with_post_method. Retrieved 9/10 statements.
# Partially parsed test__requests_with_custom_encoding. Retrieved 5/6 statements.
# Partially parsed test__requests_with_session. Retrieved 5/6 statements.
# Partially parsed test__requests_with_timeout. Retrieved 5/6 statements.
# Partially parsed test__requests_with_custom_headers. Retrieved 7/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'session'
    var_3 = {var_2: var_1}
    var_4 = module_1._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'timeout'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://invalid.url'
    var_1 = {}
    var_2 = module_0._requests(var_0, var_1)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'headers'
    var_2 = 'User-Agent'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #10
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = bool(not 200 <= var_7.status_code < 300)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_with_data_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_string. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_data_no_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_existing_query_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_case_insensitive_method. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'get'
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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #12
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = bool(not 200 <= var_7.status_code < 300)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param=value'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '?'
    var_8 = bool('?' in var_0)
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = bool(var_3 == (var_0, None))
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?key=value'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '?'
    var_8 = bool('?' in var_0)
    assert var_8 is True



# Parsed testcases at query #18
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 199
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = bool(not 200 <= var_8.status_code < 300)
    assert var_9 is True



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_question. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_non_string_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param?'
    var_1 = 'get'
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
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 123
    var_4 = {var_2: var_3}



