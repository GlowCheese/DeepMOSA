####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_string_data_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_dict_data_and_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_url_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_url_ending_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_url_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_and_get_method_lowercase. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_get_method_mixed_case. Retrieved 7/8 statements.
# Partially parsed test_query_with_none_data. Retrieved 5/6 statements.


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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

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
    var_1 = 'Get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = None
    var_4 = {var_2: var_3}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_urllib_get_with_data_appends_to_url. Retrieved 10/11 statements.
# Partially parsed test_urllib_get_without_data. Retrieved 6/7 statements.
# Partially parsed test_urllib_post_with_data. Retrieved 10/11 statements.
# Partially parsed test_urllib_get_with_existing_query_string. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_5_true. Retrieved 4/6 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_get_method_and_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_existing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_existing_question_mark_and_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_post_method_and_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_string_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_not_in_kwargs. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?a=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

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
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_requests_get_without_data. Retrieved 5/6 statements.
# Partially parsed test_requests_get_with_query_data. Retrieved 9/10 statements.
# Partially parsed test_requests_get_with_existing_query. Retrieved 9/10 statements.
# Partially parsed test_requests_post_with_data. Retrieved 9/10 statements.
# Partially parsed test_requests_with_session. Retrieved 7/9 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
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
    var_0 = 'http://example.com?existing=1'
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

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_1._requests(var_1, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_with_dict_data_for_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_for_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data_for_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_for_post_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_for_get_method_with_existing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_for_get_method_with_existing_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_and_no_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_empty_string_data_for_get. Retrieved 5/6 statements.


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
    var_3 = 'test'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
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
    var_1 = None
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = ''
    var_4 = {var_2: var_3}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_get_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_as_list. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_as_tuple. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_existing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_existing_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_get_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_post_with_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_lowercase_get. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_uppercase_get. Retrieved 7/8 statements.


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
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'param'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?a=1&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'b'
    var_4 = '2'
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
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'p'
    var_4 = 'q'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'p'
    var_4 = 'q'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_session. Retrieved 7/9 statements.
# Partially parsed test_url_opener_with_timeout. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_url_opener_get_with_data. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

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
    var_8 = module_0.url_opener(var_7, var_6)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = {var_1: var_3, var_2: var_0}
    var_5 = 'http://example.com'
    var_6 = module_1.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'gbk'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'q'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 9/13 statements.


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 12/24 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 'encoding'
    var_5 = {}
    var_6 = module_0.get(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.get(var_0, var_1, **var_7)
    var_9 = 'session'
    var_10 = {}
    var_11 = module_0.get(var_9, **var_10)
    var_12 = getattr(var_11, var_0)
    var_13 = {}
    var_14 = 'timeout'
    var_15 = 200



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_false_when_method_not_basestring. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_urllib_get_with_data_appends_to_url. Retrieved 10/11 statements.
# Partially parsed test_urllib_get_without_data. Retrieved 6/7 statements.
# Partially parsed test_urllib_post_with_data. Retrieved 10/11 statements.
# Partially parsed test_urllib_default_timeout. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 30
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = '?key=value'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 30
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_no_question_mark. Retrieved 5/6 statements.
# Partially parsed test_query_with_get_method_and_question_mark_at_end. Retrieved 5/6 statements.
# Partially parsed test_query_with_get_method_and_question_mark_with_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_get_method_and_ampersand_at_end. Retrieved 5/6 statements.
# Partially parsed test_query_with_post_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_string_method_and_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_uppercase_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_already_encoded. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_none. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

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
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = None
    var_4 = {var_2: var_3}



# Parsed testcases at query #15
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_string_data_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_and_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_get_method_with_existing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_get_method_with_existing_question_mark_and_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_and_get_method_without_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_other_types. Retrieved 5/6 statements.


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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'GET'
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

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 123
    var_4 = {var_2: var_3}



# Parsed testcases at query #17
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_query_get_with_dict_data. Retrieved 9/10 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_data_and_existing_query. Retrieved 6/7 statements.
# Partially parsed test_query_get_with_data_and_url_ending_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_and_url_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_and_no_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_no_data. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '?'
    var_10 = 'key1=value1'
    var_11 = 'key2=value2'

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key1=value1&key2=value2'
    var_4 = {var_2: var_3}
    var_5 = '?'
    var_6 = 'key1=value1'
    var_7 = 'key2=value2'

def test_case_0():
    var_0 = 'http://example.com/api?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = {var_3: var_2}
    var_5 = {var_2: var_4}
    var_6 = 'existing=param'
    var_7 = 'new=data'
    var_8 = '&'

def test_case_0():
    var_0 = 'http://example.com/api?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com/api?existing&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'param'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '?'

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'post'
    var_2 = {}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_requests_with_get_method_and_session. Retrieved 6/11 statements.
# Partially parsed test_requests_with_get_method_no_session. Retrieved 7/10 statements.
# Partially parsed test_requests_with_post_method. Retrieved 9/12 statements.
# Partially parsed test_requests_http_error. Retrieved 5/8 statements.
# Partially parsed test_requests_with_timeout. Retrieved 7/10 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'success'

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
    assert var_8 == 'posted'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'timeout_test'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'latin-1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'encoded'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_query_with_dict_data_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_tuple_data_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_get_method_url_has_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_get_method_url_ends_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_get_method_url_ends_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_get_method_no_question_mark_url. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_non_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_non_get_method_list. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_non_get_method_tuple. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_get_method_case_insensitive. Retrieved 7/8 statements.


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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
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
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_urllib_get_with_data. Retrieved 11/13 statements.
# Partially parsed test_urllib_post_with_data. Retrieved 10/11 statements.
# Partially parsed test_urllib_without_data. Retrieved 6/7 statements.
# Partially parsed test_urllib_default_timeout. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'GET'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = 'http://example.com/api?key=value'

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'POST'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com/api?key=value'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'http://example.com/test?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'param=value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 6/14 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 30
    var_5 = {}
    var_6 = module_0.get(var_0, var_4, **var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = {}



# Parsed testcases at query #25
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}
    var_8 = module_0._query(var_0, var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}
    var_8 = module_0._query(var_0, var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'rawdata'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'b'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?a=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'b'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?a=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'b'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = b'raw'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_requests_with_get_method_and_no_session. Retrieved 5/6 statements.
# Partially parsed test_requests_with_post_method. Retrieved 9/10 statements.
# Partially parsed test_requests_with_session. Retrieved 7/9 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
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
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_1._requests(var_1, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
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
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = None
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

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
    var_8 = module_0.url_opener(var_0, var_7)

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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'method'
    var_3 = 'utf-8'
    var_4 = 'get'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_urllib_get_with_data_appends_query_string. Retrieved 13/14 statements.
# Partially parsed test_urllib_get_with_data_and_existing_query. Retrieved 11/12 statements.
# Partially parsed test_urllib_get_with_data_and_existing_question_mark. Retrieved 11/12 statements.
# Partially parsed test_urllib_get_with_data_and_existing_ampersand. Retrieved 11/12 statements.
# Partially parsed test_urllib_post_with_data_encodes_utf8. Retrieved 12/13 statements.
# Partially parsed test_urllib_get_without_data_returns_original_url. Retrieved 7/8 statements.
# Partially parsed test_urllib_post_without_data_returns_none_data. Retrieved 7/8 statements.
# Partially parsed test_urllib_with_list_data_encodes_properly. Retrieved 11/12 statements.
# Partially parsed test_urllib_with_tuple_data_encodes_properly. Retrieved 11/12 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = {var_5: var_6}
    var_11 = {var_2: var_10}
    var_12 = 'http://example.com?key=value'

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_4: var_5}
    var_9 = {var_2: var_8}
    var_10 = 'http://example.com?existing=1&key=value'

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_4: var_5}
    var_9 = {var_2: var_8}
    var_10 = 'http://example.com?key=value'

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_4: var_5}
    var_9 = {var_2: var_8}
    var_10 = 'http://example.com?existing=1&key=value'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_4: var_5}
    var_9 = {var_2: var_8}
    var_10 = 'http://example.com'
    var_11 = b'key=value'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'post'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = [var_4, var_5]
    var_9 = {var_2: var_8}
    var_10 = b'a&b'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'x'
    var_5 = 'y'
    var_6 = (var_4, var_5)
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = (var_4, var_5)
    var_9 = {var_2: var_8}
    var_10 = b'x&y'

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
    var_8 = {var_4: var_5}
    var_9 = {var_2: var_8}
    var_10 = module_0._query(var_0, var_3, var_9)
    var_11 = 'data'
    var_12 = bool('data' not in var_9)
    assert var_12 is True



# Parsed testcases at query #29
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #30
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_requests_get_with_session. Retrieved 7/16 statements.
# Partially parsed test_requests_get_without_session. Retrieved 8/13 statements.
# Partially parsed test_requests_http_error. Retrieved 5/11 statements.
# Partially parsed test_requests_with_custom_timeout. Retrieved 8/13 statements.
# Partially parsed test_requests_with_encoding. Retrieved 9/14 statements.
# Partially parsed test_requests_get_with_data_conversion. Retrieved 11/17 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'response text'
    var_7 = 10

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'ok'
    var_7 = 30

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'latin-1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'text'
    var_7 = 'latin-1'
    var_8 = var_1 == var_7

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
    var_9 = 'url'
    var_10 = 1
    var_11 = 'key=value'



# Parsed testcases at query #2
#--------------------------




import builtins as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 'Response'
    var_5 = ()
    var_6 = 'status_code'
    var_7 = 'url'
    var_8 = 'reason'
    var_9 = 'headers'
    var_10 = 'text'
    var_11 = 200
    var_12 = 'OK'
    var_13 = {}
    var_14 = ''
    var_15 = {var_6: var_11, var_7: var_3, var_8: var_12, var_9: var_13, var_10: var_14}
    var_16 = [var_4, var_5, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = lambda url, timeout=5, **kw: var_19
    var_21 = module_1._requests(var_3, var_2)



# Parsed testcases at query #3
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'GET'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._urllib(var_0, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'POST'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._urllib(var_0, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._urllib(var_0, var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._urllib(var_0, var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_method_not_get_evaluates_false. Retrieved 8/9 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'post'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = 'get'
    var_7 = {}
    var_8 = module_0.get(var_0, var_6, **var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_requests_get_no_session. Retrieved 5/6 statements.
# Partially parsed test_requests_get_with_session. Retrieved 4/13 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_requests_get_http_error. Retrieved 4/15 statements.
# Partially parsed test_requests_post. Retrieved 4/13 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'post'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_string_data_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_no_data_and_get_method. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_and_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_empty_dict_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_url_containing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_url_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_dict_list_tuple_data. Retrieved 5/6 statements.


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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 12345
    var_4 = {var_2: var_3}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_dict_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_tuple_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_existing_query_string_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_trailing_question_mark_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_trailing_ampersand_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_lowercase_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_uppercase_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_not_dict_list_tuple_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_not_dict_list_tuple_post_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_empty_dict_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_empty_list_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_empty_tuple_get_method. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

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

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'stringdata'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'stringdata'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = []
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = ()
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_requests_get_with_data_and_session. Retrieved 12/17 statements.
# Partially parsed test_requests_get_without_session. Retrieved 13/16 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 13/19 statements.
# Partially parsed test_requests_with_encoding. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'data'
    var_4 = 'encoding'
    var_5 = 'timeout'
    var_6 = 'get'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'utf-8'
    var_11 = 10

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = 10
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)
    assert var_12 == 'response text'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = 10
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    assert var_8 == 'response text'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com/?existing=param'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_false_when_method_not_string. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'some_data'
    var_1 = 'http://example.com'
    var_2 = 123
    var_3 = 'data'
    var_4 = {var_3: var_0}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_url_opener_requests_get. Retrieved 9/10 statements.
# Partially parsed test_url_opener_requests_get_with_data. Retrieved 11/12 statements.
# Partially parsed test_url_opener_requests_post. Retrieved 11/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'utf-8'
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'encoding'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'utf-8'
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://httpbin.org/status/404'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'post'
    var_3 = 'key=value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key=value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #12
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
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?key=value', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?a=b', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?x=y', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?foo=bar'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?foo=bar&key=value', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?foo=bar&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?foo=bar&key=value', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?key=value', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = bool(var_3 == ('http://example.com', None))
    assert var_4 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com', b'key=value'))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com', b'a=b'))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com', b'x=y'))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = bool(var_3 == ('http://example.com', None))
    assert var_4 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com?key=value', None))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    var_8 = bool(var_7 == ('http://example.com', b'key=value'))
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0._query(var_0, var_1, var_8)
    var_10 = bool(var_9 == ('http://example.com?a=1&b=2', None))
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_url_opener_requests_get. Retrieved 7/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 'url'
    var_4 = 'reason'
    var_5 = 'headers'
    var_6 = 'text'
    var_7 = 'encoding'
    var_8 = 200
    var_9 = ''
    var_10 = {}
    var_11 = None
    var_12 = {var_2: var_8, var_3: var_9, var_4: var_9, var_5: var_10, var_6: var_9, var_7: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = 200
    var_18 = bool(200 <= var_16.status_code)
    assert var_18 is True
    var_19 = var_16.status_code
    var_20 = bool(var_16.status_code < 300)
    assert var_20 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_false_when_method_is_not_basestring. Retrieved 5/6 statements.
# Partially parsed test_predicate_false_when_method_lower_is_not_get. Retrieved 5/6 statements.
# Partially parsed test_predicate_false_when_data_is_none. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_status_code_in_success_range. Retrieved 11/13 statements.


import requests.cookies as module_0

def test_case_0():
    var_0 = 200
    var_1 = module_0.MockResponse(var_0)
    var_2 = lambda url, timeout, **kw: var_1
    var_3 = 'method'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 5
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'http://example.com'
    var_9 = []
    var_10 = {}
    var_11 = 200
    var_12 = bool(200 <= var_1.status_code)
    assert var_12 is True
    var_13 = var_1.status_code
    var_14 = bool(var_1.status_code < 300)
    assert var_14 is True



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_url_opener_when_has_request_is_false. Retrieved 5/7 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = False
    var_3 = module_0.url_opener(var_0, var_1)
    var_4 = module_0._urllib(var_0, var_1)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_status_code_in_success_range_does_not_raise. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 200
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = []
    var_6 = None
    var_7 = 'get'
    var_8 = None
    var_9 = 5
    var_10 = {}
    var_11 = 200



# Parsed testcases at query #20
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'session'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    var_11 = bool(var_10 is not None)
    assert var_11 is True

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = {var_1: var_3, var_2: var_0}
    var_5 = 'http://example.com'
    var_6 = module_1._requests(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 10
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_requests_get_without_data. Retrieved 9/10 statements.
# Partially parsed test_requests_get_with_query_data. Retrieved 13/14 statements.
# Partially parsed test_requests_post_without_data. Retrieved 9/10 statements.
# Partially parsed test_requests_post_with_data. Retrieved 13/14 statements.
# Partially parsed test_requests_with_session. Retrieved 9/11 statements.
# Partially parsed test_requests_with_timeout. Retrieved 11/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'post'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'post'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_2: var_5, var_3: var_6, var_4: var_0}
    var_8 = module_1._requests(var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 5
    var_7 = 'utf-8'
    var_8 = None
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0._requests(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'invalid'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_url_opener_with_requests_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_requests_and_post_method. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_requests_and_timeout. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_requests_and_encoding. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_requests_and_session. Retrieved 7/8 statements.


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
    var_2 = 'post'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = None
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_url_opener_with_requests_module. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = 'read'
    var_8 = hasattr(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_requests_get_with_session. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'headers'
    var_3 = 'get'
    var_4 = 'User-Agent'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/error'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

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
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True

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
    var_8 = module_0.url_opener(var_7, var_6)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

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
    var_8 = module_0.url_opener(var_7, var_6)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = {var_1: var_3, var_2: var_0}
    var_5 = 'http://example.com'
    var_6 = module_1.url_opener(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



