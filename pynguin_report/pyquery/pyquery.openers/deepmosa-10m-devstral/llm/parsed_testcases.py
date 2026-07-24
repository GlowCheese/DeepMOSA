####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_with_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_non_string_method. Retrieved 7/8 statements.


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
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 6/9 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 10/13 statements.
# Partially parsed test_url_opener_with_timeout. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_session. Retrieved 7/8 statements.


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
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'data'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.url_opener(var_1, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
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

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_1}
    var_6 = module_1.url_opener(var_0, var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_post_method_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_non_string_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_string. Retrieved 5/6 statements.


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
    var_1 = None
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'plain string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_urllib_with_get_method_and_data. Retrieved 9/11 statements.
# Partially parsed test_urllib_with_post_method_and_data. Retrieved 9/11 statements.
# Partially parsed test_urllib_with_timeout. Retrieved 5/6 statements.
# Partially parsed test_urllib_with_no_data. Retrieved 3/5 statements.


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



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_uppercase_get_method. Retrieved 7/8 statements.


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
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #7
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
    var_3 = 'key=value'
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
    var_0 = 'http://example.com?foo=bar'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?foo=bar'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?foo=bar&'
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_requests_with_get_method. Retrieved 7/8 statements.
# Partially parsed test_requests_with_post_method. Retrieved 9/10 statements.
# Partially parsed test_requests_with_session. Retrieved 7/8 statements.
# Partially parsed test_requests_with_timeout. Retrieved 7/8 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_requests_with_data_in_get. Retrieved 9/10 statements.


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
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://invalid.url'
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
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__requests_with_get_method_and_data. Retrieved 13/14 statements.
# Partially parsed test__requests_with_post_method. Retrieved 13/14 statements.
# Partially parsed test__requests_with_session. Retrieved 9/10 statements.
# Partially parsed test__requests_with_encoding. Retrieved 9/10 statements.


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
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = module_0.Session()
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_1._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True

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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'example.com?param1=value1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_0[-1]
    var_8 = bool(var_0[-1] in ('?', '&'))
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_with_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query_ending_with_question. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
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
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'plain string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #14
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 'url'
    var_4 = 'reason'
    var_5 = 'headers'
    var_6 = 'text'
    var_7 = 404
    var_8 = 'http://example.com'
    var_9 = 'Not Found'
    var_10 = {}
    var_11 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_9}
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = bool(not 200 <= var_15.status_code < 300)
    assert var_16 is True



# Parsed testcases at query #15
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
    var_7 = 404
    var_8 = 'test'
    var_9 = 'Not Found'
    var_10 = {}
    var_11 = ''
    var_12 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = bool(not 200 <= var_16.status_code < 300)
    assert var_17 is True



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

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_question. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_as_string. Retrieved 5/6 statements.


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
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__requests_with_get_method_and_no_data. Retrieved 5/6 statements.
# Partially parsed test__requests_with_get_method_and_data. Retrieved 9/10 statements.
# Partially parsed test__requests_with_post_method. Retrieved 9/10 statements.
# Partially parsed test__requests_with_encoding. Retrieved 7/8 statements.
# Partially parsed test__requests_with_session. Retrieved 7/8 statements.
# Partially parsed test__requests_with_timeout. Retrieved 7/8 statements.


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
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = module_0.Session()
    var_5 = {var_1: var_3, var_2: var_4}
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
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_url_ends_with_question_mark. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_0[-1]
    assert var_7 == '?'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/9 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 10/14 statements.
# Partially parsed test_url_opener_with_requests_non_200_status. Retrieved 5/11 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 6/11 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 7/12 statements.


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
    var_9 = {var_4: var_5}

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
    assert var_4 == b'test html'
    var_5 = None

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = b'key=value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == b'test html'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param1=value1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = bool(not var_0[-1] not in ('?', '&'))
    assert var_7 is True



# Parsed testcases at query #23
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
    var_8 = var_7()
    var_9 = bool(not 200 <= var_8.status_code < 300)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_encoding. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_timeout. Retrieved 9/10 statements.


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
    var_3 = 'session'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0.url_opener(var_0, var_9)

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
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 10
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.url_opener(var_0, var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
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
    var_0 = 'http://example.com?existing=param'
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
    var_3 = 'raw string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_url_opener_with_urllib_get. Retrieved 6/7 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 12/13 statements.
# Partially parsed test_url_opener_with_timeout. Retrieved 6/7 statements.


import pyquery.openers as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = None
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    var_7 = {}
    var_8 = module_1.get(var_0, **var_7)
    var_9 = var_8.text
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import pyquery.openers as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'session'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0.url_opener(var_0, var_9)
    var_11 = {var_5: var_6}
    var_12 = {}
    var_13 = module_1.post(var_0, var_11, **var_12)
    var_14 = var_13.text
    var_15 = bool(var_10 == var_14)
    assert var_15 is True

import pyquery.openers as module_0
import urllib.request as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    var_5 = module_1.urlopen(var_0)

import pyquery.openers as module_0
import urllib.parse as module_1
import urllib.request as module_2

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
    var_9 = {var_4: var_5}
    var_10 = module_1.urlencode(var_9)
    var_11 = module_2.urlopen(var_0, var_10)

import pyquery.openers as module_0
import urllib.request as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'timeout'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    var_5 = module_1.urlopen(var_0, timeout=var_2)

import pyquery.openers as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    var_5 = 'encoding'
    var_6 = {var_5: var_2}
    var_7 = module_1.get(var_0, **var_6)
    var_8 = var_7.text
    var_9 = bool(var_4 == var_8)
    assert var_9 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_with_data_as_dict_in_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list_in_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple_in_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_string_in_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_in_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_existing_query_string. Retrieved 7/8 statements.
# Partially parsed test_query_with_existing_query_string_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_string_method. Retrieved 7/8 statements.


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
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

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
    var_1 = 'get'
    var_2 = {}

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
    var_1 = None
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
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
    var_0 = 'http://example.com?existing=param'
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
    var_3 = 'raw string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 7/8 statements.


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

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_with_ampersand. Retrieved 7/8 statements.
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
    var_2 = 'data'
    var_3 = 'raw string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_data_is_encoded_when_not_get_method. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'get'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_timeout. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_encoding. Retrieved 7/8 statements.


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
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

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
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
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




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = 'timeout'
    var_3 = 5
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)
    var_6 = {var_2: var_3}
    var_7 = module_0._urllib(var_1, var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = 'get'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_requests_get_without_data. Retrieved 5/6 statements.
# Partially parsed test_requests_get_with_data. Retrieved 9/10 statements.
# Partially parsed test_requests_post_with_data. Retrieved 9/10 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_requests_with_session. Retrieved 7/8 statements.
# Partially parsed test_requests_with_timeout. Retrieved 7/8 statements.
# Partially parsed test_requests_with_allowed_args. Retrieved 9/10 statements.


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
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'headers'
    var_3 = 'get'
    var_4 = 'User-Agent'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data_converts_to_urlencoded. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_converts_to_urlencoded. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data_converts_to_urlencoded. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_appends_data_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_existing_query_string. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_url_ending_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_url_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data_returns_original_url_and_none. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data_encodes_to_utf8. Retrieved 5/6 statements.
# Partially parsed test_query_with_case_insensitive_get_method. Retrieved 7/8 statements.


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
    var_0 = 'http://example.com?'
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
    var_2 = 'data'
    var_3 = 'test string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #13
#--------------------------




import builtins as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 'url'
    var_4 = 'reason'
    var_5 = 'headers'
    var_6 = 'text'
    var_7 = 404
    var_8 = 'test'
    var_9 = 'Not Found'
    var_10 = {}
    var_11 = ''
    var_12 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = 'timeout'
    var_18 = 'encoding'
    var_19 = 10
    var_20 = 'utf-8'
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = 'http://example.com'
    var_23 = module_1._requests(var_22, var_21)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param1=value1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_0[-1]
    var_8 = bool(var_0[-1] in ('?', '&'))
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_data_is_encoded_when_not_none_and_not_get_method. Retrieved 7/8 statements.


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'test data'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'
    var_6 = module_0.encode(var_5)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_query_with_data_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_get_with_data_no_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_ending_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_data_ending_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_data. Retrieved 5/6 statements.
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
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing='
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

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_with_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_and_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_uppercase_get_method. Retrieved 7/8 statements.


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
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 123
    var_4 = {var_2: var_3}

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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_appends_data_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_lowercase_method. Retrieved 7/8 statements.
# Partially parsed test_query_mixed_case_method. Retrieved 7/8 statements.
# Partially parsed test_query_string_data. Retrieved 5/6 statements.


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
    var_0 = 'http://example.com?'
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

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GeT'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'plain string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param1=value1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_0[-1]
    var_8 = bool(var_0[-1] in ('?', '&'))
    assert var_8 is True



# Parsed testcases at query #23
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
    var_7 = 404
    var_8 = 'http://example.com'
    var_9 = 'Not Found'
    var_10 = {}
    var_11 = ''
    var_12 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = bool(not 200 <= var_16.status_code < 300)
    assert var_17 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_data_is_encoded_when_present. Retrieved 7/8 statements.


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'test data'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'
    var_6 = module_0.encode(var_5)



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_query_with_data_as_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_as_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_as_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_url_has_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_url_ends_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_url_ends_with_question. Retrieved 7/8 statements.
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
    var_1 = 'post'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_appends_data_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.
# Partially parsed test_query_case_insensitive_method. Retrieved 7/8 statements.
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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'raw data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 6/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)
    var_4 = 'get'
    var_5 = None



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__requests_get_without_session. Retrieved 9/10 statements.
# Partially parsed test__requests_post_without_session. Retrieved 13/14 statements.
# Partially parsed test__requests_get_with_session. Retrieved 11/12 statements.
# Partially parsed test__requests_post_with_session. Retrieved 15/16 statements.
# Partially parsed test__requests_with_encoding. Retrieved 9/10 statements.
# Partially parsed test__requests_without_encoding. Retrieved 7/8 statements.
# Partially parsed test__requests_with_timeout. Retrieved 7/8 statements.
# Partially parsed test__requests_with_default_timeout. Retrieved 5/6 statements.


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

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'method'
    var_3 = 'data'
    var_4 = 'session'
    var_5 = 'encoding'
    var_6 = 'timeout'
    var_7 = 'post'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'utf-8'
    var_12 = 10
    var_13 = {var_2: var_7, var_3: var_10, var_4: var_1, var_5: var_11, var_6: var_12}
    var_14 = module_1._requests(var_0, var_13)

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
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
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
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_query_with_data_in_kwargs. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data. Retrieved 8/9 statements.
# Partially parsed test_query_get_method_appends_data_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_appends_data_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_string_method_lowercase. Retrieved 7/8 statements.
# Partially parsed test_query_with_string_method_uppercase. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_string_method. Retrieved 8/9 statements.


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
    var_0 = 'http://example.com?'
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
    var_0 = 'http://example.com'
    var_1 = 'get'
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
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = [var_1]
    var_3 = 'data'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com?param1=value1'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = var_0[-1]
    var_8 = bool(var_0[-1] in ('?', '&'))
    assert var_8 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_query_with_data_dict. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_tuple. Retrieved 8/9 statements.
# Partially parsed test_query_with_data_string. Retrieved 5/6 statements.
# Partially parsed test_query_with_get_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_data_with_existing_query_no_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_post_method_and_data. Retrieved 7/8 statements.
# Partially parsed test_query_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_case_insensitive_get_method. Retrieved 7/8 statements.


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
    var_3 = 'key=value'
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
    var_0 = 'http://example.com'
    var_1 = 'post'
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



# Parsed testcases at query #35
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 300
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = bool(not 200 <= var_7.status_code < 300)
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 9/15 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'http://example.com?param1=value1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'param2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'get'
    var_8 = {}
    var_9 = module_0.get(var_2, **var_8)



# Parsed testcases at query #37
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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__requests_get_without_session. Retrieved 6/7 statements.
# Partially parsed test__requests_get_with_session. Retrieved 8/9 statements.
# Partially parsed test__requests_post_without_session. Retrieved 10/11 statements.
# Partially parsed test__requests_post_with_session. Retrieved 12/13 statements.
# Partially parsed test__requests_with_encoding. Retrieved 8/9 statements.
# Partially parsed test__requests_with_timeout. Retrieved 8/9 statements.
# Partially parsed test__requests_with_custom_headers. Retrieved 10/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

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
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/post'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://httpbin.org/post'
    var_2 = 'method'
    var_3 = 'data'
    var_4 = 'session'
    var_5 = 'post'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8, var_4: var_0}
    var_10 = module_1._requests(var_1, var_9)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://invalid.url'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/headers'
    var_1 = 'method'
    var_2 = 'headers'
    var_3 = 'get'
    var_4 = 'User-Agent'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



