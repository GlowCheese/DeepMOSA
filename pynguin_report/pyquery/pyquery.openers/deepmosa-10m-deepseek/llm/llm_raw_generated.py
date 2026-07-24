####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": [("key", "value")]})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": (("key", "value"),)})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_string_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_dict_data_and_post_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_and_url_with_existing_query():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_data_and_url_ending_with_question_mark():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_and_url_ending_with_ampersand():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_without_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_and_get_method_lowercase():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_and_get_method_mixed_case():
    url, data = _query("http://example.com", "Get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_none_data():
    url, data = _query("http://example.com", "GET", {"data": None})
    assert url == "http://example.com"
    assert data is None
```


# LLM-generated content at query #2
#--------------------------

def test_urllib_get_with_data_appends_to_url():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}, "timeout": 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == "http://example.com?key=value"
    assert result_data is None

def test_urllib_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == "http://example.com"
    assert result_data is None

def test_urllib_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "timeout": 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == "http://example.com"
    assert result_data == b"key=value"

def test_urllib_get_with_existing_query_string():
    url = "http://example.com?existing=param"
    kwargs = {"method": "get", "data": {"key": "value"}, "timeout": 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == "http://example.com?existing=param&key=value"
    assert result_data is None


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_5_true():
    data = {'key': 'value'}
    result = type(data) in (dict, list, tuple)
    assert result == True


# LLM-generated content at query #4
#--------------------------

```
def test_query_with_no_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_get_method_and_dict_data():
    url, data = _query("http://example.com", "GET", {'data': {'key': 'value'}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_get_method_and_list_data():
    url, data = _query("http://example.com", "GET", {'data': [('key', 'value')]})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_get_method_and_tuple_data():
    url, data = _query("http://example.com", "GET", {'data': (('key', 'value'),)})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_get_method_and_existing_question_mark():
    url, data = _query("http://example.com?", "GET", {'data': {'key': 'value'}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_get_method_and_existing_question_mark_and_ampersand():
    url, data = _query("http://example.com?a=1&", "GET", {'data': {'key': 'value'}})
    assert url == "http://example.com?a=1&key=value"
    assert data is None

def test_query_with_post_method_and_dict_data():
    url, data = _query("http://example.com", "POST", {'data': {'key': 'value'}})
    assert url == "http://example.com"
    assert data == b'key=value'

def test_query_with_non_string_method():
    url, data = _query("http://example.com", 123, {'data': {'key': 'value'}})
    assert url == "http://example.com"
    assert data == b'key=value'

def test_query_with_data_not_in_kwargs():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None
```


# LLM-generated content at query #5
#--------------------------

```
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_query_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_existing_query():
    url = "http://example.com?existing=1"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {"method": "get", "session": session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_raise_http_error():
    import pytest
    url = "http://httpbin.org/status/404"
    kwargs = {"method": "get"}
    with pytest.raises(Exception):
        _requests(url, kwargs)
```


# LLM-generated content at query #6
#--------------------------

def test_query_with_dict_data_for_get_method():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_for_get_method():
    url, data = _query("http://example.com", "get", {"data": [("key", "value")]})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_tuple_data_for_get_method():
    url, data = _query("http://example.com", "get", {"data": (("key", "value"),)})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_for_post_method():
    url, data = _query("http://example.com", "post", {"data": "test"})
    assert url == "http://example.com"
    assert data == b"test"

def test_query_with_data_for_get_method_with_existing_question_mark():
    url, data = _query("http://example.com?existing=1", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_data_for_get_method_with_existing_ampersand():
    url, data = _query("http://example.com?existing=1&", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_without_data():
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_and_no_method():
    url, data = _query("http://example.com", None, {"data": "test"})
    assert url == "http://example.com"
    assert data == b"test"

def test_query_with_empty_string_data_for_get():
    url, data = _query("http://example.com", "get", {"data": ""})
    assert url == "http://example.com"
    assert data is None


# LLM-generated content at query #7
#--------------------------

def test_query_get_with_data_as_dict():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_get_with_data_as_list():
    url, data = _query("http://example.com", "get", {"data": ["a", "b"]})
    assert url == "http://example.com?a=b"
    assert data is None

def test_query_get_with_data_as_tuple():
    url, data = _query("http://example.com", "get", {"data": ("x", "y")})
    assert url == "http://example.com?x=y"
    assert data is None

def test_query_get_with_existing_question_mark():
    url, data = _query("http://example.com?existing=1", "get", {"data": {"new": "param"}})
    assert url == "http://example.com?existing=1&new=param"
    assert data is None

def test_query_get_with_existing_ampersand():
    url, data = _query("http://example.com?a=1&", "get", {"data": {"b": "2"}})
    assert url == "http://example.com?a=1&b=2"
    assert data is None

def test_query_get_without_data():
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None

def test_query_post_with_data():
    url, data = _query("http://example.com", "post", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_post_with_string_data():
    url, data = _query("http://example.com", "post", {"data": "raw_string"})
    assert url == "http://example.com"
    assert data == b"raw_string"

def test_query_get_with_lowercase_get():
    url, data = _query("http://example.com", "GET", {"data": {"p": "q"}})
    assert url == "http://example.com?p=q"
    assert data is None

def test_query_get_with_uppercase_get():
    url, data = _query("http://example.com", "GET", {"data": {"p": "q"}})
    assert url == "http://example.com?p=q"
    assert data is None


# LLM-generated content at query #8
#--------------------------

def test_url_opener_with_requests_get():
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'session': None}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_post():
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_session():
    import requests
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_timeout():
    kwargs = {'method': 'get', 'timeout': 10}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_encoding():
    kwargs = {'method': 'get', 'encoding': 'gbk'}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_get_with_data():
    kwargs = {'method': 'get', 'data': {'q': 'test'}}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_17_evaluates_to_true():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": {"key": "value"}}
    from urllib.parse import urlencode
    data = kwargs.pop("data")
    data = urlencode(data)
    if data:
        data = data.encode("utf-8")


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_false():
    kwargs = {'method': 'get'}
    url = 'http://example.com'
    encoding = kwargs.get('encoding')
    method = kwargs.get('method', 'get').lower()
    session = kwargs.get('session')
    if session:
        meth = getattr(session, str(method))
    else:
        meth = getattr(requests, str(method))
    if method == 'get':
        url, data = _query(url, method, kwargs)
    kw = {}
    for k in allowed_args:
        if k in kwargs:
            kw[k] = kwargs[k]
    resp = meth(url=url, timeout=kwargs.get('timeout', DEFAULT_TIMEOUT), **kw)
    assert 200 <= resp.status_code < 300
```


# LLM-generated content at query #11
#--------------------------

def test_predicate_false_when_method_not_basestring():
    url = "http://example.com"
    method = 123
    kwargs = {"data": "test"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "test"


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line8_evaluates_to_false():
    url = "http://example.com"
    method = "GET"
    kwargs = {"data": None}
    result = _query(url, method, kwargs)


# LLM-generated content at query #13
#--------------------------

def test_urllib_get_with_data_appends_to_url():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}, "timeout": 30}
    result_url, result_data = _urllib(url, kwargs)
    assert "?key=value" in result_url

def test_urllib_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 30}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == url

def test_urllib_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "timeout": 30}
    result_url, result_data = _urllib(url, kwargs)
    assert result_data is not None

def test_urllib_default_timeout():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == url


# LLM-generated content at query #14
#--------------------------

```
def test_query_with_no_data():
    url, data = _query("http://example.com", "GET", {})

def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": [("key", "value")]})

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": (("key", "value"),)})

def test_query_with_get_method_and_no_question_mark():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})

def test_query_with_get_method_and_question_mark_at_end():
    url, data = _query("http://example.com?", "GET", {"data": "key=value"})

def test_query_with_get_method_and_question_mark_with_data():
    url, data = _query("http://example.com?existing=param", "GET", {"data": "key=value"})

def test_query_with_get_method_and_ampersand_at_end():
    url, data = _query("http://example.com?existing=param&", "GET", {"data": "key=value"})

def test_query_with_post_method_and_data():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})

def test_query_with_non_string_method_and_data():
    url, data = _query("http://example.com", 123, {"data": "key=value"})

def test_query_with_uppercase_get_method():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})

def test_query_with_data_already_encoded():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})

def test_query_with_data_none():
    url, data = _query("http://example.com", "GET", {"data": None})
```


# LLM-generated content at query #15
#--------------------------

```
def test_url_opener_has_request_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result is not None
```


# LLM-generated content at query #16
#--------------------------

def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?a=b"
    assert data is None

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?x=y"
    assert data is None

def test_query_with_string_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_and_post_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_and_get_method_with_existing_question_mark():
    url, data = _query("http://example.com?existing=param", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=param&key=value"
    assert data is None

def test_query_with_data_and_get_method_with_existing_question_mark_and_ampersand():
    url, data = _query("http://example.com?existing=param&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=param&key=value"
    assert data is None

def test_query_with_data_and_get_method_without_existing_query():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_without_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_other_types():
    url, data = _query("http://example.com", "GET", {"data": 123})
    assert url == "http://example.com"
    assert data == b"123"


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_9_is_false():
    url = "http://example.com"
    method = "GET"
    kwargs = {}
    _query(url, method, kwargs)


# LLM-generated content at query #18
#--------------------------

def test_query_get_with_dict_data():
    url = "http://example.com/api"
    method = "get"
    kwargs = {"data": {"key1": "value1", "key2": "value2"}}
    result_url, result_data = _query(url, method, kwargs)
    assert "?" in result_url
    assert "key1=value1" in result_url
    assert "key2=value2" in result_url
    assert result_data is None

def test_query_post_with_dict_data():
    url = "http://example.com/api"
    method = "post"
    kwargs = {"data": {"key1": "value1"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == url
    assert result_data == b"key1=value1"

def test_query_get_with_string_data():
    url = "http://example.com/api"
    method = "get"
    kwargs = {"data": "key1=value1&key2=value2"}
    result_url, result_data = _query(url, method, kwargs)
    assert "?" in result_url
    assert "key1=value1" in result_url
    assert "key2=value2" in result_url
    assert result_data is None

def test_query_get_with_data_and_existing_query():
    url = "http://example.com/api?existing=param"
    method = "get"
    kwargs = {"data": {"new": "data"}}
    result_url, result_data = _query(url, method, kwargs)
    assert "existing=param" in result_url
    assert "new=data" in result_url
    assert "&" in result_url
    assert result_data is None

def test_query_get_with_data_and_url_ending_with_question_mark():
    url = "http://example.com/api?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com/api?key=value"
    assert result_data is None

def test_query_get_with_data_and_url_ending_with_ampersand():
    url = "http://example.com/api?existing&"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com/api?existing&key=value"
    assert result_data is None

def test_query_with_no_data():
    url = "http://example.com/api"
    method = "get"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == url
    assert result_data is None

def test_query_with_list_data():
    url = "http://example.com/api"
    method = "post"
    kwargs = {"data": ["a", "b", "c"]}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == b"a&b&c"

def test_query_with_tuple_data():
    url = "http://example.com/api"
    method = "post"
    kwargs = {"data": ("x", "y")}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == b"x&y"

def test_query_get_with_data_and_no_question_mark():
    url = "http://example.com/api"
    method = "GET"
    kwargs = {"data": {"param": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert "?" in result_url
    assert result_url == "http://example.com/api?param=value"
    assert result_data is None

def test_query_post_with_no_data():
    url = "http://example.com/api"
    method = "post"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == url
    assert result_data is None


# LLM-generated content at query #19
#--------------------------

```python
def test_requests_with_get_method_and_session():
    url = "http://example.com"
    kwargs = {"method": "get", "session": mock_session, "encoding": "utf-8"}
    mock_session.get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "success"
    result = _requests(url, kwargs)
    assert result == "success"

def test_requests_with_get_method_no_session():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    mock_get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "success"
    result = _requests(url, kwargs)
    assert result == "success"

def test_requests_with_post_method():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    mock_post.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "posted"
    result = _requests(url, kwargs)
    assert result == "posted"

def test_requests_http_error():
    url = "http://example.com"
    kwargs = {"method": "get"}
    mock_get.return_value = mock_response
    mock_response.status_code = 404
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        pass

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    mock_get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "timeout_test"
    result = _requests(url, kwargs)
    assert result == "timeout_test"

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "latin-1"}
    mock_get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "encoded"
    result = _requests(url, kwargs)
    assert result == "encoded"
```


# LLM-generated content at query #20
#--------------------------

def test_query_with_dict_data_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_get_method():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?a=b"
    assert data is None

def test_query_with_tuple_data_get_method():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?x=y"
    assert data is None

def test_query_with_data_get_method_url_has_question_mark():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_data_get_method_url_ends_with_question_mark():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_get_method_url_ends_with_ampersand():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_data_get_method_no_question_mark_url():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_non_get_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_non_get_method_list():
    url, data = _query("http://example.com", "POST", {"data": ["a", "b"]})
    assert url == "http://example.com"
    assert data == b"a=b"

def test_query_with_data_non_get_method_tuple():
    url, data = _query("http://example.com", "POST", {"data": ("x", "y")})
    assert url == "http://example.com"
    assert data == b"x=y"

def test_query_without_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_get_method_case_insensitive():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None


# LLM-generated content at query #21
#--------------------------

def test_urllib_get_with_data():
    url = "http://example.com/api"
    kwargs = {'method': 'GET', 'data': {'key': 'value'}, 'timeout': 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url.startswith("http://example.com/api?key=value")

def test_urllib_post_with_data():
    url = "http://example.com/api"
    kwargs = {'method': 'POST', 'data': {'key': 'value'}, 'timeout': 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == url
    assert result_data is not None

def test_urllib_without_data():
    url = "http://example.com/api"
    kwargs = {'method': 'GET', 'timeout': 10}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url == url
    assert result_data is None

def test_urllib_default_timeout():
    url = "http://example.com/api"
    kwargs = {'method': 'GET', 'data': {'key': 'value'}}
    result_url, result_data = _urllib(url, kwargs)
    assert result_url.startswith("http://example.com/api?key=value")


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com/test?"
    method = "GET"
    kwargs = {"data": "param=value"}
    data = kwargs.pop('data')
    result_url, result_data = _query(url, method, kwargs)


# LLM-generated content at query #23
#--------------------------

def test_predicate_evaluates_to_false():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.url = "http://example.com"
    mock_response.reason = "OK"
    mock_response.headers = {}
    mock_response.text = "<html></html>"
    kwargs = {'timeout': 10}
    meth = lambda url, timeout, **kw: mock_response
    resp = meth(url="http://example.com", timeout=kwargs.get('timeout', 30))
    assert not (200 <= resp.status_code < 300) == False


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_8_evaluates_to_false():
    url = "http://example.com"
    method = "POST"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)


# LLM-generated content at query #25
#--------------------------

```
def test_query_with_no_data():
    result = _query("http://example.com", "GET", {})

def test_query_with_dict_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": {"key": "value"}})

def test_query_with_list_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": [("key", "value")]})

def test_query_with_tuple_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": (("key", "value"),)})

def test_query_with_data_and_post_method():
    result = _query("http://example.com", "POST", {"data": "rawdata"})

def test_query_with_get_method_and_url_without_query():
    result = _query("http://example.com", "get", {"data": {"a": "1"}})

def test_query_with_get_method_and_url_with_question_mark():
    result = _query("http://example.com?", "GET", {"data": {"b": "2"}})

def test_query_with_get_method_and_url_with_query():
    result = _query("http://example.com?a=1", "GET", {"data": {"b": "2"}})

def test_query_with_get_method_and_url_ending_with_ampersand():
    result = _query("http://example.com?a=1&", "GET", {"data": {"b": "2"}})

def test_query_with_data_as_bytes():
    result = _query("http://example.com", "POST", {"data": b"raw"})

def test_query_with_empty_data_dict():
    result = _query("http://example.com", "GET", {"data": {}})

def test_query_with_empty_data_string():
    result = _query("http://example.com", "POST", {"data": ""})
```


# LLM-generated content at query #26
#--------------------------

```python
def test_requests_with_get_method_and_no_session():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_post_method():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 5}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_raises_http_error():
    import requests
    from requests.exceptions import HTTPError
    url = "http://httpbin.org/status/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        assert True
```


# LLM-generated content at query #27
#--------------------------

def test_url_opener_requests_get_no_data():
    url_opener("http://example.com", {"method": "get", "session": None})

def test_url_opener_requests_get_with_data():
    url_opener("http://example.com", {"method": "get", "data": {"key": "value"}})

def test_url_opener_requests_post_with_data():
    url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})

def test_url_opener_requests_with_encoding():
    url_opener("http://example.com", {"encoding": "utf-8", "method": "get"})

def test_url_opener_requests_with_timeout():
    url_opener("http://example.com", {"method": "get", "timeout": 10})


# LLM-generated content at query #28
#--------------------------

```
def test_urllib_get_with_data_appends_query_string():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'timeout': 10}
    result_url, result_data = _query(url, 'get', {'data': {'key': 'value'}})
    expected_url = "http://example.com?key=value"
    assert result_url == expected_url
    assert result_data is None

def test_urllib_get_with_data_and_existing_query():
    url = "http://example.com?existing=1"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result_url, result_data = _query(url, 'get', {'data': {'key': 'value'}})
    expected_url = "http://example.com?existing=1&key=value"
    assert result_url == expected_url
    assert result_data is None

def test_urllib_get_with_data_and_existing_question_mark():
    url = "http://example.com?"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result_url, result_data = _query(url, 'get', {'data': {'key': 'value'}})
    expected_url = "http://example.com?key=value"
    assert result_url == expected_url
    assert result_data is None

def test_urllib_get_with_data_and_existing_ampersand():
    url = "http://example.com?existing=1&"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result_url, result_data = _query(url, 'get', {'data': {'key': 'value'}})
    expected_url = "http://example.com?existing=1&key=value"
    assert result_url == expected_url
    assert result_data is None

def test_urllib_post_with_data_encodes_utf8():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result_url, result_data = _query(url, 'post', {'data': {'key': 'value'}})
    expected_url = "http://example.com"
    expected_data = b'key=value'
    assert result_url == expected_url
    assert result_data == expected_data

def test_urllib_get_without_data_returns_original_url():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 5}
    result_url, result_data = _query(url, 'get', {})
    assert result_url == url
    assert result_data is None

def test_urllib_post_without_data_returns_none_data():
    url = "http://example.com"
    kwargs = {'method': 'post', 'timeout': 5}
    result_url, result_data = _query(url, 'post', {})
    assert result_url == url
    assert result_data is None

def test_urllib_with_list_data_encodes_properly():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': ['a', 'b']}
    result_url, result_data = _query(url, 'post', {'data': ['a', 'b']})
    expected_data = b'a&b'
    assert result_url == url
    assert result_data == expected_data

def test_urllib_with_tuple_data_encodes_properly():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': ('x', 'y')}
    result_url, result_data = _query(url, 'post', {'data': ('x', 'y')})
    expected_data = b'x&y'
    assert result_url == url
    assert result_data == expected_data

def test_urllib_get_with_data_removes_data_from_kwargs():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    original_kwargs = {'data': {'key': 'value'}}
    _query(url, 'get', original_kwargs)
    assert 'data' not in original_kwargs
```


# LLM-generated content at query #29
#--------------------------

def test_url_opener_has_request_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result is None


# LLM-generated content at query #30
#--------------------------

```
def test_has_request_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result is not None
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_requests_get_with_session():
    url = "http://example.com"
    kwargs = {"method": "get", "session": mock.Mock(), "encoding": "utf-8"}
    mock_session = kwargs["session"]
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "response text"
    mock_session.get.return_value = mock_response
    result = _requests(url, kwargs)
    assert result == "response text"
    mock_session.get.assert_called_once_with(url=url, timeout=10)

def test_requests_get_without_session():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "response text"
    with mock.patch("requests.get", return_value=mock_response) as mock_get:
        result = _requests(url, kwargs)
        assert result == "response text"
        mock_get.assert_called_once_with(url=url, timeout=10)

def test_requests_http_error():
    url = "http://example.com"
    kwargs = {"method": "get"}
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    mock_response.headers = {}
    with mock.patch("requests.get", return_value=mock_response):
        try:
            _requests(url, kwargs)
            assert False
        except HTTPError as e:
            assert e.status_code == 404
            assert e.reason == "Not Found"

def test_requests_with_custom_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 30}
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "ok"
    with mock.patch("requests.get", return_value=mock_response) as mock_get:
        result = _requests(url, kwargs)
        assert result == "ok"
        mock_get.assert_called_once_with(url=url, timeout=30)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "latin-1"}
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "text"
    with mock.patch("requests.get", return_value=mock_response) as mock_get:
        result = _requests(url, kwargs)
        assert result == "text"
        mock_response.encoding == "latin-1"

def test_requests_get_with_data_conversion():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "ok"
    with mock.patch("requests.get", return_value=mock_response) as mock_get:
        _requests(url, kwargs)
        mock_get.assert_called_once()
        call_url = mock_get.call_args[1]["url"]
        assert "key=value" in call_url
```


# LLM-generated content at query #2
#--------------------------

```python
def test_status_code_in_success_range():
    kwargs = {'method': 'get'}
    url = "http://example.com"
    resp = type('Response', (), {'status_code': 200, 'url': url, 'reason': 'OK', 'headers': {}, 'text': ''})()
    meth = lambda url, timeout=5, **kw: resp
    _requests(url, kwargs)  # Should not raise HTTPError
```


# LLM-generated content at query #3
#--------------------------

def test_urllib_get_method_with_data():
    url = "http://example.com"
    kwargs = {"method": "GET", "data": {"key": "value"}, "timeout": 5}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_post_method_with_data():
    url = "http://example.com"
    kwargs = {"method": "POST", "data": {"key": "value"}, "timeout": 5}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_method_without_data():
    url = "http://example.com"
    kwargs = {"method": "GET", "timeout": 5}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_default_timeout():
    url = "http://example.com"
    kwargs = {"method": "GET", "data": "test"}
    result = _urllib(url, kwargs)
    assert result is not None


# LLM-generated content at query #4
#--------------------------

def test_method_not_get_evaluates_false():
    kwargs = {'method': 'post', 'encoding': 'utf-8'}
    url = 'http://example.com'
    method = kwargs.get('method', 'get').lower()
    assert not (method == 'get')


# LLM-generated content at query #5
#--------------------------

```
def test_requests_get_no_session():
    url = "http://example.com"
    kwargs = {"method": "get"}
    html = _requests(url, kwargs)
    assert isinstance(html, str)

def test_requests_get_with_session():
    class FakeSession:
        def get(self, url, timeout=10):
            class FakeResponse:
                status_code = 200
                text = "response text"
            return FakeResponse()
    session = FakeSession()
    url = "http://example.com"
    kwargs = {"method": "get", "session": session}
    html = _requests(url, kwargs)
    assert html == "response text"

def test_requests_get_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    html = _requests(url, kwargs)
    assert isinstance(html, str)

def test_requests_get_http_error():
    class FakeSession:
        def get(self, url, timeout=10):
            class FakeResponse:
                status_code = 404
                reason = "Not Found"
                headers = {}
            return FakeResponse()
    session = FakeSession()
    url = "http://example.com"
    kwargs = {"method": "get", "session": session}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        pass

def test_requests_post():
    class FakeSession:
        def post(self, url, timeout=10, data=None):
            class FakeResponse:
                status_code = 200
                text = "post response"
            return FakeResponse()
    session = FakeSession()
    url = "http://example.com"
    kwargs = {"method": "post", "session": session}
    html = _requests(url, kwargs)
    assert html == "post response"
```


# LLM-generated content at query #6
#--------------------------

def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?0=a&1=b"
    assert data is None

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?0=x&1=y"
    assert data is None

def test_query_with_string_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": "key=value"})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_no_data_and_get_method():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_and_post_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_empty_dict_data():
    url, data = _query("http://example.com", "GET", {"data": {}})
    assert url == "http://example.com?"
    assert data is None

def test_query_with_url_containing_question_mark():
    url, data = _query("http://example.com?existing=param", "GET", {"data": {"new": "value"}})
    assert url == "http://example.com?existing=param&new=value"
    assert data is None

def test_query_with_url_ending_with_ampersand():
    url, data = _query("http://example.com?existing=param&", "GET", {"data": {"new": "value"}})
    assert url == "http://example.com?existing=param&new=value"
    assert data is None

def test_query_with_non_dict_list_tuple_data():
    url, data = _query("http://example.com", "GET", {"data": 12345})
    assert url == "http://example.com"
    assert data == b"12345"


# LLM-generated content at query #7
#--------------------------

```
def test_query_with_no_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_dict_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_list_get_method():
    url, data = _query("http://example.com", "GET", {"data": [("key", "value")]})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_tuple_get_method():
    url, data = _query("http://example.com", "GET", {"data": (("key", "value"),)})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_existing_query_string_get_method():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_trailing_question_mark_get_method():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_trailing_ampersand_get_method():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_data_post_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_lowercase_get_method():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_uppercase_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_data_not_dict_list_tuple_get_method():
    url, data = _query("http://example.com", "GET", {"data": "stringdata"})
    assert url == "http://example.com"
    assert data == b"stringdata"

def test_query_with_data_not_dict_list_tuple_post_method():
    url, data = _query("http://example.com", "POST", {"data": "stringdata"})
    assert url == "http://example.com"
    assert data == b"stringdata"

def test_query_with_data_empty_dict_get_method():
    url, data = _query("http://example.com", "GET", {"data": {}})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_empty_list_get_method():
    url, data = _query("http://example.com", "GET", {"data": []})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_empty_tuple_get_method():
    url, data = _query("http://example.com", "GET", {"data": ()})
    assert url == "http://example.com"
    assert data is None
```


# LLM-generated content at query #8
#--------------------------

```
def test_requests_get_with_data_and_session():
    url = "http://example.com"
    kwargs = {'method': 'get', 'session': mock_session, 'data': {'key': 'value'}, 'encoding': 'utf-8', 'timeout': 10}
    mock_session.get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "response text"
    result = _requests(url, kwargs)
    assert result == "response text"

def test_requests_get_without_session():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'encoding': 'utf-8', 'timeout': 10}
    requests.get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "response text"
    result = _requests(url, kwargs)
    assert result == "response text"

def test_requests_raises_http_error():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'encoding': 'utf-8', 'timeout': 10}
    requests.get.return_value = mock_response
    mock_response.status_code = 404
    mock_response.url = url
    mock_response.reason = "Not Found"
    mock_response.headers = {}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError as e:
        assert e.url == url
        assert e.status_code == 404
        assert e.reason == "Not Found"

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'latin-1', 'timeout': 10}
    requests.get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.text = "response text"
    result = _requests(url, kwargs)
    assert mock_response.encoding == 'latin-1'
    assert result == "response text"
```


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com/?existing=param"
    method = "get"
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)


# LLM-generated content at query #10
#--------------------------

def test_predicate_false_when_method_not_string():
    data = "some_data"
    url = "http://example.com"
    result_url, result_data = _query(url, 123, {'data': data})
    assert result_url == url
    assert result_data == data


# LLM-generated content at query #11
#--------------------------

def test_url_opener_requests_get():
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'session': None}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_requests_get_with_data():
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'encoding': 'utf-8'}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_requests_post():
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'encoding': 'utf-8'}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_requests_invalid_status():
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    try:
        url_opener('http://httpbin.org/status/404', kwargs)
        assert False
    except HTTPError:
        assert True

def test_url_opener_urllib_get():
    kwargs = {'method': 'get'}
    result = url_opener('http://example.com', kwargs)
    assert result is not None

def test_url_opener_urllib_post():
    kwargs = {'method': 'post', 'data': 'key=value'}
    result = url_opener('http://example.com', kwargs)
    assert result is not None

def test_url_opener_urllib_get_with_data():
    kwargs = {'method': 'get', 'data': 'key=value'}
    result = url_opener('http://example.com', kwargs)
    assert result is not None


# LLM-generated content at query #12
#--------------------------

def test_query_get_with_dict_data():
    result = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_get_with_list_data():
    result = _query("http://example.com", "get", {"data": ["a", "b"]})
    assert result == ("http://example.com?a=b", None)

def test_query_get_with_tuple_data():
    result = _query("http://example.com", "get", {"data": ("x", "y")})
    assert result == ("http://example.com?x=y", None)

def test_query_get_with_existing_question_mark():
    result = _query("http://example.com?foo=bar", "get", {"data": {"key": "value"}})
    assert result == ("http://example.com?foo=bar&key=value", None)

def test_query_get_with_trailing_ampersand():
    result = _query("http://example.com?foo=bar&", "get", {"data": {"key": "value"}})
    assert result == ("http://example.com?foo=bar&key=value", None)

def test_query_get_without_question_mark():
    result = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_get_with_no_data():
    result = _query("http://example.com", "get", {})
    assert result == ("http://example.com", None)

def test_query_post_with_dict_data():
    result = _query("http://example.com", "post", {"data": {"key": "value"}})
    assert result == ("http://example.com", b"key=value")

def test_query_post_with_list_data():
    result = _query("http://example.com", "post", {"data": ["a", "b"]})
    assert result == ("http://example.com", b"a=b")

def test_query_post_with_tuple_data():
    result = _query("http://example.com", "post", {"data": ("x", "y")})
    assert result == ("http://example.com", b"x=y")

def test_query_post_with_no_data():
    result = _query("http://example.com", "post", {})
    assert result == ("http://example.com", None)

def test_query_get_with_uppercase_method():
    result = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_post_with_uppercase_method():
    result = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert result == ("http://example.com", b"key=value")

def test_query_get_with_existing_data_as_parameter():
    result = _query("http://example.com", "get", {"data": {"a": 1, "b": 2}})
    assert result == ("http://example.com?a=1&b=2", None)


# LLM-generated content at query #13
#--------------------------

def test_url_opener_requests_get():
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_urllib_get():
    kwargs = {'method': 'get'}
    result = url_opener('http://example.com', kwargs)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

def test_status_code_is_successful():
    resp = type('Response', (), {'status_code': 200, 'url': '', 'reason': '', 'headers': {}, 'text': '', 'encoding': None})()
    assert 200 <= resp.status_code < 300


# LLM-generated content at query #15
#--------------------------

def test_predicate_false_when_method_is_not_basestring():
    url = "http://example.com"
    method = 123
    kwargs = {"data": "some_data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "some_data"

def test_predicate_false_when_method_lower_is_not_get():
    url = "http://example.com"
    method = "POST"
    kwargs = {"data": "some_data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "some_data"

def test_predicate_false_when_data_is_none():
    url = "http://example.com"
    method = "get"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data is None


# LLM-generated content at query #16
#--------------------------

```python
def test_status_code_in_success_range():
    resp = MockResponse(200)
    meth = lambda url, timeout, **kw: resp
    kwargs = {'method': 'get', 'timeout': 5}
    url = "http://example.com"
    allowed_args = []
    _ = meth(url=url, timeout=kwargs.get('timeout', DEFAULT_TIMEOUT), **{})
    assert 200 <= resp.status_code < 300

class MockResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.url = "http://example.com"
        self.reason = "OK"
        self.headers = {}
        self.text = "success"
```


# LLM-generated content at query #17
#--------------------------

```
def test_url_opener_when_HAS_REQUEST_is_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})
```


# LLM-generated content at query #18
#--------------------------

```
def test_url_opener_when_has_request_is_false(self):
    url = "http://example.com"
    kwargs = {}
    original_HAS_REQUEST = HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener(url, kwargs)
    HAS_REQUEST = original_HAS_REQUEST
    assert result == _urllib(url, kwargs)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_status_code_in_success_range_does_not_raise():
    resp = MockResponse(status_code=200)
    meth = lambda url, timeout, **kw: resp
    url = "http://example.com"
    kwargs = {"method": "get"}
    allowed_args = []
    encoding = None
    method = "get"
    session = None
    resp = meth(url=url, timeout=5, **{})
    assert 200 <= resp.status_code < 300
```


# LLM-generated content at query #20
#--------------------------

```
def test_has_request_returns_false():
    result = url_opener("http://example.com", {})
    assert result is not None
```


# LLM-generated content at query #21
#--------------------------

```python
def test_requests_get_without_data():
    kwargs = {'method': 'get', 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_with_data():
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_post():
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_session():
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_encoding():
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_timeout():
    kwargs = {'method': 'get', 'timeout': 10, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert result is not None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "session": None}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_query_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}, "encoding": "utf-8", "session": None}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_without_data():
    url = "http://example.com"
    kwargs = {"method": "post", "encoding": "utf-8", "session": None}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "encoding": "utf-8", "session": None}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "session": session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 5, "encoding": "utf-8", "session": None}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_invalid_method():
    url = "http://example.com"
    kwargs = {"method": "invalid", "encoding": "utf-8", "session": None}
    try:
        _requests(url, kwargs)
        assert False
    except AttributeError:
        pass

def test_requests_http_error():
    url = "http://httpbin.org/status/404"
    kwargs = {"method": "get", "encoding": "utf-8", "session": None}
    try:
        _requests(url, kwargs)
        assert False
    except Exception:
        pass
```


# LLM-generated content at query #23
#--------------------------

def test_url_opener_with_requests_and_get_method():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_post_method():
    url = "http://example.com"
    kwargs = {"method": "post"}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 5}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_session():
    url = "http://example.com"
    kwargs = {"method": "get", "session": None}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #24
#--------------------------

def test_url_opener_with_requests_module():
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    result = url_opener("http://example.com", kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_module():
    kwargs = {"method": "get", "timeout": 10}
    result = url_opener("http://example.com", kwargs)
    assert hasattr(result, "read")


# LLM-generated content at query #25
#--------------------------

```python
def test_requests_get_with_session():
    url = "http://example.com"
    kwargs = {"method": "get", "session": MockSession()}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_without_session():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_with_allowed_args():
    url = "http://example.com"
    kwargs = {"method": "get", "headers": {"User-Agent": "test"}}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_get_raises_http_error():
    url = "http://example.com/error"
    kwargs = {"method": "get"}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        assert True

def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert result is not None
```


# LLM-generated content at query #26
#--------------------------

def test_url_opener_requests_get_no_data():
    kwargs = {'method': 'get'}
    result = url_opener("http://example.com", kwargs)
    assert result is not None

def test_url_opener_requests_get_with_data():
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = url_opener("http://example.com", kwargs)
    assert result is not None

def test_url_opener_requests_post():
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener("http://example.com", kwargs)
    assert result is not None

def test_url_opener_requests_with_session():
    import requests
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = url_opener("http://example.com", kwargs)
    assert result is not None

def test_url_opener_requests_with_encoding():
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = url_opener("http://example.com", kwargs)
    assert result is not None

def test_url_opener_requests_with_timeout():
    kwargs = {'method': 'get', 'timeout': 10}
    result = url_opener("http://example.com", kwargs)
    assert result is not None


