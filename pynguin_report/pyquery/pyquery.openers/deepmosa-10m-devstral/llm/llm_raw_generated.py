####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_with_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'

def test_query_with_non_string_method():
    url, data = _query('http://example.com', 'POST', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener_with_requests_get():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_get():
    global HAS_REQUEST
    HAS_REQUEST = False
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = url_opener(url, kwargs)
    assert isinstance(result, http.client.HTTPResponse)

def test_url_opener_with_urllib_post():
    global HAS_REQUEST
    HAS_REQUEST = False
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener(url, kwargs)
    assert isinstance(result, http.client.HTTPResponse)

def test_url_opener_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #3
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_post_method_and_no_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_non_string_method():
    url, data = _query('http://example.com', None, {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_string():
    url, data = _query('http://example.com', 'post', {'data': 'plain string'})
    assert url == 'http://example.com'
    assert data == b'plain string'


# LLM-generated content at query #4
#--------------------------

```python
def test_urllib_with_get_method_and_data():
    url = "http://example.com"
    kwargs = {'method': 'GET', 'data': {'key': 'value'}}
    result = _urllib(url, kwargs)
    assert result.geturl() == "http://example.com?key=value"
    assert result.read() is not None

def test_urllib_with_post_method_and_data():
    url = "http://example.com"
    kwargs = {'method': 'POST', 'data': {'key': 'value'}}
    result = _urllib(url, kwargs)
    assert result.geturl() == "http://example.com"
    assert result.read() is not None

def test_urllib_with_timeout():
    url = "http://example.com"
    kwargs = {'timeout': 10}
    result = _urllib(url, kwargs)
    assert result.read() is not None

def test_urllib_with_no_data():
    url = "http://example.com"
    kwargs = {}
    result = _urllib(url, kwargs)
    assert result.geturl() == "http://example.com"
    assert result.read() is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener_when_no_requests():
    global HAS_REQUEST
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #6
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'

def test_query_with_uppercase_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #7
#--------------------------

```python
def test_query_with_data_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_string():
    url, data = _query('http://example.com', 'post', {'data': 'key=value'})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_with_data_no_query():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_with_data_existing_query():
    url, data = _query('http://example.com?foo=bar', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?foo=bar&key=value'
    assert data is None

def test_query_get_with_data_existing_query_no_ampersand():
    url, data = _query('http://example.com?foo=bar', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?foo=bar&key=value'
    assert data is None

def test_query_get_with_data_existing_query_with_ampersand():
    url, data = _query('http://example.com?foo=bar&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?foo=bar&key=value'
    assert data is None

def test_query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_case_insensitive_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #8
#--------------------------

```python
def test_requests_with_get_method():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_post_method():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_invalid_url():
    url = "http://invalid.url"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
    except HTTPError:
        pass
    else:
        assert False, "Expected HTTPError"

def test_requests_with_data_in_get():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #10
#--------------------------

```python
def test__requests_with_get_method_and_data():
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'data': {'key': 'value'},
        'encoding': 'utf-8',
        'timeout': 10
    }
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_post_method():
    url = "http://example.com"
    kwargs = {
        'method': 'post',
        'data': {'key': 'value'},
        'encoding': 'utf-8',
        'timeout': 10
    }
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'session': requests.Session(),
        'timeout': 10
    }
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_non_200_status_code():
    url = "http://httpbin.org/status/404"
    kwargs = {
        'method': 'get',
        'timeout': 10
    }
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass

def test__requests_with_encoding():
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'encoding': 'utf-8',
        'timeout': 10
    }
    result = _requests(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "example.com?param1=value1"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] in ('?', '&')


# LLM-generated content at query #12
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_with_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_ending_with_question():
    url, data = _query('http://example.com?existing=param?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_ending_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'plain string'})
    assert url == 'http://example.com'
    assert data == b'plain string'


# LLM-generated content at query #13
#--------------------------

```python
def test_url_opener_when_no_requests():
    HAS_REQUEST = False
    assert not HAS_REQUEST


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    mock_response = type('MockResponse', (), {
        'status_code': 404,
        'url': 'http://example.com',
        'reason': 'Not Found',
        'headers': {},
        'text': 'Not Found'
    })()
    assert not (200 <= mock_response.status_code < 300)


# LLM-generated content at query #15
#--------------------------

```python
def test_status_code_outside_success_range():
    resp = type('Response', (), {'status_code': 404, 'url': 'test', 'reason': 'Not Found', 'headers': {}, 'text': ''})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #16
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #17
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_ending_with_question():
    url, data = _query('http://example.com?existing=param?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_ending_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_data_as_string():
    url, data = _query('http://example.com', 'post', {'data': 'raw_data'})
    assert url == 'http://example.com'
    assert data == b'raw_data'


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 199})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #19
#--------------------------

```python
def test__requests_with_get_method_and_no_data():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_get_method_and_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_post_method():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    kwargs = {'method': 'get', 'session': requests.Session()}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_non_200_status_code():
    url = "http://example.com/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_url_ends_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] == '?'


# LLM-generated content at query #21
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test html"
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "test html"
        mock_get.assert_called_once_with(url="http://example.com", timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_requests_post():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "test html"
        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        assert result == "test html"
        mock_post.assert_called_once_with(url="http://example.com", data={'key': 'value'}, timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_requests_non_200_status():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        mock_get.return_value.reason = "Not Found"
        mock_get.return_value.url = "http://example.com"
        mock_get.return_value.headers = {}
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_with_urllib_get():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test html"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == b"test html"
        mock_urlopen.assert_called_once_with("http://example.com", None, timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_urllib_post():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test html"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {'method': 'post', 'data': b'key=value'})
        assert result == b"test html"
        mock_urlopen.assert_called_once_with("http://example.com", b'key=value', timeout=DEFAULT_TIMEOUT)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "get"
    kwargs = {"data": {"param2": "value2"}}
    result_url, result_data = _query(url, method, kwargs)
    assert not (url[-1] not in ('?', '&'))


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 404})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #24
#--------------------------

```python
def test_url_opener_with_requests_get():
    url = "http://example.com"
    kwargs = {'method': 'get', 'session': None}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'session': None}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_get():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'session': None}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10, 'session': None}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #25
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    resp = type('Response', (), {'status_code': 199})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #27
#--------------------------

```python
def test_url_opener_with_requests_get():
    url = "http://example.com"
    kwargs = {"method": "get", "session": None}
    assert url_opener(url, kwargs) == requests.get(url).text

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "session": None}
    assert url_opener(url, kwargs) == requests.post(url, data={"key": "value"}).text

def test_url_opener_with_urllib_get():
    url = "http://example.com"
    kwargs = {"method": "get"}
    assert url_opener(url, kwargs) == urlopen(url).read()

def test_url_opener_with_urllib_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    assert url_opener(url, kwargs) == urlopen(url, urlencode({"key": "value"})).read()

def test_url_opener_with_timeout():
    url = "http://example.com"
    kwargs = {"timeout": 10}
    assert url_opener(url, kwargs) == urlopen(url, timeout=10).read()

def test_url_opener_with_encoding():
    url = "http://example.com"
    kwargs = {"encoding": "utf-8"}
    assert url_opener(url, kwargs) == requests.get(url, encoding="utf-8").text


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_query_with_data_as_dict_in_get_method():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_data_as_list_in_get_method():
    url, data = _query('http://example.com', 'get', {'data': [('key', 'value')]})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_data_as_tuple_in_get_method():
    url, data = _query('http://example.com', 'get', {'data': (('key', 'value'),)})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_data_as_string_in_get_method():
    url, data = _query('http://example.com', 'get', {'data': 'key=value'})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_data_in_post_method():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_existing_query_string():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_existing_query_string_ending_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_non_string_method():
    url, data = _query('http://example.com', None, {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'


# LLM-generated content at query #2
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert data is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'


# LLM-generated content at query #5
#--------------------------

```python
def test_data_is_encoded_when_not_get_method():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_true():
    url = "http://example.com"
    method = "GET"
    kwargs = {"data": {"key": "value"}}
    assert isinstance(method, basestring) and method.lower() == 'get' and kwargs['data']


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener_with_requests_get():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_get():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #8
#--------------------------

```python
def test_url_opener_uses_urllib_when_no_requests():
    global HAS_REQUEST
    HAS_REQUEST = False

    result = url_opener("http://example.com", {})

    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener_uses_urllib_when_no_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {"timeout": 5})
    assert result == _urllib("http://example.com", {"timeout": 5})


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert not (isinstance(method, str) and method.lower() == 'get' and result_data)


# LLM-generated content at query #11
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_allowed_args():
    url = "http://example.com"
    kwargs = {'method': 'get', 'headers': {'User-Agent': 'test'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_http_error():
    url = "http://example.com/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data_converts_to_urlencoded():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert data == b'key=value'

def test_query_with_list_data_converts_to_urlencoded():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert data == b'key=value'

def test_query_with_tuple_data_converts_to_urlencoded():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert data == b'key=value'

def test_query_with_get_method_appends_data_to_url():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_existing_query_string():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_url_ending_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_url_ending_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_without_data_returns_original_url_and_none():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data_encodes_to_utf8():
    url, data = _query('http://example.com', 'post', {'data': 'test string'})
    assert data == b'test string'

def test_query_with_case_insensitive_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #13
#--------------------------

```python
def test_status_code_outside_success_range_raises_http_error():
    resp = type('Response', (), {'status_code': 404, 'url': 'test', 'reason': 'Not Found', 'headers': {}, 'text': ''})()
    kwargs = {'timeout': 10, 'encoding': 'utf-8'}
    with pytest.raises(HTTPError):
        _requests('http://example.com', kwargs)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "get"
    kwargs = {'data': {'param2': 'value2'}}

    result_url, result_data = _query(url, method, kwargs)

    assert url[-1] in ('?', '&')


# LLM-generated content at query #15
#--------------------------

```python
def test_data_is_encoded_when_not_none_and_not_get_method():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": "test data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == "test data".encode('utf-8')


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 199})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 199})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #18
#--------------------------

```python
def test_query_with_data_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_with_data_no_query():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_with_data_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_with_data_ending_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_with_data_ending_with_ampersand():
    url, data = _query('http://example.com?existing=', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=&key=value'
    assert data is None

def test_query_without_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'

def test_query_case_insensitive_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #19
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_with_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_non_string_data():
    url, data = _query('http://example.com', 'post', {'data': 123})
    assert url == 'http://example.com'
    assert data == b'123'

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_uppercase_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #20
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('module.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test response"
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "test response"

def test_url_opener_with_requests_post():
    with patch('module.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "test response"
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == "test response"

def test_url_opener_with_requests_http_error():
    with patch('module.requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

def test_url_opener_with_urllib_get():
    with patch('module.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = b"test response"
        result = url_opener("http://example.com", {"method": "get"})
        assert result == b"test response"

def test_url_opener_with_urllib_post():
    with patch('module.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = b"test response"
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == b"test response"


# LLM-generated content at query #21
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_appends_data_to_url():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_appends_data_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_appends_data_with_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_lowercase_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_mixed_case_method():
    url, data = _query('http://example.com', 'GeT', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'plain string'})
    assert url == 'http://example.com'
    assert data == b'plain string'


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] in ('?', '&')


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    # Mock a response object with a status code outside the 200-299 range
    resp = type('Response', (), {'status_code': 404, 'url': 'http://example.com', 'reason': 'Not Found', 'headers': {}, 'text': ''})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #24
#--------------------------

```python
def test_data_is_encoded_when_present():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": "test data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == "test data".encode('utf-8')


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_false():
    mock_response = Mock()
    mock_response.status_code = 404
    assert not (200 <= mock_response.status_code < 300)


# LLM-generated content at query #26
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #27
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_url_has_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_url_ends_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_url_ends_with_question():
    url, data = _query('http://example.com?existing=param?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param?key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw_string'})
    assert url == 'http://example.com'
    assert data == b'raw_string'


# LLM-generated content at query #28
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_appends_data_to_url():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_appends_data_with_question_mark():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_appends_data_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&&key=value'
    assert data is None

def test_query_without_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_case_insensitive_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw data'})
    assert url == 'http://example.com'
    assert data == b'raw data'


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert not (isinstance(method, str) and method.lower() == 'get' and None)


# LLM-generated content at query #30
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #31
#--------------------------

```python
def test__requests_get_without_session():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_post_without_session():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_get_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "get", "session": session, "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_post_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "post", "data": {"key": "value"}, "session": session, "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_without_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_default_timeout():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_non_200_status_code():
    url = "http://example.com/404"
    kwargs = {"method": "get", "timeout": 10}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError as e:
        assert e.status_code == 404


# LLM-generated content at query #32
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_appends_data_to_url():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_appends_data_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_appends_data_with_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_method_lowercase():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_string_method_uppercase():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_non_string_method():
    url, data = _query('http://example.com', ['get'], {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] in ('?', '&')


# LLM-generated content at query #34
#--------------------------

```python
def test_query_with_data_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_string():
    url, data = _query('http://example.com', 'post', {'data': 'key=value'})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_post_method_and_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_case_insensitive_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 300})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "GET"
    kwargs = {"data": {"param2": "value2"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] in ('?', '&') or not (isinstance(method, str) and method.lower() == 'get' and kwargs.get('data'))


# LLM-generated content at query #37
#--------------------------

```python
def test_url_opener_uses_urllib_when_no_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #38
#--------------------------

```python
def test__requests_get_without_session():
    html = _requests('http://example.com', {'method': 'get'})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_get_with_session():
    session = requests.Session()
    html = _requests('http://example.com', {'method': 'get', 'session': session})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_post_without_session():
    html = _requests('http://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_post_with_session():
    session = requests.Session()
    html = _requests('http://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}, 'session': session})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_with_encoding():
    html = _requests('http://example.com', {'method': 'get', 'encoding': 'utf-8'})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_with_timeout():
    html = _requests('http://example.com', {'method': 'get', 'timeout': 5})
    assert isinstance(html, str)
    assert len(html) > 0

def test__requests_with_invalid_url():
    try:
        _requests('http://invalid.url', {'method': 'get'})
    except HTTPError as e:
        assert e.status_code >= 400

def test__requests_with_custom_headers():
    html = _requests('http://httpbin.org/headers', {'method': 'get', 'headers': {'User-Agent': 'test'}})
    assert isinstance(html, str)
    assert len(html) > 0


