####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

def test_url_opener():
    # Test Case 1: Successful requests via requests library (if available)
    if HAS_REQUEST:
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>success</html>"
            mock_response.url = "http://example.com"
            mock_get.return_value = mock_response

            result = url_opener("http://example.com", {"method": "get"})
            assert result == "<html>success</html>"
            mock_get.assert_called_once()

        # Test Case 2: HTTP Error handling in requests
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            mock_response.url = "http://example.com/404"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener("http://example.com/404", {"method": "get"})

        # Test Case 3: Query parameter encoding in GET
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""
            mock_get.return_value = mock_response

            kwargs = {"method": "get", "data": {"key": "value"}}
            url_opener("http://example.com", kwargs)
            # Check if url was modified to include query string
            args, kwargs_call = mock_get.call_args
            assert "key=value" in kwargs_call['url']

    # Test Case 4: urllib fallback (simulating HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            url = "http://example.com"
            kwargs = {"method": "get", "data": {"a": "b"}}
            
            url_opener(url, kwargs)
            
            # Verify urlopen was called with encoded URL
            args, kwargs_call = mock_urlopen.call_args
            assert "a=b" in args[0]
            assert kwargs_call['timeout'] == DEFAULT_TIMEOUT

    # Test Case 5: Testing _query logic directly for edge cases
    # Check append & if '?' already exists
    url_existing = "http://example.com?existing=true"
    kwargs_data = {"method": "get", "data": {"new": "val"}}
    new_url, data = _query(url_existing, "get", kwargs_data.copy())
    assert "existing=true" in new_url
    assert "new=val" in new_url
    assert "&" in new_url

    # Check append ? if no '?' exists
    url_no_query = "http://example.com"
    kwargs_data_simple = {"method": "get", "data": {"new": "val"}}
    new_url_simple, _ = _query(url_no_query, "get", kwargs_data_simple.copy())
    assert new_url_simple == "http://example.com?new=val"

    # Test Case 6: POST-like data handling (where method is not GET)
    kwargs_post = {"method": "post", "data": {"key": "value"}}
    url_post, data_post = _query("http://example.com", "post", kwargs_post.copy())
    # Data should be encoded as bytes for urlopen/requests post bodies
    assert data_post == b"key=value"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("get", {"data": {"key": "val"}}, "http://example.com?key=val"),
    ("get", {"data": [("a", "1"), ("b", "2")]}, "http://example.com?a=1&b=2"),
    ("get", {"data": {"key": "val"}, "other": "ignore"}, "http://example.com?key=val"),
    ("post", {"data": {"key": "val"}}, "http://example.com"), # data remains in body for non-GET
    ("get", {}, "http://example.com"),
])
def test_query_logic(method, params, expected_url):
    kwargs = {'method': method}
    if 'data' in params:
        kwargs['data'] = params['data']
    url, data = _query("http://example.com", method, kwargs)
    assert url == expected_url
    if 'data' in params and method.lower() != 'get':
        assert data is not None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        result = url_opener("http://example.com", {'method': 'get', 'encoding': 'utf-8'})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.Session') as mock_session_class:
        
        mock_session = MagicMock()
        mock_session_instance = mock_session_class.return_value
        mock_session_instance.get.return_value.status_code = 200
        mock_session_instance.get.return_value.text = "session_data"
        
        kwargs = {'method': 'get', 'session': mock_session_instance}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_data"
        mock_session_instance.get.assert_called_once()

def test_url_opener_allowed_args_passing():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        kwargs = {
            'method': 'get',
            'headers': {'User-Agent': 'test'},
            'cookies': {'session': '123'},
            'timeout': 10,
            'unrelated': 'noise' # Should be filtered out
        }
        
        url_opener("http://example.com", kwargs)
        
        args, kwargs_passed = mock_get.call_args
        assert kwargs_passed['headers'] == {'User-Agent': 'test'}
        assert kwargs_passed['cookies'] == {'session': '1203'}
        assert kwargs_passed['timeout'] == 10
        assert 'unrelated' not in kwargs_passed
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test GET with params
            kwargs = {'params': {'key': 'val'}, 'timeout': 10}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            # Check if _query appended data to URL correctly via the logic in _requests
            # Note: _requests calls _query which modifies the URL passed to meth
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']
            assert kwargs_call['timeout'] == 10

        # Test HTTP Error handling in requests
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.url = "http://example.com"
        mock_error_resp.reason = "Not Found"
        mock_error_resp.headers = {}

        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 2: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urllib_resp = MagicMock()
            mock_urllib_resp.read.return_value = b"<html>Urllib</html>"
            # urlopen returns a file-like object, we simulate the return value
            mock_urlopen.return_value = mock_urllib_resp

            kwargs = {'method': 'GET', 'timeout': 5}
            result = url_opener("http://example.com", kwargs)
            
            # Note: _urllib returns the result of urlopen directly. 
            # In a real scenario, one would call .read(), but based on code 
            # it returns the response object itself.
            assert mock_urlopen.called
            args, kwargs_call = mock_urlopen.call_args
            assert args[0] == "http://example.com"
            assert kwargs_call['timeout'] == 5

    # Test Case 3: _query logic for different methods/data types
    with patch('__main__.HAS_REQUEST', True):
        # Testing POST with dict data in _query via url_opener
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            kwargs = {'data': {'a': 1, 'b': 2}, 'method': 'post'}
            url_opener("http://example.com", kwargs)
            
            args, kwargs_call = mock_post.call_args
            # urlencode converts dict to string
            assert "a=1&b=2" in kwargs_call['url'] or "a=1&b=2" in args[0]

    # Test Case 4: Testing allowed_args passthrough
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'headers': {'User-Agent': 'Test'}, 'auth': ('user', 'pass')}
            url_opener("http://example.com", kwargs)
            
            args, kwargs_call = mock_get.call_args
            assert kwargs_call['headers'] == {'User-Agent': 'Test'}
            assert kwargs_call['auth'] == ('user', 'pass')
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            mock_get.assert_called_once_with(
                url="http://example.com", 
                timeout=10, 
                headers={'User-Agent': 'test'}
            )

    # Test Case 2: Using requests with query parameters (GET method)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Params</html>"
        mock_response.url = "http://example.com"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'data': {'key': 'val'}, 'method': 'GET'}
            url_opener("http://example.com", kwargs)
            # Verify _query appended data to URL
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']

    # Test Case 3: Using requests with HTTP Error (Non-2xx status)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            kwargs = {'method': 'POST', 'data': {'a': 'b'}}
            url_opener("http://example.com", kwargs)
            
            # Verify urlencode was applied and data was encoded to bytes
            args, kwargs_call = mock_urlopen.call_args
            assert "a=b" in args[0]
            assert args[1] == b'a=b'

    # Test Case 5: Using urllib with method 'GET' and existing query string
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            
            # Test appending & to existing query params
            kwargs = {'method': 'GET', 'data': {'c': 'd'}}
            url_opener("http://example.com?a=b", kwargs)
            
            args, _ = mock_urlopen.call_args
            assert "http://example.com?a=b&c=d" in args[0]

    # Test Case 6: Verify encoding handling in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        
        with patch('requests.get', return_value=mock_response):
            kwargs = {'encoding': 'utf-16'}
            url_opener("http://example.com", kwargs)
            assert mock_response.encoding == 'utf-16'
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Mocking dependencies and constants
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Setup mock response for successful GET
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response

        # Test Case 1: Successful GET with params via requests
        kwargs = {'params': {'key': 'val'}, 'timeout': 10}
        # Note: _requests implementation uses method from kwargs or defaults to 'get'
        # and processes url via _query. We test the flow logic here.
        result = url_opener("http://example.com", {'method': 'get', 'data': {'a': 1}})
        assert result == "<html>success</html>"
        # Verify that _query appended data to URL
        called_url = mock_get.call_args[1]['url']
        assert "a=1" in called_url

        # Test Case 2: Successful POST via requests
        mock_post.return_value = MagicMock(status_code=201, text="created", url="http://example.com")
        result_post = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'val'}})
        assert result_post == "created"

        # Test Case 3: HTTP Error raising
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.reason = "Not Found"
        mock_error_resp.url = "http://example.com"
        mock_error_resp.headers = {}
        mock_get.return_value = mock_error_resp
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

    # Test Case 4: urllib fallback (when HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_urlopen.return_value = MagicMock(read=lambda: b"urllib_data")
        # urllib returns the response object, we check if urlopen was called correctly
        res = url_opener("http://example.com", {'method': 'get', 'data': {'test': '1'}})
        
        args, kwargs = mock_urlopen.call_args
        assert "http://example.com?test=1" in args[0]
        assert kwargs['timeout'] == 60

    # Test Case 5: Session usage in requests
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        
        url_opener("http://example.com", {'method': 'get', 'session': mock_session})
        mock_session.get.assert_called()

    # Test Case 6: Encoding handling
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response.encoding = 'utf-8'
        url_opener("http://example.com", {'method': 'get', 'encoding': 'latin-1'})
        assert mock_response.encoding == 'latin-1'
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, data, expected_url", [
    ("GET", {"key": "val"}, "http://test.com?key=val"),
    ("GET", {"a": 1, "b": 2}, "http://test.com?a=1&b=2"),
    ("GET", "raw_data", "http://test.com?raw_data"),
    ("POST", {"key": "val"}, "http://test.com"),
])
def test_query_logic(method, data, expected_url):
    kwargs = {'data': data}
    url, processed_data = _query("http://test.com", method, kwargs)
    assert url == expected_url
    if isinstance(data, (dict, list, tuple)):
        assert processed_data is None
    elif isinstance(data, str):
        assert processed_data == data.encode('utf-8')

def test_url_opener_requests_success():
    with patch('HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch('HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_params():
    with patch('HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response

        kwargs = {
            'method': 'get',
            'data': {'query': 'pytest'},
            'headers': {'User-Agent': 'test'},
            'timeout': 10
        }
        url_opener("http://example.com", kwargs)
        
        args, kwargs_passed = mock_get.call_args
        assert 'query=pytest' in kwargs_passed['url']
        assert kwargs_passed['headers'] == {'User-Agent': 'test'}
        assert kwargs_passed['timeout'] == 10

def test_url_opener_with_session():
    with patch('HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "session_data"
        mock_session.get.return_value = mock_response

        result = url_opener("http://example.com", {'session': mock_session, 'method': 'get'})
        
        assert result == "session_data"
        mock_session.get.assert_called()
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: requests is available (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "success"
            mock_get.assert_called_once()

    # Test Case 2: requests is available, but returns Error (404)
    with patch('__main__.HAS_REQUEST', True):
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.url = "http://example.com"
        mock_error_resp.reason = "Not Found"
        mock_error_resp.headers = {}
        
        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test Case 3: requests is NOT available (HAS_REQUEST = False) - Using urllib
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock(read=lambda: b"urllib_data")
            # Note: urlopen returns a file-like object, text access is via .read() in real use, 
            # but we follow the logic that _urllib returns the result of urlopen directly.
            
            result = url_opener("http://example.com", {'method': 'get', 'data': {'key': 'val'}})
            assert "key=val" in str(mock_urlopen.call_args[0][0])
            assert mock_urlopen.called

    # Test Case 4: Testing _query logic within url_opener (GET with data)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "queried"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test appending '?' and encoding dict to query string
            url_opener("http://example.com", {'method': 'get', 'data': {'a': 1}})
            called_url = mock_get.call_args[1]['url']
            assert "a=1" in called_url
            assert "?" in called_url

    # Test Case 5: Testing allowed_args filtering
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "filtered"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {
                'method': 'get',
                'headers': {'User-Agent': 'test'},
                'unsupported_arg': 'ignore_me'
            }
            url_opener("http://example.com", kwargs)
            # Check that 'headers' is passed but 'unsupported_arg' is not
            passed_kwargs = mock_get.call_args[1]
            assert 'headers' in passed_kwargs
            assert 'unsupported_arg' not in passed_kwargs

    # Test Case 6: Testing session usage
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_success"
        
        url_opener("http://example.com", {'method': 'get', 'session': mock_session})
        mock_session.get.assert_called()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, data, expected_url", [
    ('get', {'a': '1'}, "http://test.com?a=1"),
    ('get', {'a': '1', 'b': '2'}, "http://test.com?a=1&b=2"),
    ('get', None, "http://test.com"),
    ('post', {'a': '1'}, "http://test.com"),
])
def test_query_logic(method, data, expected_url):
    kwargs = {'method': method}
    if data:
        kwargs['data'] = data
    url, processed_data = _query("http://test.com", method, kwargs)
    assert url == expected_url
    if data and isinstance(data, dict):
        assert processed_data is not None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_get.return_value = mock_response
        
        result = url_opener("http://test.com", {'method': 'get'})
        
        assert result == "success"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://test.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://test.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://test.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_data_params():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "created"
        mock_post.return_value = mock_response
        
        kwargs = {'method': 'post', 'data': {'key': 'val'}, 'auth': ('u', 'p')}
        result = url_opener("http://test.com", kwargs)
        
        assert result == "created"
        # Check if data was encoded and passed correctly via the _query logic internal to _requests
        # Note: in _requests, for POST, it doesn't append to URL, but passes via kwarg
        args, kwargs_call = mock_post.call_args
        assert kwargs_call['auth'] == ('u', 'p')

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response
        
        url_opener("http://test.com", {'encoding': 'utf-16'})
        assert mock_response.encoding == 'utf-16'
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Requests is available - Successful GET request
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success_html"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        kwargs = {'timeout': 10, 'encoding': 'utf-8'}
        result = url_opener("http://example.com", kwargs)

        assert result == "success_html"
        mock_get.assert_called_once()

    # Test case 2: Requests is available - HTTPError on non-2xx status
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {})

    # Test case 3: Requests is available - GET with data/params encoding
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "encoded"
        mock_get.return_value = mock_response

        # Passing data in kwargs should trigger _query logic for GET
        kwargs = {'data': {'key': 'value'}, 'method': 'get'}
        url_opener("http://example.com", kwargs)
        
        # Verify url was modified to include query string
        args, kwargs_called = mock_get.call_args
        assert "key=value" in kwargs_called['url']

    # Test case 4: Requests is NOT available - Successful urllib request
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'timeout': 5})
        
        mock_urlopen.assert_called_once_with(
            "http://example.com", None, timeout=5
        )

    # Test case 5: Requests is NOT available - urllib with data/POST
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_urlopen.return_value = MagicMock()
        
        kwargs = {'method': 'post', 'data': {'a': 'b'}}
        url_opener("http://example.com", kwargs)
        
        # Check that data was encoded and passed to urlopen
        args, kwargs_called = mock_urlopen.call_args
        assert args[0] == "http://example.com"
        assert args[1] == b'a=b'

    # Test case 6: Verify allowed_args filtering in _requests
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "filtered"
        mock_get.return_value = mock_response

        kwargs = {
            'auth': ('user', 'pass'),
            'headers': {'User-Agent': 'test'},
            'invalid_arg': 'should_be_ignored'
        }
        url_opener("http://example.com", kwargs)
        
        actual_kwargs = mock_get.call_args.kwargs
        assert 'auth' in actual_kwargs
        assert 'headers' in actual_kwargs
        assert 'invalid_arg' not in actual_kwargs
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>success</html>"
            mock_response.url = "http://example.com"
            mock_get.return_value = mock_response

            # Test simple GET
            result = url_opener("http://example.com", {})
            assert result == "<html>success</html>"
            mock_get.assert_called_with(url="http://example.com", timeout=60)

            # Test GET with params (data)
            kwargs = {'data': {'key': 'value'}, 'timeout': 10}
            result = url_opener("http://example.com", kwargs)
            assert "key=value" in mock_get.call_args[1]['url']
            assert mock_get.call_args[1]['timeout'] == 10

            # Test GET with encoding
            kwargs = {'encoding': 'utf-8'}
            url_opener("http://example.com", kwargs)
            assert mock_response.encoding == 'utf-8'

            # Test HTTP Error (404)
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 2: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response

            # Test urllib GET
            url_opener("http://example.com", {})
            mock_urlopen.assert_called_with("http://example.com", None, timeout=60)

            # Test urllib with data (POST-like behavior in _query logic)
            kwargs = {'method': 'post', 'data': {'a': 'b'}}
            url_opener("http://example.com", kwargs)
            # Note: _query converts dict to urlencoded string and encodes to bytes
            args, kwargs_call = mock_urlopen.call_args
            assert args[0] == "http://example.com"
            assert args[1] == b'a=b'

    # Test Case 3: Session usage in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_data"
        
        kwargs = {'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        assert result == "session_data"
        mock_session.get.assert_called()

    # Test Case 4: Allowed arguments filtering
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = ""
            
            kwargs = {
                'data': {'id': '1'},
                'headers': {'User-Agent': 'test'},
                'unallowed_arg': 'should_be_ignored'
            }
            url_opener("http://example.com", kwargs)
            # Check that headers were passed but unallowed_arg was not
            call_kwargs = mock_get.call_args[1]
            assert 'headers' in call_kwargs
            assert 'unallowed_arg' not in call_kwargs
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Mocking requests successful GET
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response
        
        kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "<html>Success</html>"
        mock_get.assert_called_once()
        # Verify data/params were handled via _query logic if applicable
        # In this case, no data was passed, so URL remains same

    # Test case 2: Mocking requests HTTPError (404)
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://example.com/bad"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com/bad", {})

    # Test case 3: Mocking urllib GET (when HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'timeout': 5})
        mock_urlopen.assert_called_once()

    # Test case 4: Testing _query logic for GET with data (params)
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = Magicmock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        # Testing URL encoding logic in _query
        kwargs = {'data': {'key': 'val'}, 'method': 'get'}
        url_opener("http://example.com", kwargs)
        
        # Check if URL was modified to include query string
        args, kwargs_call = mock_get.call_args
        assert "http://example.com?key=val" in kwargs_call['url']

    # Test case 5: Testing _query logic for POST (data in body)
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "created"
        mock_post.return_value = mock_response
        
        # For POST, data should be encoded and sent in the 'data' parameter, not URL
        kwargs = {'data': {'key': 'val'}, 'method': 'post'}
        url_opener("http://example.com", kwargs)
        
        args, kwargs_call = mock_post.call_args
        assert "http://example.com" in kwargs_call['url']
        # data should be bytes after encoding
        assert b'key=val' in kwargs_call['data']

    # Test case 6: Testing session usage
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_req_get:
        
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_ok"
        
        kwargs = {'session': mock_session, 'method': 'get'}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_ok"
        mock_session.get.assert_called_once()
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST is True)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>Success</html>"
        mock_resp.url = "http://example.com"
        mock_response_method = MagicMock(return_value=mock_resp)
        
        with patch('requests.get', mock_response_method):
            # Test GET with params in kwargs (via _query logic)
            kwargs = {'params': {'key': 'val'}} # Note: _query uses 'data' key for encoding
            # Testing the specific implementation of _query which looks for 'data'
            result = url_opener("http://example.com", {'data': {'a': 1}, 'timeout': 10})
            
            assert result == "<html>Success</html>"
            # Verify URL was encoded correctly by _query
            args, kwargs_call = mock_response_method.call_args
            assert "http://example.com?a=1" in args[0] or "http://example.com?a=1" == args[0]

    # Test case 2: Using requests with error (HTTPError)
    with patch('__main__.HAS_REQUEST', True):
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.reason = "Not Found"
        mock_error_resp.url = "http://example.com"
        mock_error_resp.headers = {}
        
        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test case 3: Using urllib (HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urllib_response = MagicMock()
        with patch('__main__.urlopen', return_value=mock_urllib_response) as mock_urlopen:
            # Test POST-like behavior with data
            url_opener("http://example.com", {'data': {'key': 'val'}, 'method': 'post'})
            
            args, kwargs_call = mock_urlopen.call_args
            assert "http://example.com?key=val" in args[0] or "key=val" in args[1]

    # Test case 4: _query logic for URL appending
    # Testing existing query params handling
    url, data = _query("http://test.com?existing=true", "get", {'data': {'new': 'val'}})
    assert url == "http://test.com?existing=true&new=val"

    # Test case 5: allowed_args filtering
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        
        with patch('requests.get', return_value=mock_resp) as mock_get:
            kwargs = {
                'data': {'foo': 'bar'},
                'auth': ('user', 'pass'),
                'invalid_arg': 'should_be_ignored'
            }
            url_opener("http://example.com", kwargs)
            
            # Check that only allowed args were passed to the request method
            _, call_kwargs = mock_get.call_args
            assert 'auth' in call_kwargs
            assert 'invalid_arg' not in call_kwargs

    # Test case 6: encoding handling in _requests
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "encoded"
        
        with patch('requests.get', return_value=mock_resp):
            url_opener("http://example.com", {'encoding': 'utf-16'})
            assert mock_resp.encoding == 'utf-16'
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>Success</html>"
            mock_response.url = "http://example.com"
            mock_get.return_value = mock_response

            kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()
            args, kwargs_passed = mock_get.call_args
            assert args[0] == "http://example.com"
            assert kwargs_passed['timeout'] == 10
            assert kwargs_passed['headers'] == {'User-Agent': 'test'}

    # Test case 2: Using requests with data (GET method converts dict to query params)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "data_result"
            mock_get.return_value = mock_response

            kwargs = {'data': {'key': 'value'}}
            result = url_opener("http://example.com", kwargs)
            
            assert "key=value" in mock_get.call_args[1]['url']
            assert result == "data_result"

    # Test case 3: Using requests and handling HTTPError (404)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.url = "http://example.com"
            mock_response.reason = "Not Found"
            mock_response.headers = {}
            mock_get.return='value' # Dummy
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            kwargs = {'method': 'POST', 'data': 'payload'}
            url_opener("http://example.com", kwargs)
            
            # Check if urlopen was called with encoded data
            args, kwargs_passed = mock_urlopen.call_args
            assert args[0] == "http://example.com"
            assert args[1] == b'payload'
            assert kwargs_passed['timeout'] == 60

    # Test case 5: Using urllib with GET and query params
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            
            kwargs = {'method': 'GET', 'data': {'a': 1}}
            url_opener("http://example.com", kwargs)
            
            args, _ = mock_urlopen.call_args
            assert "a=1" in args[0]
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

    # Test case 2: Using requests with data/query params (GET)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'data': {'key': 'val'}, 'method': 'get'}
            url_opener("http://example.com", kwargs)
            # Check if url was encoded: http://example.com?key=val
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']

    # Test case 3: Using requests and raising HTTPError on non-200 status
    with patch('__main__.HAS_REQUEST', True):
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.url = "http://example.com"
        mock_error_resp.reason = "Not Found"
        mock_error_resp.headers = {}

        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urllib_resp = MagicMock()
        mock_urllib_resp.read.return_value = b"urllib content"
        
        with patch('urllib.request.urlopen', return_value=mock_urllib_resp) as mock_urlopen:
            result = url_opener("http://example.com", {'method': 'get'})
            # Note: urllib returns the response object, not text directly like _requests
            assert result.read() == b"urllib content"
            mock_urlopen.assert_called_once()

    # Test case 5: Testing _query logic specifically via url_opener
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test URL with existing params
            url_opener("http://example.com?existing=1", {'data': {'new': '2'}, 'method': 'get'})
            args, kwargs_call = mock_get.call_args
            assert "existing=1" in kwargs_call['url']
            assert "new=2" in kwargs_call['url']

    # Test case 6: Testing allowed_args passing to requests
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'headers': {'User-Agent': 'Test'}, 'timeout': 10}
            url_opener("http://example.com", kwargs)
            _, kwargs_call = mock_get.call_args
            assert kwargs_call['headers'] == {'User-Agent': 'Test'}
            assert kwargs_call['timeout'] == 10
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ('get', {'data': {'a': '1', 'b': '2'}}, "http://example.com?a=1&b=2"),
    ('get', {'data': [('a', '1'), ('b', '2')]}, "http://example/path?a=1&b=2"),
    ('get', {'data': {'a': '1'}}, "http://example.com?a=1"),
    ('post', {'data': {'key': 'val'}}, "http://example.com"), # data handled in _requests/urllib logic
])
def test_url_opener_query_logic(method, params, expected_url):
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "success"
        mock_resp.url = "http://example.com"
        
        with patch('requests.get', return_value=mock_resp) as mock_get:
            kwargs = {'method': method, 'data': params.get('data')} if 'data' in params else {}
            if 'data' in params:
                kwargs['data'] = params['data']
                
            result = url_opener("http://example.com", kwargs)
            
            assert result == "success"
            # Check if query string was appended correctly for GET
            if method.lower() == 'get' and 'data' in params:
                args, kwargs_call = mock_get.call_args
                assert args[0] == expected_url

def test_url_opener_requests_error():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"
        mock_resp.url = "http://example.com"
        mock_resp.headers = {}
        
        with patch('requests.get', return_value=mock_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

def test_url_opener_urllib_fallback():
    with patch('__main__.HAS_REQUEST', False):
        mock_response = MagicMock()
        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            mock_response.read.return_value = b"content"
            # Note: _urllib returns the object from urlopen, which is usually a response object
            # In a real scenario, we'd check if it returns the stream or text. 
            # Based on the provided code, _urllib returns the raw urllib object.
            result = url_opener("http://example.com", {'method': 'get', 'data': {'k': 'v'}})
            assert mock_urlopen.called
            args, kwargs = mock_urlopen.call_args
            assert "http://example.com?k=v" in args[0]

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_success"
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_success"
        mock_session.get.assert_called()

def test_url_opener_allowed_args():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        
        kwargs = {
            'method': 'post',
            'headers': {'User-Agent': 'test'},
            'cookies': {'session': '123'}
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            url_opener("http://example.com", kwargs)
            _, passed_kwargs = mock_post.call_args
            assert passed_kwargs['headers'] == {'User-Agent': 'test'}
            assert passed_kwargs['cookies'] == {'session': '123'}
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Mocking requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test GET with params via kwargs
            kwargs = {'data': {'key': 'value'}, 'timeout': 10}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>success</html>"
            # Verify url encoding in _query logic
            args, kwargs_called = mock_get.call_args
            assert "key=value" in kwargs_called['url']
            assert kwargs_called['timeout'] == 10

        # Test HTTP Error handling in _requests
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        mock_error_response.url = "http://example.com"
        mock_error_response.reason = "Not Found"
        mock_error_response.headers = {}
        
        with patch('requests.get', return_value=mock_error_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test case 2: Mocking urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('__main__.urlopen') as mock_urlopen:
            mock_stream = MagicMock()
            mock_stream.read.return_value = b"urllib content"
            # urlopen returns a file-like object; we simulate the return for simplicity
            # Note: In real urllib, one would call .read() on the result
            mock_urlopen.return_value = mock_stream
            
            # Mocking _urllib behavior (which calls urlopen)
            kwargs = {'method': 'get', 'timeout': 5}
            url_opener("http://example.com", kwargs)
            
            args, kwargs_called = mock_urlopen.call_args
            assert args[0] == "http://example.com"
            assert kwargs_called['timeout'] == 5

    # Test case 3: Verifying _query logic for complex URL appending
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = ""
            
            # Test appending with existing query params
            url_opener("http://example.com?a=b", {'data': {'c': 'd'}, 'method': 'get'})
            args, kwargs_called = mock_get.call_args
            assert "http://example.com?a=b&c=d" in kwargs_called['url']

    # Test case 4: Testing session usage in _requests
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session data"
        
        kwargs = {'session': mock_session, 'method': 'get'}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session data"
        mock_session.get.assert_called()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, data, expected_url", [
    ('GET', {'a': '1'}, "http://test.com?a=1"),
    ('GET', {'a': '1', 'b': '2'}, "http://test.com?a=1&b=2"),
    ('GET', None, "http://test.com"),
    ('POST', {'key': 'val'}, "http://test.com"),
])
def test_query_logic(method, data, expected_url):
    kwargs = {'method': method}
    if data:
        kwargs['data'] = data
    url, processed_data = _query("http://test.com", method, kwargs)
    
    if method.upper() == 'GET' and data:
        assert url == expected_url
    else:
        assert url == "http://test.com"

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        result = url_opener("http://example.com", {'method': 'get'})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_failure():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_data_post():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        mock_post.return_value = mock_response
        
        payload = {'key': 'value'}
        result = url_opener("http://example.com", {'method': 'post', 'data': payload})
        
        assert result == "Created"
        # Check if data was passed as bytes in post
        args, kwargs = mock_post.call_args
        assert kwargs['data'] == b'key=value'

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Session Content"
        mock_session.get.return_value = mock_response
        
        result = url_opener("http://example.com", {'method': 'get', 'session': mock_session})
        
        assert result == "Session Content"
        mock_session.get.assert_called_once()

def test_url_opener_with_allowed_args():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_get.return_value = mock_response
        
        kwargs = {
            'method': 'get',
            'headers': {'User-Agent': 'Test'},
            'timeout': 10,
            'extra_arg': 'ignore_me'
        }
        
        url_opener("http://example.com", kwargs)
        
        _, call_kwargs = mock_get.call_args
        assert call_kwargs['headers'] == {'User-Agent': 'Test'}
        assert call_kwargs['timeout'] == 10
        assert 'extra_arg' not in call_kwargs
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, data, expected_url", [
    ("GET", {"a": "1"}, "http://test.com?a=1"),
    ("GET", {"a": "1", "b": "2"}, "http://test.com?a=1&b=2"),
    ("GET", None, "http://test.com"),
    ("POST", {"key": "val"}, "http://test.com"),
])
def test_query_logic(method, data, expected_url):
    kwargs = {}
    if data:
        kwargs['data'] = data
    
    url, processed_data = _query("http://test.com", method, kwargs)
    
    if method.upper() == "GET" and data:
        assert url == expected_url
    else:
        assert url == "http://test.com"

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        result = url_opener("http://example.com", {"method": "get"})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_failure():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {"method": "get"})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_data_post():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        mock_post.return_value = mock_response
        
        kwargs = {'data': {'key': 'val'}, 'method': 'post'}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "Created"
        # Check if data was encoded and passed in body
        args, kwargs_call = mock_post.call_args
        assert b'key=val' in args[0] or kwargs_call['data'] == b'key=val' or any(b'key=val' == v for v in kwargs_call.values())

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Session Content"
        mock_session.get.return_value = mock_response
        
        result = url_opener("http://example.com", {"session": mock_session, "method": "get"})
        
        assert result == "Session Content"
        mock_session.get.assert_called()

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response
        
        url_opener("http://example.com", {"encoding": "utf-8"})
        
        assert mock_response.encoding == "utf-8"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("GET", {"data": {"key": "val"}}, "http://test.com?key=val"),
    ("GET", {"data": [("a", "1"), ("b", "2")]}, "http://test.com?a=1&b=2"),
    ("GET", {"data": {"key": "val"}}, "http://test.com/path?key=val"),
])
def test_query_logic(method, params, expected_url):
    kwargs = {"data": params}
    url, data = _query("http://test.com/path", method, kwargs)
    assert url == expected_url

@patch('__main__.HAS_REQUEST', True)
@patch('requests.get')
def test_url_opener_requests_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>success</html>"
    mock_response.url = "http://test.com"
    mock_get.return_value = mock_response

    kwargs = {"timeout": 10, "headers": {"User-Agent": "test"}}
    result = url_opener("http://test.com", kwargs)

    assert result == "<html>success</html>"
    mock_get.assert_called_once()
    args, kwargs_call = mock_get.call_args
    assert kwargs_call['timeout'] == 10
    assert kwargs_call['headers'] == {"User-Agent": "test"}

@patch('__main__.HAS_REQUEST', True)
@patch('requests.get')
def test_url_opener_requests_error(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    mock_response.url = "http://test.com"
    mock_response.headers = {}
    mock_get.return_value = mock_response

    with pytest.raises(HTTPError):
        url_opener("http://test.com", {})

@patch('__main__.HAS_REQUEST', False)
@patch('__main__.urlopen')
def test_url_opener_urllib_success(mock_urlopen):
    mock_response = MagicMock()
    mock_urlopen.return_value = mock_response
    
    kwargs = {"method": "POST", "data": {"key": "val"}}
    url_opener("http://test.com", kwargs)
    
    # Check if urlencode was applied to the URL in GET mode or data passed to post
    args, kwargs_call = mock_urlopen.call_args
    assert "http://test.com" in args[0]

@patch('__main__.HAS_REQUEST', True)
@patch('requests.get')
def test_url_opener_encoding(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "content"
    mock_get.return_value = mock_response

    result = url_opener("http://test.com", {"encoding": "utf-8"})
    assert mock_response.encoding == "utf-8"
    assert result == "content"

@patch('__main__.HAS_REQUEST', True)
@patch('requests.get')
def test_url_opener_session(mock_get):
    mock_session = MagicMock()
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.text = "session_data"
    
    kwargs = {"session": mock_session}
    result = url_opener("http://test.com", kwargs)
    
    assert result == "session_data"
    mock_session.get.assert_called_once()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("get", {"data": {"key": "val"}}, "http://test.com?key=val"),
    ("get", {"data": [("a", "1"), ("b", "2")]}, "http://test.com?a=1&b=2"),
    ("get", {"data": {"q": "search term"}}, "http://test.com?q=search+term"),
    ("post", {"data": {"key": "val"}}, "http://test.com"),
])
def test_query_logic(method, params, expected_url):
    kwargs = {"data": params}
    url, data = _query("http://test.com", method, kwargs)
    assert url == expected_url
    if params and method.lower() == 'get':
        assert data is None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        result = url_opener("http://test.com", {"method": "get"})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://test.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://test.com", {"method": "get"})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://test.com", {"method": "get", "data": {"a": "b"}})
        
        # Check if urlencode was applied to the URL in urlopen call
        args, kwargs = mock_urlopen.call_args
        assert "a=b" in args[0]

def test_url_opener_requests_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "session_data"
        mock_session.get.return_value = mock_response
        
        result = url_opener("http://test.com", {"method": "get", "session": mock_session})
        
        assert result == "session_data"
        mock_session.get.assert_called_once()

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response
        
        url_opener("http://test.com", {"encoding": "utf-8"})
        
        assert mock_response.encoding == "utf-8"

def test_url_opener_complex_query_params():
    # Test handling of existing query parameters in URL
    kwargs = {"data": {"b": "2"}}
    url, data = _query("http://test.com?a=1", "get", kwargs)
    assert url == "http://test.com?a=1&b=2"
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, data, expected_url", [
    ('GET', {'a': '1'}, "http://test.com?a=1"),
    ('GET', {'a': '1', 'b': '2'}, "http://test.com?a=1&b=2"),
    ('GET', None, "http://test.com"),
    ('POST', {'key': 'val'}, "http://test.com"),
])
def test_query_logic(method, data, expected_url):
    kwargs = {}
    if data:
        kwargs['data'] = data
    url, processed_data = _query("http://test.com", method, kwargs)
    
    if method.upper() == 'GET' and data:
        assert url == expected_url
    elif data:
        assert processed_data is not None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()
        args, kwargs_passed = mock_get.call_args
        assert kwargs_passed['timeout'] == 10
        assert kwargs_passed['headers'] == {'User-Agent': 'test'}

def test_url_opener_requests_http_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'timeout': 5})
        
        mock_urlopen.assert_called_once()
        args, kwargs = mock_urlopen.call_args
        assert args[0] == "http://example.com"
        assert kwargs['timeout'] == 5

def test_url_opener_query_params_integration():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        # Testing GET with data injection into URL
        url_opener("http://test.com", {'method': 'GET', 'data': {'id': '123'}})
        
        called_url = mock_get.call_args[1]['url']
        assert "id=123" in called_url

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response
        
        url_opener("http://example.com", {'encoding': 'utf-8'})
        
        assert mock_response.encoding == 'utf-8'
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ('GET', {'data': {'a': '1', 'b': '2'}}, "http://example.com?a=1&b=2"),
    ('GET', {'data': 'simple_string'}, "http://example.com?simple_string"),
    ('POST', {'data': {'key': 'val'}}, "http://example.com"),
    ('GET', {}, "http://example.com"),
])
def test_query_logic(method, params, expected_url):
    url = "http://example.com"
    kwargs = {'method': method}
    kwargs.update(params)
    
    result_url, result_data = _query(url, method, kwargs.copy())
    assert result_url == expected_url
    if params.get('data') and isinstance(params['data'], (dict, list, tuple)):
        assert result_data is not None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        kwargs = {'method': 'get', 'timeout': 10}
        result = url_opener("http://test.com", kwargs)
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_failure():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://test.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://test.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        # urllib.request.urlopen returns a file-like object; 
        # in this context, the code expects the return value of urlopen to be returned directly
        mock_urlopen.return_value = mock_response
        
        kwargs = {'method': 'get', 'timeout': 5}
        result = url_opener("http://test.com", kwargs)
        
        assert result == mock_response
        mock_urlopen.assert_called_once()

def test_url_opener_requests_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "session_content"
        mock_session.get.return_value = mock_response
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://test.com", kwargs)
        
        assert result == "session_content"
        mock_session.get.assert_called_once()

def test_query_url_with_existing_params():
    url = "http://example.com?existing=true"
    kwargs = {'method': 'GET', 'data': {'new': 'val'}}
    
    result_url, _ = _query(url, 'GET', kwargs)
    assert result_url == "http://example.com?existing=true&new=val"

def test_query_url_with_trailing_separator():
    url = "http://example.com?"
    kwargs = {'method': 'GET', 'data': {'new': 'val'}}
    
    result_url, _ = _query(url, 'GET', kwargs)
    assert result_url == "http://example.com?new=val"

def test_requests_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "utf8_content"
        mock_get.return_value = mock_response
        
        kwargs = {'method': 'get', 'encoding': 'utf-8'}
        url_opener("http://test.com", kwargs)
        
        assert mock_response.encoding == 'utf-8'

def test_requests_allowed_args_passing():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_get.return_value = mock_response
        
        kwargs = {'method': 'get', 'headers': {'User-Agent': 'test'}, 'auth': ('u', 'p')}
        url_opener("http://test.com", kwargs)
        
        # Check if headers and auth were passed to the underlying request call
        args, kwargs_passed = mock_get.call_args
        assert kwargs_passed['headers'] == {'User-Agent': 'test'}
        assert kwargs_passed['auth'] == ('u', 'p')
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method", ["get", "post"])
def test_url_opener(method):
    url = "http://example.com"
    kwargs = {"method": method, "timeout": 10}
    
    if HAS_REQUEST:
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>success</html>"
            mock_response.url = url
            mock_get.return_value = mock_response
            
            result = url_opener(url, kwargs)
            
            assert result == "<html>success</html>"
            mock_get.assert_called_once()
    else:
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"<html>success</html>"
            # urllib open returns a file-like object, we simulate the context
            mock_urlopen.return_value = mock_response
            
            result = url_opener(url, kwargs)
            
            assert mock_urlopen.called
    
def test_url_opener_query_params():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    
    if HAS_REQUEST:
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "ok"
            mock_get.return_value = mock_response
            
            url_opener(url, kwargs)
            # Check if data was appended to URL via _query logic
            args, kwargs_call = mock_get.call_args
            assert "key=value" in kwargs_call['url']
    else:
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            url_opener(url, kwargs)
            args, kwargs_call = mock_urlopen.call_args
            assert "key=value" in args[0]

def test_url_opener_http_error():
    url = "http://example.com/404"
    kwargs = {"method": "get"}
    
    if HAS_REQUEST:
        with patch('requests.get') as mock_get:
            mock_response = MagicDISCARD = MagicMock()
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            mock_response.url = url
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            with pytest.raises(HTTPError):
                url_opener(url, kwargs)
    else:
        with patch('urllib.request.urlopen', side_effect=HTTPError(url, 404, "Not Found", {}, None)):
            with pytest.raises(HTTPError):
                url_opener(url, kwargs)

def test_url_opener_session():
    if HAS_REQUEST:
        url = "http://example.com"
        kwargs = {"method": "get", "session": MagicMock()}
        mock_session = kwargs["session"]
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_ok"
        
        result = url_opener(url, kwargs)
        assert result == "session_ok"
        mock_session.get.assert_called_once()

def test_url_opener_encoding():
    if HAS_REQUEST:
        url = "http://example.com"
        kwargs = {"method": "get", "encoding": "utf-8"}
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "encoded"
            mock_get.return_value = mock_response
            
            url_opener(url, kwargs)
            assert mock_response.encoding == "utf-8"

def test_url_opener_allowed_args():
    if HAS_REQUEST:
        url = "http://example.com"
        kwargs = {"method": "get", "headers": {"User-Agent": "test"}, "extra": "ignore"}
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""
            mock_get.return_value = mock_response
            
            url_opener(url, kwargs)
            _, call_kwargs = mock_get.call_args
            assert 'headers' in call_kwargs
            assert call_kwargs['headers'] == {"User-Agent": "test"}
            assert 'extra' not in call_kwargs
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("get", {"data": {"key": "val"}}, "http://test.com?key=val"),
    ("get", {"data": [("a", "b")]}, "http://test.com?a=b"),
    ("get", {"data": "raw_string"}, "http://test.com?raw_string"),
    ("post", {"data": {"key": "val"}}, "http://test.com"),
])
def test_url_opener_query_logic(method, params, expected_url):
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        kwargs = {'method': method, 'data': params}
        # Force urllib path by setting HAS_REQUEST to False
        with patch('__main__.HAS_REQUEST', False):
            url_opener("http://test.com", kwargs)
            actual_url, actual_data = mock_urlopen.call_args[0][0], mock_urlopen.call_args[0][1]
            assert actual_url == expected_url
            if params and isinstance(params, (dict, list, tuple)):
                assert actual_data is not None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>success</html>"
        mock_resp.url = "http://test.com"
        mock_resp.reason = "OK"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = url_opener("http://test.com", {'method': 'get', 'encoding': 'utf-8'})
            assert result == "<html>success</html>"
            mock_get.assert_called_once()

def test_url_opener_requests_http_error():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"
        mock_resp.url = "http://test.com"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp):
            with pytest.raises(HTTPError):
                url_opener("http://test.com", {'method': 'get'})

def test_url_opener_urllib_execution():
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            url_opener("http://test.com", {'method': 'post', 'data': {'id': '1'}})
            # Verify urlopen was called with the encoded data
            args, kwargs = mock_urlopen.call_args
            assert "id=1" in args[0]

def test_url_opener_requests_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "session_data"
        mock_session.get.return_value = mock_resp

        url_opener("http://test.com", {'method': 'get', 'session': mock_session})
        mock_session.get.assert_called_once()

def test_query_append_ampersand():
    # Test logic where url already has params
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        with patch('__main__.HAS_REQUEST', False):
            url_opener("http://test.com?existing=true", {'method': 'get', 'data': {'new': 'val'}})
            actual_url = mock_urlopen.call_args[0][0]
            assert "existing=true" in actual_url
            assert "new=val" in actual_url
            assert "&" in actual_url
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("GET", {"data": {"a": "1"}}, "http://test.com?a=1"),
    ("GET", {"data": [{"a": "1"}]}, "http://test.com?a=1"),
    ("GET", {"data": {"a": "1", "b": "2"}}, "http://test.com?a=1&b=2"),
    ("GET", {"data": {"a": "1"}}, "http://test.com/path?a=1"),
    ("POST", {"data": {"key": "val"}}, "http://test.com"), # data is popped from kwargs in _query
])
def test_url_opener_query_logic(method, params, expected_url):
    # Testing the internal _query logic via url_opener simulation
    kwargs = {'method': method}
    if 'data' in params:
        kwargs['data'] = params['data']
    
    url, data = _query("http://test.com/path", method, kwargs)
    
    if method.upper() == "GET" and params.get('data'):
        assert url.startswith("http://test.com/path?") or url.startswith("http://test.com/path&")
        assert "a=1" in url
    else:
        # For POST, data is handled separately in _requests but extracted here
        pass

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False):
        with patch('__main__.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            url = "http://example.com"
            kwargs = {'method': 'get', 'timeout': 10}
            
            result = url_opener(url, kwargs)
            
            mock_urlopen.assert_called_once()
            assert result == mock_response

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html></html>"
        mock_resp.url = "http://example.com"
        mock_resp.reason = "OK"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp) as mock_get:
            kwargs = {'method': 'get', 'timeout': 5, 'headers': {'User-Agent': 'test'}}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html></html>"
            mock_get.assert_called_once()
            assert mock_get.call_args[1]['headers'] == {'User-Agent': 'test'}

def test_url_opener_requests_http_error():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.url = "http://example.com"
        mock_resp.reason = "Not Found"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

def test_url_opener_requests_encoding():
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "content"
        
        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = url_opener("http://example.com", {'method': 'get', 'encoding': 'utf-8'})
            assert mock_resp.encoding == 'utf-8'
            assert result == "content"

def test_url_opener_session_usage():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_content"
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_content"
        mock_session.get.assert_called_once()
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {'method': 'get', 'encoding': 'utf-8'})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

    # Test case 2: Using requests with error (HTTPError)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test case 3: Using requests with query parameters (GET)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "data"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", {'method': 'get', 'data': {'key': 'val'}})
            # Check if url was encoded correctly
            args, kwargs = mock_get.call_args
            assert "key=val" in kwargs['url']

    # Test case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urllib_resp = MagicMock()
        with patch('urllib.request.urlopen', return_value=mock_urllib_resp) as mock_urlopen:
            # Mocking the context manager behavior of urlopen if needed, 
            # but here we just need it to return an object
            url_opener("http://example.com", {'method': 'get'})
            mock_urlopen.assert_called_once()

    # Test case 5: Using urllib with POST data
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            url_opener("http://example.com", {'method': 'post', 'data': {'a': 'b'}})
            args, kwargs = mock_urlopen.call_args
            assert "a=b" in args[0]
            assert args[1] is not None # data should be encoded bytes

    # Test case 6: Using session with requests
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_data"
        
        with patch('requests.get') as mock_reg_get:
            result = url_opener("http://example.com", {'method': 'get', 'session': mock_session})
            assert result == "session_data"
            mock_session.get.assert_called_once()
            mock_reg_get.assert_not_called()
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Mocking dependencies
    with patch('__main__.HAS_REQUEST', True), \
         patch('__main__.requests') as mock_requests, \
         patch('__main__.urlopen') as mock_urlopen:
        
        # Setup Mock Response for requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_requests.get.return_value = mock_response

        # Test Case 1: Successful GET request with requests
        kwargs = {'timeout': 10, 'data': {'key': 'val'}}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "<html>Success</html>"
        # Verify _query logic (params appended to URL)
        mock_requests.get.assert_called_with(
            url="http://example.com?key=val", 
            timeout=10, 
            data=None # data popped by _query
        )

        # Test Case 2: HTTP Error handling in requests
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

        # Test Case 3: Testing urllib fallback
        with patch('__main__.HAS_REQUEST', False), \
             patch('__main__.urlopen') as mock_urllib_open:
            
            mock_urllib_open.return_value = MagicMock(read=lambda: b"urllib data")
            # Note: urlopen returns a file-like object, we simulate content via read() 
            # but since _urllib returns the object itself in original code:
            res_urllib = url_opener("http://example.com", {'method': 'get', 'data': {'a': '1'}})
            
            assert "a=1" in str(mock_urllib_open.call_args[0][0])
            mock_urllib_open.assert_called()

        # Test Case 4: Testing encoding and allowed args
        kwargs_advanced = {
            'method': 'post',
            'data': {'foo': 'bar'},
            'headers': {'User-Agent': 'test'},
            'encoding': 'utf-8'
        }
        mock_response.status_code = 201
        mock_response.text = "Created"
        
        result_advanced = url_opener("http://example.com/post", kwargs_advanced)
        
        assert result_advanced == "Created"
        args, kwargs_call = mock_requests.post.call_args
        assert kwargs_call['headers'] == {'User-Agent': 'test'}
        assert kwargs_call['timeout'] == 60 # Default

    # Test Case 5: _query logic for URL construction (no params)
    from __main__ import _query
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None

    # Test Case 6: _query logic for URL construction (existing query string)
    url, data = _query("http://example.com?existing=true", "get", {'new': 'val'})
    assert url == "http://example.com?existing=true&new=val"

    # Test Case 7: _query logic for POST data encoding
    url, data = _query("http://example.com", "post", {'key': 'value'})
    assert data == b'key=value'
```


