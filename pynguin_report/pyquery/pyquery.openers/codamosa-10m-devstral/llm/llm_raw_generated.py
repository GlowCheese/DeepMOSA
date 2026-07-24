####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_url_opener():
    # Test with requests library
    if HAS_REQUEST:
        # Test GET request
        html = url_opener("https://httpbin.org/get", {"method": "get"})
        assert isinstance(html, str)
        assert "httpbin.org" in html

        # Test POST request
        html = url_opener("https://httpbin.org/post", {"method": "post", "data": {"key": "value"}})
        assert isinstance(html, str)
        assert "httpbin.org" in html

        # Test with encoding
        html = url_opener("https://httpbin.org/get", {"method": "get", "encoding": "utf-8"})
        assert isinstance(html, str)

        # Test with timeout
        html = url_opener("https://httpbin.org/get", {"method": "get", "timeout": 10})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener("https://httpbin.org/status/404", {"method": "get"})
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener("https://httpbin.org/get", {"method": "get"})
        assert response.getcode() == 200

        # Test POST request
        response = url_opener("https://httpbin.org/post", {"method": "post", "data": {"key": "value"}})
        assert response.getcode() == 200

        # Test with timeout
        response = url_opener("https://httpbin.org/get", {"method": "get", "timeout": 10})
        assert response.getcode() == 200


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        # Mock the requests.get method
        import requests
        from unittest.mock import patch, MagicMock

        # Test successful GET request
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "Success"
            mock_response.encoding = 'utf-8'
            mock_get.return_value = mock_response

            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "Success"

        # Test successful POST request
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "Success"
            mock_response.encoding = 'utf-8'
            mock_post.return_value = mock_response

            result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
            assert result == "Success"

        # Test HTTP error
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            mock_response.url = "http://example.com"
            mock_response.reason = "Not Found"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            try:
                url_opener("http://example.com", {'method': 'get'})
                assert False, "Expected HTTPError"
            except HTTPError as e:
                assert e.code == 404

    # Test with urllib (fallback)
    else:
        from unittest.mock import patch, MagicMock

        # Test successful GET request
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"Success"
            mock_urlopen.return_value = mock_response

            result = url_opener("http://example.com", {'method': 'get'})
            assert result == b"Success"

        # Test successful POST request
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"Success"
            mock_urlopen.return_value = mock_response

            result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
            assert result == b"Success"


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener(mocker):
    # Test with requests library available
    mocker.patch('requests.get', return_value=mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com',
        reason='OK',
        headers={},
        encoding='utf-8'
    ))
    mocker.patch('requests.post', return_value=mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com',
        reason='OK',
        headers={},
        encoding='utf-8'
    ))
    mocker.patch('requests.put', return_value=mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com',
        reason='OK',
        headers={},
        encoding='utf-8'
    ))

    # Test GET request with requests
    result = url_opener('http://test.com', {'method': 'get'})
    assert result == 'test response'

    # Test POST request with requests
    result = url_opener('http://test.com', {'method': 'post', 'data': {'key': 'value'}})
    assert result == 'test response'

    # Test PUT request with requests
    result = url_opener('http://test.com', {'method': 'put', 'data': {'key': 'value'}})
    assert result == 'test response'

    # Test with encoding parameter
    result = url_opener('http://test.com', {'method': 'get', 'encoding': 'latin-1'})
    assert result == 'test response'

    # Test with session parameter
    session = mocker.Mock()
    session.get.return_value = mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com',
        reason='OK',
        headers={},
        encoding='utf-8'
    )
    result = url_opener('http://test.com', {'method': 'get', 'session': session})
    assert result == 'test response'

    # Test with non-200 status code
    mocker.patch('requests.get', return_value=mocker.Mock(
        status_code=404,
        text='not found',
        url='http://test.com',
        reason='Not Found',
        headers={},
        encoding='utf-8'
    ))
    try:
        url_opener('http://test.com', {'method': 'get'})
        assert False, "Expected HTTPError"
    except HTTPError as e:
        assert e.code == 404

    # Test with urllib when requests is not available
    mocker.patch('requests.get', side_effect=ImportError)
    mocker.patch('urllib.request.urlopen', return_value=mocker.Mock(
        read=lambda: b'test response'
    ))
    result = url_opener('http://test.com', {'method': 'get'})
    assert result.read() == b'test response'

    # Test with data parameter in GET request
    mocker.patch('requests.get', return_value=mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com?key=value',
        reason='OK',
        headers={},
        encoding='utf-8'
    ))
    result = url_opener('http://test.com', {'method': 'get', 'data': {'key': 'value'}})
    assert result == 'test response'

    # Test with data parameter in GET request with existing query string
    mocker.patch('requests.get', return_value=mocker.Mock(
        status_code=200,
        text='test response',
        url='http://test.com?existing=param&key=value',
        reason='OK',
        headers={},
        encoding='utf-8'
    ))
    result = url_opener('http://test.com?existing=param', {'method': 'get', 'data': {'key': 'value'}})
    assert result == 'test response'


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert html is not None

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert html is not None

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert html is not None

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert html is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response is not None

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.read() is not None

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.read() is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #6
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=DEFAULT_TIMEOUT,
            **{}
        )

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"Mocked urllib response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        assert result.read() == b"Mocked urllib response"
        mock_urlopen.assert_called_once_with(
            "http://example.com",
            None,
            timeout=DEFAULT_TIMEOUT
        )

    # Test with data parameter (GET)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response with data"
        mock_get.return_value = mock_response

        data = {'key': 'value'}
        result = url_opener("http://example.com", {'method': 'get', 'data': data})
        assert result == "Mocked response with data"
        mock_get.assert_called_once_with(
            url="http://example.com?key=value",
            timeout=DEFAULT_TIMEOUT,
            **{}
        )

    # Test with data parameter (POST)
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked POST response"
        mock_post.return_value = mock_response

        data = {'key': 'value'}
        result = url_opener("http://example.com", {'method': 'post', 'data': data})
        assert result == "Mocked POST response"
        mock_post.assert_called_once_with(
            url="http://example.com",
            data='key=value',
            timeout=DEFAULT_TIMEOUT,
            **{}
        )

    # Test with HTTPError
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test response'
        mock_get.assert_called_once()

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test response'
        mock_urlopen.return_value = mock_response

        with patch('__main__.HAS_REQUEST', False):
            result = url_opener('http://example.com', {'method': 'get'})
            assert result.read() == b'test response'
            mock_urlopen.assert_called_once()

    # Test HTTPError handling
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_response.url = 'http://example.com'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})

    # Test data handling
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_post.return_value = mock_response

        test_data = {'key': 'value'}
        url_opener('http://example.com', {'method': 'post', 'data': test_data})
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['data'] == test_data

    # Test timeout
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_get.return_value = mock_response

        url_opener('http://example.com', {'method': 'get', 'timeout': 30})
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs['timeout'] == 30


# LLM-generated content at query #8
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Mock requests.get to return a successful response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'Success'
            mock_response.encoding = 'utf-8'
            mock_get.return_value = mock_response

            result = url_opener('http://example.com', {'method': 'get'})
            assert result == 'Success'

        # Test with POST method
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'Post Success'
            mock_post.return_value = mock_response

            result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
            assert result == 'Post Success'

        # Test with non-200 status code
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.reason = 'Not Found'
            mock_response.url = 'http://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener('http://example.com', {'method': 'get'})

    # Test with urllib (fallback)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'Success'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == b'Success'

    # Test with query parameters
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Query Success'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get', 'data': {'param': 'value'}})
        mock_get.assert_called_once_with(url='http://example.com?param=value', timeout=DEFAULT_TIMEOUT)
        assert result == 'Query Success'

    # Test with timeout
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Timeout Success'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get', 'timeout': 30})
        mock_get.assert_called_once_with(url='http://example.com', timeout=30)
        assert result == 'Timeout Success'


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin' in html.lower()

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'httpbin' in html.lower()

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert isinstance(html, str)

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response is not None
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #10
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    import unittest.mock as mock
    with mock.patch('requests.get') as mock_get:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test response'

    # Test with urllib (mocked)
    with mock.patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = mock.Mock()
        mock_response.read.return_value = b'test response'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == b'test response'

    # Test with POST data
    with mock.patch('requests.post') as mock_post:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = 'post response'
        mock_post.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == 'post response'

    # Test with GET data
    with mock.patch('requests.get') as mock_get:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = 'get with data'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get', 'data': {'key': 'value'}})
        assert result == 'get with data'

    # Test with non-200 status code
    with mock.patch('requests.get') as mock_get:
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_response.url = 'http://example.com'
        mock_response.reason = 'Not Found'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})


# LLM-generated content at query #11
#--------------------------

```python
def test_url_opener():
    # Test with requests (mock)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test content'
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test content'

    # Test with urllib (mock)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test content'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test content'

    # Test with data
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test content'
        mock_post.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == 'test content'

    # Test with encoding
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test content'
        mock_response.encoding = 'latin-1'
        mock_get.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get', 'encoding': 'latin-1'})
        assert result == 'test content'

    # Test with HTTPError
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})


# LLM-generated content at query #12
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mock)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test content'
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get'})
        assert result == 'test content'

    # Test with urllib (mock)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test content'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get'})
        assert result == b'test content'

    # Test with POST method and data
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'post content'
        mock_post.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == 'post content'

    # Test with GET method and query parameters
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'get content'
        mock_get.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get', 'data': {'key': 'value'}})
        mock_get.assert_called_once_with(url='http://test.com?key=value', timeout=60)
        assert result == 'get content'

    # Test with HTTPError
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_response.url = 'http://test.com'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener('http://test.com', {'method': 'get'})

    # Test with timeout
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'timeout content'
        mock_get.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get', 'timeout': 30})
        mock_get.assert_called_once_with(url='http://test.com', timeout=30)
        assert result == 'timeout content'


# LLM-generated content at query #13
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request with requests
        kwargs = {'method': 'get'}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)
        assert 'httpbin.org' in result

        # Test POST request with requests
        kwargs = {'method': 'post', 'data': {'key': 'value'}}
        result = url_opener('https://httpbin.org/post', kwargs)
        assert isinstance(result, str)
        assert 'httpbin.org' in result

        # Test with encoding parameter
        kwargs = {'method': 'get', 'encoding': 'utf-8'}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)

        # Test with session parameter
        session = requests.Session()
        kwargs = {'method': 'get', 'session': session}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)

        # Test with invalid URL to check HTTPError
        try:
            url_opener('https://httpbin.org/status/404', kwargs)
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404

    # Test with urllib (fallback)
    # Mock urlopen to avoid actual network calls
    original_urlopen = urllib.request.urlopen

    def mock_urlopen(url, data=None, timeout=DEFAULT_TIMEOUT):
        class MockResponse:
            def __init__(self):
                self.status = 200
                self.reason = 'OK'
            def read(self):
                return b'mock response'
        return MockResponse()

    urllib.request.urlopen = mock_urlopen

    try:
        kwargs = {'method': 'get'}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert result == 'mock response'

        kwargs = {'method': 'post', 'data': {'key': 'value'}}
        result = url_opener('https://httpbin.org/post', kwargs)
        assert result == 'mock response'
    finally:
        urllib.request.urlopen = original_urlopen


# LLM-generated content at query #14
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=DEFAULT_TIMEOUT,
            **{'method': 'get'}
        )

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"Mocked response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        assert result == b"Mocked response"
        mock_urlopen.assert_called_once_with(
            "http://example.com",
            None,
            timeout=DEFAULT_TIMEOUT
        )

    # Test with POST data
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked POST response"
        mock_post.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        assert result == "Mocked POST response"
        mock_post.assert_called_once_with(
            url="http://example.com",
            timeout=DEFAULT_TIMEOUT,
            data='key=value'
        )

    # Test with GET data
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked GET response"
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get', 'data': {'key': 'value'}})
        assert result == "Mocked GET response"
        mock_get.assert_called_once_with(
            url="http://example.com?key=value",
            timeout=DEFAULT_TIMEOUT,
            **{'method': 'get'}
        )

    # Test with HTTPError
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://example.com"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})


# LLM-generated content at query #15
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin' in html.lower()

        # Test POST request with data
        test_data = {'key': 'value'}
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': test_data})
        assert isinstance(html, str)
        assert 'form' in html.lower()

        # Test with encoding parameter
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with custom timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert isinstance(html, str)

        # Test with headers
        headers = {'User-Agent': 'test'}
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'headers': headers})
        assert isinstance(html, str)

        # Test with query parameters in GET
        params = {'param1': 'value1', 'param2': 'value2'}
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'data': params})
        assert isinstance(html, str)
        assert 'param1=value1' in html or 'param1' in html

        # Test error handling for non-200 status
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()

        # Test POST request with data
        test_data = urlencode({'key': 'value'}).encode('utf-8')
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': test_data})
        assert response is not None
        html = response.read().decode('utf-8')
        assert 'form' in html.lower()

        # Test with custom timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response is not None

        # Test with query parameters in GET
        params = urlencode({'param1': 'value1', 'param2': 'value2'})
        response = url_opener(f'https://httpbin.org/get?{params}', {'method': 'get'})
        assert response is not None
        html = response.read().decode('utf-8')
        assert 'param1' in html


# LLM-generated content at query #16
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert html is not None
        assert isinstance(html, str)

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert html is not None
        assert isinstance(html, str)

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert html is not None
        assert isinstance(html, str)

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert html is not None
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None
        assert hasattr(response, 'read')

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response is not None
        assert hasattr(response, 'read')

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response is not None
        assert hasattr(response, 'read')


# LLM-generated content at query #17
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        with requests_mock.Mocker() as m:
            test_url = "http://example.com"
            test_response = "test response"
            m.get(test_url, text=test_response)

            result = url_opener(test_url, {})
            assert result == test_response

            # Test with POST method
            post_url = "http://example.com/post"
            post_data = {"key": "value"}
            m.post(post_url, text="post response")

            result = url_opener(post_url, {"method": "post", "data": post_data})
            assert result == "post response"

            # Test with query parameters
            query_url = "http://example.com/query"
            query_params = {"param1": "value1", "param2": "value2"}
            m.get(f"{query_url}?param1=value1&param2=value2", text="query response")

            result = url_opener(query_url, {"data": query_params})
            assert result == "query response"

            # Test with encoding
            encoding_url = "http://example.com/encoding"
            m.get(encoding_url, text="encoded response")

            result = url_opener(encoding_url, {"encoding": "utf-8"})
            assert result == "encoded response"

            # Test with HTTP error
            error_url = "http://example.com/error"
            m.get(error_url, status_code=404)

            with pytest.raises(HTTPError):
                url_opener(error_url, {})

    # Test with urllib (fallback)
    with patch('urllib.request.urlopen') as mock_urlopen:
        test_url = "http://example.com"
        mock_response = Mock()
        mock_response.read.return_value = b"urllib response"
        mock_urlopen.return_value = mock_response

        result = url_opener(test_url, {})
        assert result == b"urllib response"

        # Test with POST method
        post_url = "http://example.com/post"
        post_data = {"key": "value"}
        encoded_data = urlencode(post_data).encode('utf-8')
        mock_urlopen.return_value = mock_response

        result = url_opener(post_url, {"method": "post", "data": post_data})
        mock_urlopen.assert_called_with(post_url, encoded_data, timeout=DEFAULT_TIMEOUT)

        # Test with query parameters
        query_url = "http://example.com/query"
        query_params = {"param1": "value1", "param2": "value2"}
        expected_url = f"{query_url}?param1=value1&param2=value2"
        mock_urlopen.return_value = mock_response

        result = url_opener(query_url, {"data": query_params})
        mock_urlopen.assert_called_with(expected_url, None, timeout=DEFAULT_TIMEOUT)


# LLM-generated content at query #18
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request with requests
        kwargs = {'method': 'get'}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)

        # Test POST request with requests
        kwargs = {'method': 'post', 'data': {'key': 'value'}}
        result = url_opener('https://httpbin.org/post', kwargs)
        assert isinstance(result, str)

        # Test with encoding
        kwargs = {'method': 'get', 'encoding': 'utf-8'}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)

        # Test with session
        session = requests.Session()
        kwargs = {'method': 'get', 'session': session}
        result = url_opener('https://httpbin.org/get', kwargs)
        assert isinstance(result, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            kwargs = {'method': 'get'}
            url_opener('https://httpbin.org/status/404', kwargs)
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    # Test GET request with urllib
    kwargs = {'method': 'get'}
    result = url_opener('https://httpbin.org/get', kwargs)
    assert isinstance(result, str)

    # Test POST request with urllib
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = url_opener('https://httpbin.org/post', kwargs)
    assert isinstance(result, str)

    # Test with invalid URL (should raise HTTPError)
    try:
        kwargs = {'method': 'get'}
        url_opener('https://httpbin.org/status/404', kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_url_opener():
    # Test with requests library
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.getcode() == 200

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response.getcode() == 200


# LLM-generated content at query #20
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.url = "http://test.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get'})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://test.com",
            timeout=DEFAULT_TIMEOUT,
            **{}
        )

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"Mocked response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get'})
        assert result == b"Mocked response"
        mock_urlopen.assert_called_once_with(
            "http://test.com",
            None,
            timeout=DEFAULT_TIMEOUT
        )

    # Test with data in GET request
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.url = "http://test.com?key=value"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get', 'data': {'key': 'value'}})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://test.com?key=value",
            timeout=DEFAULT_TIMEOUT,
            **{}
        )

    # Test with HTTPError
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.url = "http://test.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://test.com", {'method': 'get'})

    # Test with custom timeout
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.url = "http://test.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get', 'timeout': 10})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://test.com",
            timeout=10,
            **{}
        )


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocking)
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Success"
        mock_get.return_value.encoding = 'utf-8'
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "Success"

    # Test with urllib (mocking)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"Success"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "Success"

    # Test HTTPError with requests
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

    # Test with data parameter
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "Posted"
        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        assert result == "Posted"

    # Test with encoding parameter
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Encoded"
        mock_get.return_value.encoding = 'latin-1'
        result = url_opener("http://example.com", {'method': 'get', 'encoding': 'latin-1'})
        assert result == "Encoded"

    # Test with timeout parameter
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Timeout"
        result = url_opener("http://example.com", {'method': 'get', 'timeout': 30})
        mock_get.assert_called_with(url="http://example.com", timeout=30)

    # Test with allowed_args parameters
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Allowed Args"
        result = url_opener("http://example.com", {'method': 'get', 'headers': {'User-Agent': 'test'}})
        mock_get.assert_called_with(url="http://example.com", timeout=DEFAULT_TIMEOUT, headers={'User-Agent': 'test'})


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener():
    # Test with requests
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert 'httpbin.org' in html

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert 'httpbin.org' in html

        # Test with query parameters
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert 'httpbin.org' in html

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert 'httpbin.org' in html

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert 'httpbin.org' in html

        # Test with invalid URL
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.getcode() == 200

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with query parameters
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response.getcode() == 200

        # Test with invalid URL
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener():
    # Test with requests library
    if HAS_REQUEST:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'test response'
            mock_response.encoding = 'utf-8'
            mock_get.return_value = mock_response

            result = url_opener('http://example.com', {'method': 'get'})
            assert result == 'test response'
            mock_get.assert_called_once_with(
                url='http://example.com',
                timeout=DEFAULT_TIMEOUT
            )

        # Test with data parameter
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'test response'
            mock_post.return_value = mock_response

            test_data = {'key': 'value'}
            result = url_opener('http://example.com', {'method': 'post', 'data': test_data})
            assert result == 'test response'
            mock_post.assert_called_once_with(
                url='http://example.com',
                data=urlencode(test_data).encode('utf-8'),
                timeout=DEFAULT_TIMEOUT
            )

        # Test with HTTPError
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.reason = 'Not Found'
            mock_response.url = 'http://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener('http://example.com', {'method': 'get'})

    # Test with urllib (fallback)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test response'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://example.com', {'method': 'get'})
        assert result == b'test response'
        mock_urlopen.assert_called_once_with(
            'http://example.com',
            None,
            timeout=DEFAULT_TIMEOUT
        )

    # Test with GET method and data
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test response'
        mock_urlopen.return_value = mock_response

        test_data = {'key': 'value'}
        result = url_opener('http://example.com', {'method': 'get', 'data': test_data})
        assert result == b'test response'
        mock_urlopen.assert_called_once_with(
            'http://example.com?key=value',
            None,
            timeout=DEFAULT_TIMEOUT
        )


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        with pytest.raises(HTTPError):
            url_opener('http://httpbin.org/status/404', {'method': 'get'})

        response = url_opener('http://httpbin.org/get', {'method': 'get'})
        assert isinstance(response, str)

        response = url_opener('http://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(response, str)

        response = url_opener('http://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert isinstance(response, str)

        session = requests.Session()
        response = url_opener('http://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(response, str)

    # Test with urllib
    else:
        with pytest.raises(HTTPError):
            url_opener('http://httpbin.org/status/404', {'method': 'get'})

        response = url_opener('http://httpbin.org/get', {'method': 'get'})
        assert isinstance(response, bytes)

        response = url_opener('http://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(response, bytes)

        response = url_opener('http://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert isinstance(response, bytes)


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)

        # Test POST request with data
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.getcode() == 200

        # Test POST request with data
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #6
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.url = "http://test.com"
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get'})
        assert result == "test response"

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get'})
        assert result == "test response"

    # Test HTTPError with requests
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_response.url = "http://test.com"
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://test.com", {'method': 'get'})

    # Test data handling
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.url = "http://test.com"
        mock_post.return_value = mock_response

        test_data = {'key': 'value'}
        url_opener("http://test.com", {'method': 'post', 'data': test_data})
        called_data = mock_post.call_args[1]['data']
        assert called_data == urlencode(test_data).encode('utf-8')

    # Test GET with query parameters
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.url = "http://test.com?key=value"
        mock_get.return_value = mock_response

        url_opener("http://test.com", {'method': 'get', 'data': {'key': 'value'}})
        called_url = mock_get.call_args[1]['url']
        assert called_url == "http://test.com?key=value"


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request with requests
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test POST request with requests
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test with encoding parameter
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404

    # Test with urllib
    else:
        # Test GET request with urllib
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.getcode() == 200

        # Test POST request with urllib
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404


# LLM-generated content at query #8
#--------------------------

```python
def test_url_opener():
    # Test with requests (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {"method": "get"})
        assert result == "test response"

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://example.com", {"method": "get"})
        assert result == "test response"

    # Test HTTPError with requests
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

    # Test data handling with GET
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_get.return_value = mock_response

        url_opener("http://example.com", {"method": "get", "data": {"key": "value"}})
        called_url = mock_get.call_args[1]['url']
        assert "key=value" in called_url

    # Test data handling with POST
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_post.return_value = mock_response

        url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        called_data = mock_post.call_args[1]['data']
        assert called_data == "key=value".encode('utf-8')

    # Test timeout
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_get.return_value = mock_response

        url_opener("http://example.com", {"method": "get", "timeout": 30})
        assert mock_get.call_args[1]['timeout'] == 30

    # Test encoding
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response"
        mock_response.encoding = 'utf-8'
        mock_get.return_value = mock_response

        url_opener("http://example.com", {"method": "get", "encoding": 'latin-1'})
        assert mock_response.encoding == 'latin-1'


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request with requests
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test POST request with requests
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'httpbin.org' in html

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request with urllib
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response.getcode() == 200

        # Test POST request with urllib
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response.getcode() == 200

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #10
#--------------------------

```python
def test_url_opener():
    # Test with requests library
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert html is not None

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert html is not None

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert html is not None

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert html is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #11
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)
        assert 'httpbin' in html.lower()

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'httpbin' in html.lower()

        # Test with query parameters
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert isinstance(html, str)
        assert 'key=value' in html

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with headers
        html = url_opener('https://httpbin.org/headers', {'method': 'get', 'headers': {'User-Agent': 'test'}})
        assert isinstance(html, str)
        assert 'test' in html

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(response, http.client.HTTPResponse)
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(response, http.client.HTTPResponse)
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()

        # Test with query parameters
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'data': {'key': 'value'}})
        assert isinstance(response, http.client.HTTPResponse)
        html = response.read().decode('utf-8')
        assert 'key=value' in html

        # Test with timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert isinstance(response, http.client.HTTPResponse)
        html = response.read().decode('utf-8')
        assert 'httpbin' in html.lower()


# LLM-generated content at query #12
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        # Test GET request with data as dict
        test_url = "http://httpbin.org/get"
        kwargs = {"method": "get", "data": {"key": "value"}}
        result = url_opener(test_url, kwargs)
        assert isinstance(result, str)
        assert "httpbin.org" in result

        # Test POST request with data as string
        test_url = "http://httpbin.org/post"
        kwargs = {"method": "post", "data": "test data"}
        result = url_opener(test_url, kwargs)
        assert isinstance(result, str)
        assert "httpbin.org" in result

        # Test with custom timeout
        test_url = "http://httpbin.org/get"
        kwargs = {"timeout": 10}
        result = url_opener(test_url, kwargs)
        assert isinstance(result, str)

        # Test with encoding
        test_url = "http://httpbin.org/get"
        kwargs = {"encoding": "utf-8"}
        result = url_opener(test_url, kwargs)
        assert isinstance(result, str)

        # Test with session
        session = requests.Session()
        test_url = "http://httpbin.org/get"
        kwargs = {"session": session}
        result = url_opener(test_url, kwargs)
        assert isinstance(result, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            test_url = "http://httpbin.org/status/404"
            kwargs = {}
            url_opener(test_url, kwargs)
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    test_url = "http://httpbin.org/get"
    kwargs = {"method": "get"}
    result = url_opener(test_url, kwargs)
    assert isinstance(result, str)
    assert "httpbin.org" in result.decode('utf-8') if isinstance(result, bytes) else "httpbin.org" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert isinstance(html, str)

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert isinstance(html, str)

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'session': session})
        assert isinstance(html, str)

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert hasattr(response, 'read')

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert hasattr(response, 'read')

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #14
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request with requests
        html = url_opener("https://httpbin.org/get", {"method": "get"})
        assert isinstance(html, str)

        # Test POST request with requests
        html = url_opener("https://httpbin.org/post", {"method": "post", "data": {"key": "value"}})
        assert isinstance(html, str)

        # Test with encoding parameter
        html = url_opener("https://httpbin.org/get", {"method": "get", "encoding": "utf-8"})
        assert isinstance(html, str)

        # Test with session
        session = requests.Session()
        html = url_opener("https://httpbin.org/get", {"method": "get", "session": session})
        assert isinstance(html, str)

        # Test with invalid URL to check HTTPError
        try:
            url_opener("https://httpbin.org/status/404", {"method": "get"})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib (fallback)
    # Mock urlopen to avoid actual network calls
    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda *args, **kwargs: "mocked response"

    html = url_opener("https://httpbin.org/get", {"method": "get"})
    assert html == "mocked response"

    # Restore original urlopen
    urllib.request.urlopen = original_urlopen


# LLM-generated content at query #15
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.encoding = "utf-8"
        mock_get.return_value = mock_response

        result = url_opener("http://example.com", {"method": "get"})
        assert result == "Mocked response"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=DEFAULT_TIMEOUT,
            auth=None,
            data=None,
            headers=None,
            verify=None,
            cert=None,
            config=None,
            hooks=None,
            proxies=None,
            cookies=None
        )

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"Mocked urllib response"
        mock_urlopen.return_value = mock_response

        with patch('builtins.HAS_REQUEST', False):
            result = url_opener("http://example.com", {"method": "get"})
            mock_urlopen.assert_called_once_with(
                "http://example.com",
                None,
                timeout=DEFAULT_TIMEOUT
            )

    # Test HTTPError with requests
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

    # Test GET with data
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.encoding = "utf-8"
        mock_get.return_value = mock_response

        url_opener("http://example.com", {"method": "get", "data": {"key": "value"}})
        mock_get.assert_called_once_with(
            url="http://example.com?key=value",
            timeout=DEFAULT_TIMEOUT,
            auth=None,
            data=None,
            headers=None,
            verify=None,
            cert=None,
            config=None,
            hooks=None,
            proxies=None,
            cookies=None
        )

    # Test POST with data
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Mocked response"
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.encoding = "utf-8"
        mock_post.return_value = mock_response

        url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        mock_post.assert_called_once_with(
            url="http://example.com",
            timeout=DEFAULT_TIMEOUT,
            auth=None,
            data="key=value",
            headers=None,
            verify=None,
            cert=None,
            config=None,
            hooks=None,
            proxies=None,
            cookies=None
        )


# LLM-generated content at query #16
#--------------------------

```python
def test_url_opener():
    # Test with requests library (mocked)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_response.encoding = 'utf-8'
        mock_response.url = 'http://test.com'
        mock_response.reason = 'OK'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get'})
        assert result == 'test response'

    # Test with urllib (mocked)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b'test response'
        mock_urlopen.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get'})
        assert result == b'test response'

    # Test HTTPError with requests
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = 'http://test.com'
        mock_response.reason = 'Not Found'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener('http://test.com', {'method': 'get'})

    # Test with data parameter (GET)
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_response.encoding = 'utf-8'
        mock_response.url = 'http://test.com?param1=value1&param2=value2'
        mock_response.reason = 'OK'
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'get', 'data': {'param1': 'value1', 'param2': 'value2'}})
        assert result == 'test response'

    # Test with data parameter (POST)
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test response'
        mock_response.encoding = 'utf-8'
        mock_response.url = 'http://test.com'
        mock_response.reason = 'OK'
        mock_response.headers = {}
        mock_post.return_value = mock_response

        result = url_opener('http://test.com', {'method': 'post', 'data': {'param1': 'value1', 'param2': 'value2'}})
        assert result == 'test response'


# LLM-generated content at query #17
#--------------------------

```python
def test_url_opener():
    # Test with requests library
    if HAS_REQUEST:
        # Test GET request
        html = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert html is not None

        # Test POST request
        html = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert html is not None

        # Test with encoding
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'encoding': 'utf-8'})
        assert html is not None

        # Test with custom timeout
        html = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert html is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass

    # Test with urllib
    else:
        # Test GET request
        response = url_opener('https://httpbin.org/get', {'method': 'get'})
        assert response is not None

        # Test POST request
        response = url_opener('https://httpbin.org/post', {'method': 'post', 'data': {'key': 'value'}})
        assert response is not None

        # Test with custom timeout
        response = url_opener('https://httpbin.org/get', {'method': 'get', 'timeout': 10})
        assert response is not None

        # Test with invalid URL (should raise HTTPError)
        try:
            url_opener('https://httpbin.org/status/404', {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError:
            pass


# LLM-generated content at query #18
#--------------------------

```python
def test_url_opener():
    # Test with requests (if available)
    if HAS_REQUEST:
        # Mock requests.get to return a successful response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "Mocked response"
            mock_response.encoding = 'utf-8'
            mock_response.reason = "OK"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "Mocked response"

            # Test with encoding parameter
            mock_response.encoding = 'latin-1'
            result = url_opener("http://example.com", {'method': 'get', 'encoding': 'latin-1'})
            assert result == "Mocked response"

            # Test with non-200 status code
            mock_response.status_code = 404
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test with urllib (fallback)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"Mocked urllib response"
        mock_urlopen.return_value = mock_response

        result = url_opener("http://example.com", {'method': 'get'})
        assert result == b"Mocked urllib response"

    # Test _query function with GET method and data
    url, data = _query("http://example.com", "get", {'data': {'key': 'value'}})
    assert url == "http://example.com?key=value"
    assert data is None

    # Test _query function with POST method and data
    url, data = _query("http://example.com", "post", {'data': {'key': 'value'}})
    assert url == "http://example.com"
    assert data == b"key=value"

    # Test _query function with existing query string
    url, data = _query("http://example.com?existing=param", "get", {'data': {'key': 'value'}})
    assert url == "http://example.com?existing=param&key=value"
    assert data is None

    # Test _query function with data as string
    url, data = _query("http://example.com", "post", {'data': "raw data"})
    assert url == "http://example.com"
    assert data == b"raw data"


