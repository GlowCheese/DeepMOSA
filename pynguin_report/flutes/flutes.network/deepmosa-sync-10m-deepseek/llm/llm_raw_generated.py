####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_download_with_default_filename():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=os.path.join(tmpdir, "file.txt")):
                result = download("http://example.com/file.txt", save_dir=tmpdir)
                assert os.path.basename(result) == "file.txt"

def test_download_with_custom_filename():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=os.path.join(tmpdir, "custom.bin")):
                result = download("http://example.com/file.txt", save_dir=tmpdir, filename="custom.bin")
                assert os.path.basename(result) == "custom.bin"

def test_download_google_drive_url():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.cookies.items.return_value = []
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.Session', return_value=mock_session):
            result = download("https://drive.google.com/file/d/abc123/view", save_dir=tmpdir)
            assert os.path.basename(result) == "abc123"

def test_download_existing_file_skips():
    import tempfile
    import os
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "existing.txt")
        with open(filepath, "w") as f:
            f.write("content")
        with patch('flutes.network._download') as mock_download:
            result = download("http://example.com/existing.txt", save_dir=tmpdir)
            assert result == filepath
            mock_download.assert_not_called()

def test_download_with_progress_bar():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=os.path.join(tmpdir, "file.txt")):
                result = download("http://example.com/file.txt", save_dir=tmpdir, progress=True)
                assert os.path.exists(result)

def test_download_with_extraction_tar():
    import tempfile
    import os
    import tarfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        tar_path = os.path.join(tmpdir, "archive.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(b"fake tar content")
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=tar_path):
                with patch('tarfile.is_tarfile', return_value=True):
                    with patch('tarfile.open') as mock_tar_open:
                        mock_tar = MagicMock()
                        mock_tar_open.return_value.__enter__.return_value = mock_tar
                        result = download("http://example.com/archive.tar.gz", save_dir=tmpdir, extract=True)
                        mock_tar.extractall.assert_called_once_with(tmpdir)

def test_download_with_extraction_zip():
    import tempfile
    import os
    import zipfile
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        zip_path = os.path.join(tmpdir, "archive.zip")
        with open(zip_path, "wb") as f:
            f.write(b"fake zip content")
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=zip_path):
                with patch('zipfile.is_zipfile', return_value=True):
                    with patch('zipfile.ZipFile') as mock_zip_open:
                        mock_zip = MagicMock()
                        mock_zip_open.return_value.__enter__.return_value = mock_zip
                        result = download("http://example.com/archive.zip", save_dir=tmpdir, extract=True)
                        mock_zip.extractall.assert_called_once_with(tmpdir)

def test_download_github_raw_url():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.Session', return_value=mock_session):
            with patch('flutes.network._download', return_value=os.path.join(tmpdir, "script.py")):
                result = download("https://github.com/user/repo/raw/main/script.py?raw=true", save_dir=tmpdir)
                assert os.path.basename(result) == "script.py"

def test_download_no_save_dir_uses_temp():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch('requests.Session', return_value=mock_session):
        with patch('flutes.network._download', return_value=os.path.join(tempfile.gettempdir(), "tempfile.bin")):
            result = download("http://example.com/tempfile.bin", save_dir=None)
            assert result.startswith(tempfile.gettempdir())

def test_download_google_drive_with_token():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response1 = MagicMock()
        mock_response1.cookies.items.return_value = [('download_warning_token', 'abc')]
        mock_response2 = MagicMock()
        mock_response2.cookies.items.return_value = []
        mock_response2.iter_content.return_value = [b"data"]
        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_response1, mock_response2]
        with patch('requests.Session', return_value=mock_session):
            result = download("https://drive.google.com/file/d/def456/view", save_dir=tmpdir)
            assert mock_session.get.call_count == 2
            assert os.path.basename(result) == "def456"


# LLM-generated content at query #2
#--------------------------

def test__download_from_google_drive_success():
    mock_response = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    result = _download_from_google_drive('https://drive.google.com/file/d/abc123/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'

def test__download_from_google_drive_with_token():
    mock_cookies = {'download_warning_token': 'xyz'}
    mock_response_with_token = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1'], 'cookies': mock_cookies})()
    mock_response_final = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1'], 'cookies': {}})()
    call_count = 0
    def mock_get(url, params, stream):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_with_token
        return mock_response_final
    mock_session = type('MockSession', (), {'get': mock_get})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    result = _download_from_google_drive('https://drive.google.com/file/d/def456/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert call_count == 2

def test__download_from_google_drive_with_progress():
    mock_response = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    progress_updates = []
    progress_close_called = False
    mock_bar_fn = lambda: type('MockProgress', (), {'update': lambda size: progress_updates.append(size), 'close': lambda: globals().__setitem__('progress_close_called', True)})()
    result = _download_from_google_drive('https://drive.google.com/file/d/ghi789/view', 'file.txt', '/tmp', mock_bar_fn)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert progress_updates == [len(b'chunk1'), len(b'chunk2')]
    assert progress_close_called == True

def test__extract_google_drive_file_id_standard():
    result = _extract_google_drive_file_id('https://drive.google.com/file/d/abc123/view')
    assert result == 'abc123'

def test__extract_google_drive_file_id_with_additional_path():
    result = _extract_google_drive_file_id('https://drive.google.com/file/d/def456/preview/extra')
    assert result == 'def456'

def test__extract_google_drive_file_id_no_trailing_slash():
    result = _extract_google_drive_file_id('https://drive.google.com/file/d/ghi789')
    assert result == 'ghi789'


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_27_evaluates_to_false():
    import requests
    from unittest.mock import Mock, patch, mock_open
    import os
    url = "https://drive.google.com/file/d/12345/view"
    filename = "test.txt"
    path = "/tmp"
    bar_fn = None
    mock_response = Mock(spec=requests.Response)
    mock_response.iter_content.return_value = [b'']
    mock_response.cookies.items.return_value = []
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        with patch('os.path.join', return_value='/tmp/test.txt'):
            with patch('builtins.open', mock_open()):
                result = _download_from_google_drive(url, filename, path, bar_fn)
    assert result == '/tmp/test.txt'


# LLM-generated content at query #4
#--------------------------

def test_download_from_google_drive_success_with_bar():
    mock_bar = lambda: MockProgress()
    mock_response = MockResponse()
    mock_session = MockSession(mock_response)
    original_requests = requests.Session
    requests.Session = lambda: mock_session
    result = _download_from_google_drive('https://drive.google.com/file/d/abc123/view', 'file.txt', '/tmp', mock_bar)
    requests.Session = original_requests
    assert result == '/tmp/file.txt'
    assert mock_session.called_get == 2
    assert mock_response.written_data == b'chunk1chunk2'
    assert mock_response.closed

def test_download_from_google_drive_success_without_bar():
    mock_response = MockResponse()
    mock_session = MockSession(mock_response)
    original_requests = requests.Session
    requests.Session = lambda: mock_session
    result = _download_from_google_drive('https://drive.google.com/d/xyz456', 'data.bin', '/home/user', None)
    requests.Session = original_requests
    assert result == '/home/user/data.bin'
    assert mock_session.called_get == 1
    assert mock_response.written_data == b'chunk1chunk2'
    assert mock_response.closed

def test_download_from_google_drive_no_token():
    mock_response = MockResponse(has_token=False)
    mock_session = MockSession(mock_response)
    original_requests = requests.Session
    requests.Session = lambda: mock_session
    result = _download_from_google_drive('https://drive.google.com/d/def789', 'out.txt', '/var', None)
    requests.Session = original_requests
    assert result == '/var/out.txt'
    assert mock_session.called_get == 1
    assert mock_response.written_data == b'chunk1chunk2'
    assert mock_response.closed

def test_download_from_google_drive_extract_id():
    mock_response = MockResponse()
    mock_session = MockSession(mock_response)
    original_requests = requests.Session
    requests.Session = lambda: mock_session
    result = _download_from_google_drive('https://drive.google.com/file/d/complex_id_123_abc/details', 'test.txt', '/tmp', None)
    requests.Session = original_requests
    assert result == '/tmp/test.txt'
    assert mock_session.last_params['id'] == 'complex_id_123_abc'


# LLM-generated content at query #5
#--------------------------

def test_download_from_google_drive_with_valid_url_and_bar_fn():
    mock_bar = lambda: type('obj', (object,), {'update': lambda self, x: None, 'close': lambda self: None})()
    mock_response = type('obj', (object,), {'iter_content': lambda self, chunk_size: [b'data']})()
    mock_session = type('obj', (object,), {'get': lambda self, url, params, stream: mock_response})()
    original_requests = __import__('requests')
    __import__ = lambda x: type('obj', (object,), {'Session': lambda: mock_session})() if x == 'requests' else original_requests
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', lambda: mock_bar())
    assert result == '/tmp/file.txt'

def test_download_from_google_drive_without_bar_fn():
    mock_response = type('obj', (object,), {'iter_content': lambda self, chunk_size: [b'data']})()
    mock_session = type('obj', (object,), {'get': lambda self, url, params, stream: mock_response})()
    original_requests = __import__('requests')
    __import__ = lambda x: type('obj', (object,), {'Session': lambda: mock_session})() if x == 'requests' else original_requests
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', None)
    assert result == '/tmp/file.txt'

def test_download_from_google_drive_with_token():
    mock_bar = lambda: type('obj', (object,), {'update': lambda self, x: None, 'close': lambda self: None})()
    mock_response_with_token = type('obj', (object,), {'cookies': {'download_warning_token': 'abc'}, 'iter_content': lambda self, chunk_size: [b'data']})()
    mock_session = type('obj', (object,), {'get': lambda self, url, params, stream: mock_response_with_token})()
    original_requests = __import__('requests')
    __import__ = lambda x: type('obj', (object,), {'Session': lambda: mock_session})() if x == 'requests' else original_requests
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', lambda: mock_bar())
    assert result == '/tmp/file.txt'

def test_download_from_google_drive_with_empty_chunks():
    mock_bar = lambda: type('obj', (object,), {'update': lambda self, x: None, 'close': lambda self: None})()
    mock_response = type('obj', (object,), {'iter_content': lambda self, chunk_size: [b'', b'data', b'']})()
    mock_session = type('obj', (object,), {'get': lambda self, url, params, stream: mock_response})()
    original_requests = __import__('requests')
    __import__ = lambda x: type('obj', (object,), {'Session': lambda: mock_session})() if x == 'requests' else original_requests
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', lambda: mock_bar())
    assert result == '/tmp/file.txt'


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_download_with_direct_url():
    url = "https://example.com/file.txt"
    save_dir = "/tmp/test"
    filename = "downloaded.txt"
    result = download(url, save_dir=save_dir, filename=filename, progress=False)
    assert result == os.path.join(save_dir, filename)

def test_download_with_google_drive_url():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test"
    filename = "abc123"
    result = download(url, save_dir=save_dir, filename=None, progress=False)
    assert result == os.path.join(save_dir, filename)

def test_download_with_custom_filename():
    url = "https://example.com/file.txt"
    save_dir = "/tmp/test"
    filename = "custom.txt"
    result = download(url, save_dir=save_dir, filename=filename, progress=False)
    assert result == os.path.join(save_dir, filename)

def test_download_without_save_dir():
    url = "https://example.com/file.txt"
    result = download(url, save_dir=None, filename=None, progress=False)
    assert result.startswith(tempfile.gettempdir())

def test_download_with_progress_bar():
    url = "https://example.com/file.txt"
    save_dir = "/tmp/test"
    filename = "file.txt"
    result = download(url, save_dir=save_dir, filename=filename, progress=True)
    assert result == os.path.join(save_dir, filename)

def test_download_existing_file():
    url = "https://example.com/file.txt"
    save_dir = "/tmp/test"
    filename = "existing.txt"
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("content")
    result = download(url, save_dir=save_dir, filename=filename, progress=False)
    assert result == filepath

def test_download_with_extraction():
    url = "https://example.com/archive.tar.gz"
    save_dir = "/tmp/test"
    filename = "archive.tar.gz"
    result = download(url, save_dir=save_dir, filename=filename, extract=True, progress=False)
    assert result == os.path.join(save_dir, filename)

def test_download_github_raw_url():
    url = "https://github.com/user/repo/raw/main/file.txt?raw=true"
    save_dir = "/tmp/test"
    filename = "file.txt"
    result = download(url, save_dir=save_dir, filename=None, progress=False)
    assert result == os.path.join(save_dir, filename)

def test_download_with_bar_fn():
    class MockBar:
        def __init__(self):
            self.total = None
            self.closed = False
        def update(self, n):
            pass
        def refresh(self):
            pass
        def close(self):
            self.closed = True
    bar_instance = MockBar()
    def bar_fn():
        return bar_instance
    url = "https://example.com/file.txt"
    save_dir = "/tmp/test"
    filename = "file.txt"
    result = download(url, save_dir=save_dir, filename=filename, bar_fn=bar_fn)
    assert result == os.path.join(save_dir, filename)
    assert bar_instance.closed

def test_download_google_drive_with_bar_fn():
    class MockBar:
        def __init__(self):
            self.total = None
            self.closed = False
        def update(self, n):
            pass
        def refresh(self):
            pass
        def close(self):
            self.closed = True
    bar_instance = MockBar()
    def bar_fn():
        return bar_instance
    url = "https://drive.google.com/file/d/xyz789/view"
    save_dir = "/tmp/test"
    filename = "xyz789"
    result = download(url, save_dir=save_dir, filename=None, bar_fn=bar_fn)
    assert result == os.path.join(save_dir, filename)
    assert bar_instance.closed


# LLM-generated content at query #2
#--------------------------

def test_download_with_direct_url_and_default_filename():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "data.txt")

def test_download_with_direct_url_and_custom_filename():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test"
    filename = "custom.txt"
    result = download(url, save_dir=save_dir, filename=filename)
    assert result == os.path.join(save_dir, "custom.txt")

def test_download_with_github_raw_url():
    url = "https://github.com/user/repo/raw/main/file.py?raw=true"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "file.py")

def test_download_with_google_drive_url():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "abc123")

def test_download_with_google_drive_url_and_custom_filename():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test"
    filename = "file.zip"
    result = download(url, save_dir=save_dir, filename=filename)
    assert result == os.path.join(save_dir, "file.zip")

def test_download_with_none_save_dir():
    url = "https://example.com/data.txt"
    result = download(url, save_dir=None)
    assert result.startswith(tempfile.gettempdir())

def test_download_with_existing_file():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test"
    filename = "existing.txt"
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("content")
    result = download(url, save_dir=save_dir, filename=filename)
    assert result == filepath

def test_download_with_progress_bar():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir, progress=True)
    assert result == os.path.join(save_dir, "data.txt")

def test_download_with_custom_bar_fn():
    class MockBar:
        def __init__(self):
            self.total = None
            self.closed = False
        def update(self, n):
            pass
        def close(self):
            self.closed = True
        def refresh(self):
            pass
    mock_bar_instance = MockBar()
    def mock_bar_fn():
        return mock_bar_instance
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir, bar_fn=mock_bar_fn)
    assert result == os.path.join(save_dir, "data.txt")

def test_download_with_extract_tar():
    url = "https://example.com/archive.tar.gz"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir, extract=True)
    assert result == os.path.join(save_dir, "archive.tar.gz")

def test_download_with_extract_zip():
    url = "https://example.com/archive.zip"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir, extract=True)
    assert result == os.path.join(save_dir, "archive.zip")

def test_download_with_unknown_compression_type():
    url = "https://example.com/unknown.rar"
    save_dir = "/tmp/test"
    result = download(url, save_dir=save_dir, extract=True)
    assert result == os.path.join(save_dir, "unknown.rar")


# LLM-generated content at query #3
#--------------------------

def test_download_without_progress_bar():
    url = "http://example.com/file.txt"
    filename = "file.txt"
    path = "/tmp"
    result = _download(url, filename, path, None)
    assert result == os.path.join(path, filename)

def test_download_with_progress_bar():
    class MockProgress:
        def __init__(self):
            self.total = None
            self.updated = False
            self.closed = False
        def refresh(self):
            pass
        def update(self, value):
            self.updated = True
        def close(self):
            self.closed = True
    mock_progress = MockProgress()
    def mock_bar_fn():
        return mock_progress
    url = "http://example.com/file.txt"
    filename = "file.txt"
    path = "/tmp"
    result = _download(url, filename, path, mock_bar_fn)
    assert result == os.path.join(path, filename)
    assert mock_progress.updated == True
    assert mock_progress.closed == True

def test_download_with_progress_bar_and_total_size():
    class MockProgress:
        def __init__(self):
            self.total = None
            self.updated = False
            self.closed = False
        def refresh(self):
            pass
        def update(self, value):
            self.updated = True
        def close(self):
            self.closed = True
    mock_progress = MockProgress()
    def mock_bar_fn():
        return mock_progress
    url = "http://example.com/file.txt"
    filename = "file.txt"
    path = "/tmp"
    original_urlretrieve = urllib.request.urlretrieve
    def mock_urlretrieve(url, filepath, reporthook):
        reporthook(1, 1024, 2048)
        reporthook(2, 1024, 2048)
        return filepath, None
    urllib.request.urlretrieve = mock_urlretrieve
    result = _download(url, filename, path, mock_bar_fn)
    urllib.request.urlretrieve = original_urlretrieve
    assert result == os.path.join(path, filename)
    assert mock_progress.total == 2048
    assert mock_progress.updated == True
    assert mock_progress.closed == True

def test_download_with_progress_bar_and_unknown_total():
    class MockProgress:
        def __init__(self):
            self.total = None
            self.updated = False
            self.closed = False
        def refresh(self):
            pass
        def update(self, value):
            self.updated = True
        def close(self):
            self.closed = True
    mock_progress = MockProgress()
    def mock_bar_fn():
        return mock_progress
    url = "http://example.com/file.txt"
    filename = "file.txt"
    path = "/tmp"
    original_urlretrieve = urllib.request.urlretrieve
    def mock_urlretrieve(url, filepath, reporthook):
        reporthook(1, 1024, -1)
        reporthook(2, 1024, -1)
        return filepath, None
    urllib.request.urlretrieve = mock_urlretrieve
    result = _download(url, filename, path, mock_bar_fn)
    urllib.request.urlretrieve = original_urlretrieve
    assert result == os.path.join(path, filename)
    assert mock_progress.total == None
    assert mock_progress.updated == True
    assert mock_progress.closed == True


# LLM-generated content at query #4
#--------------------------

def test_download_from_google_drive_success():
    mock_response = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda self, url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, *args: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'

def test_download_from_google_drive_with_token():
    mock_cookies = {'download_warning_token': 'abc'}
    mock_response_with_token = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1'], 'cookies': mock_cookies})()
    mock_response_final = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1'], 'cookies': {}})()
    call_count = 0
    def mock_get(url, params, stream):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_with_token
        return mock_response_final
    mock_session = type('MockSession', (), {'get': mock_get})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, *args: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    result = _download_from_google_drive('https://drive.google.com/file/d/67890/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert call_count == 2

def test_download_from_google_drive_with_progress():
    mock_response = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda self, url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.path.join = lambda path, filename: f'{path}/{filename}'
    open_mock = type('OpenMock', (), {'__enter__': lambda self: self, '__exit__': lambda self, *args: None, 'write': lambda self, chunk: None})()
    builtins_open = open
    open = lambda filepath, mode: open_mock
    progress_updates = []
    progress_close_called = False
    mock_bar_fn = lambda: type('MockProgress', (), {'update': lambda self, size: progress_updates.append(size), 'close': lambda self: setattr(self, 'close_called', True)})()
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'file.txt', '/tmp', mock_bar_fn)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert len(progress_updates) == 2
    assert progress_updates[0] == len(b'chunk1')
    assert progress_updates[1] == len(b'chunk2')


# LLM-generated content at query #5
#--------------------------

def test_download_from_google_drive_success():
    mock_response = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    open_mock = lambda filepath, mode: type('MockFile', (), {'write': lambda data: None, '__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None})()
    os.path.join = lambda path, filename: 'test_path/test_file'
    result = _download_from_google_drive('https://drive.google.com/file/d/12345/view', 'test_file', 'test_path', None)
    assert result == 'test_path/test_file'

def test_download_from_google_drive_with_token():
    mock_cookies = {'download_warning_token': 'abc123'}
    mock_response_with_token = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'chunk1'], 'cookies': mock_cookies})()
    mock_session = type('MockSession', (), {'get': lambda url, params, stream: mock_response_with_token})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    open_mock = lambda filepath, mode: type('MockFile', (), {'write': lambda data: None, '__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None})()
    os.path.join = lambda path, filename: 'path/file'
    result = _download_from_google_drive('https://drive.google.com/d/67890', 'file', 'path', None)
    assert result == 'path/file'

def test_download_from_google_drive_with_progress_bar():
    mock_response = type('MockResponse', (), {'iter_content': lambda chunk_size: [b'data'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    open_mock = lambda filepath, mode: type('MockFile', (), {'write': lambda data: None, '__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None})()
    os.path.join = lambda path, filename: 'dest/file'
    mock_bar = type('MockBar', (), {'update': lambda size: None, 'close': lambda: None})()
    bar_fn = lambda: mock_bar
    result = _download_from_google_drive('https://drive.google.com/d/abc123', 'file', 'dest', bar_fn)
    assert result == 'dest/file'

def test_extract_google_drive_file_id():
    url = 'https://drive.google.com/file/d/1a2b3c4d5e/view?usp=sharing'
    result = _extract_google_drive_file_id(url)
    assert result == '1a2b3c4d5e'

def test_extract_google_drive_file_id_no_slash():
    url = 'https://drive.google.com/d/xyz789'
    result = _extract_google_drive_file_id(url)
    assert result == 'xyz789'


# LLM-generated content at query #6
#--------------------------

def test_download_from_google_drive_success():
    mock_response = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda self, url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    mock_open = type('MockOpen', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, data: None})()
    builtins_open = open
    open = lambda filepath, mode: mock_open
    result = _download_from_google_drive('https://drive.google.com/file/d/abc123/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'

def test_download_from_google_drive_with_token():
    mock_response_with_token = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1'], 'cookies': {'download_warning_token': 'yes'}})()
    mock_response_final = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1'], 'cookies': {}})()
    call_count = 0
    def mock_get(url, params, stream):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_with_token
        return mock_response_final
    mock_session = type('MockSession', (), {'get': mock_get})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    mock_open = type('MockOpen', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, data: None})()
    builtins_open = open
    open = lambda filepath, mode: mock_open
    result = _download_from_google_drive('https://drive.google.com/file/d/def456/view', 'file.txt', '/tmp', None)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert call_count == 2

def test_download_from_google_drive_with_progress():
    mock_response = type('MockResponse', (), {'iter_content': lambda self, chunk_size: [b'chunk1', b'chunk2'], 'cookies': {}})()
    mock_session = type('MockSession', (), {'get': lambda self, url, params, stream: mock_response})()
    mock_requests = type('MockRequests', (), {'Session': lambda: mock_session})()
    import sys
    sys.modules['requests'] = mock_requests
    import os
    os.makedirs = lambda path, exist_ok: None
    mock_open = type('MockOpen', (), {'__enter__': lambda self: self, '__exit__': lambda self, exc_type, exc_val, exc_tb: None, 'write': lambda self, data: None})()
    builtins_open = open
    open = lambda filepath, mode: mock_open
    progress_updates = []
    mock_progress = type('MockProgress', (), {'update': lambda self, size: progress_updates.append(size), 'close': lambda self: None})()
    mock_bar_fn = lambda: mock_progress
    result = _download_from_google_drive('https://drive.google.com/file/d/ghi789/view', 'file.txt', '/tmp', mock_bar_fn)
    open = builtins_open
    assert result == '/tmp/file.txt'
    assert progress_updates == [6, 6]


# LLM-generated content at query #7
#--------------------------

def test_download_from_google_drive_with_valid_url_and_bar_fn():
    mock_bar = Mock()
    mock_bar_instance = Mock()
    mock_bar.return_value = mock_bar_instance
    mock_response = Mock()
    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
    mock_response.cookies.items.return_value = []
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    requests.Session = Mock(return_value=mock_session)
    os.path.join = Mock(return_value='/fake/path/file.txt')
    open_mock = mock_open()
    result = _download_from_google_drive('https://drive.google.com/file/d/abc123/view', 'file.txt', '/fake/path', mock_bar)
    requests.Session.assert_called_once()
    mock_session.get.assert_called_once_with('https://docs.google.com/uc?export=download', params={'id': 'abc123'}, stream=True)
    open_mock.assert_called_once_with('/fake/path/file.txt', 'wb')
    handle = open_mock()
    handle.write.assert_any_call(b'chunk1')
    handle.write.assert_any_call(b'chunk2')
    mock_bar_instance.update.assert_any_call(6)
    mock_bar_instance.update.assert_any_call(6)
    mock_bar_instance.close.assert_called_once()
    assert result == '/fake/path/file.txt'

def test_download_from_google_drive_with_token_confirmation():
    mock_bar = Mock()
    mock_bar_instance = Mock()
    mock_bar.return_value = mock_bar_instance
    mock_response1 = Mock()
    mock_response1.cookies.items.return_value = [('download_warning_token', 'confirm_value')]
    mock_response2 = Mock()
    mock_response2.iter_content.return_value = [b'data']
    mock_response2.cookies.items.return_value = []
    mock_session = Mock()
    mock_session.get.side_effect = [mock_response1, mock_response2]
    requests.Session = Mock(return_value=mock_session)
    os.path.join = Mock(return_value='/path/file.txt')
    open_mock = mock_open()
    result = _download_from_google_drive('https://drive.google.com/d/xyz456', 'file.txt', '/path', mock_bar)
    calls = [call('https://docs.google.com/uc?export=download', params={'id': 'xyz456'}, stream=True),
             call('https://docs.google.com/uc?export=download', params={'id': 'xyz456', 'confirm': 'confirm_value'}, stream=True)]
    mock_session.get.assert_has_calls(calls)
    handle = open_mock()
    handle.write.assert_called_once_with(b'data')
    mock_bar_instance.update.assert_called_once_with(4)
    mock_bar_instance.close.assert_called_once()
    assert result == '/path/file.txt'

def test_download_from_google_drive_without_bar_fn():
    mock_response = Mock()
    mock_response.iter_content.return_value = [b'chunk']
    mock_response.cookies.items.return_value = []
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    requests.Session = Mock(return_value=mock_session)
    os.path.join = Mock(return_value='/some/path/file.bin')
    open_mock = mock_open()
    result = _download_from_google_drive('https://drive.google.com/file/d/id789/', 'file.bin', '/some/path', None)
    mock_session.get.assert_called_once_with('https://docs.google.com/uc?export=download', params={'id': 'id789'}, stream=True)
    handle = open_mock()
    handle.write.assert_called_once_with(b'chunk')
    assert result == '/some/path/file.bin'

def test_extract_google_drive_file_id_standard_url():
    result = _extract_google_drive_file_id('https://drive.google.com/file/d/abc123/view')
    assert result == 'abc123'

def test_extract_google_drive_file_id_with_query_params():
    result = _extract_google_drive_file_id('https://drive.google.com/d/xyz456?usp=sharing')
    assert result == 'xyz456'

def test_extract_google_drive_file_id_no_trailing_slash():
    result = _extract_google_drive_file_id('https://drive.google.com/d/fileId')
    assert result == 'fileId'


# LLM-generated content at query #8
#--------------------------

def test_download_from_google_drive_token_present():
    import requests
    from unittest.mock import Mock, patch
    response_mock = Mock()
    response_mock.cookies.items.return_value = [('download_warning_token', 'abc123')]
    result = _get_confirm_token(response_mock)
    assert result == 'abc123'

def test_download_from_google_drive_token_absent():
    import requests
    from unittest.mock import Mock, patch
    response_mock = Mock()
    response_mock.cookies.items.return_value = [('other_cookie', 'value')]
    result = _get_confirm_token(response_mock)
    assert result is None

def test_download_from_google_drive_token_empty_cookies():
    import requests
    from unittest.mock import Mock, patch
    response_mock = Mock()
    response_mock.cookies.items.return_value = []
    result = _get_confirm_token(response_mock)
    assert result is None


# LLM-generated content at query #9
#--------------------------

def test_download_with_direct_url_and_default_filename():
    url = "https://example.com/data.tar.gz"
    save_dir = "/tmp/test"
    result = download(url, save_dir)
    assert result == os.path.join(save_dir, "data.tar.gz")

def test_download_with_direct_url_and_custom_filename():
    url = "https://example.com/data.tar.gz"
    save_dir = "/tmp/test"
    filename = "custom.tar.gz"
    result = download(url, save_dir, filename)
    assert result == os.path.join(save_dir, "custom.tar.gz")

def test_download_with_google_drive_url_and_default_filename():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test"
    result = download(url, save_dir)
    assert result == os.path.join(save_dir, "abc123")

def test_download_with_google_drive_url_and_custom_filename():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test"
    filename = "file.zip"
    result = download(url, save_dir, filename)
    assert result == os.path.join(save_dir, "file.zip")

def test_download_with_github_raw_url_and_default_filename():
    url = "https://github.com/user/repo/raw/main/data.zip?raw=true"
    save_dir = "/tmp/test"
    result = download(url, save_dir)
    assert result == os.path.join(save_dir, "data.zip")

def test_download_without_save_dir_uses_temp_dir():
    url = "https://example.com/file.txt"
    result = download(url)
    assert result.startswith(tempfile.gettempdir())

def test_download_with_existing_file_skips_download():
    url = "https://example.com/existing.txt"
    save_dir = "/tmp/test"
    filename = "existing.txt"
    filepath = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("content")
    result = download(url, save_dir, filename)
    assert result == filepath

def test_download_with_extract_tar_file():
    url = "https://example.com/archive.tar.gz"
    save_dir = "/tmp/test"
    filename = "archive.tar.gz"
    result = download(url, save_dir, filename, extract=True)
    assert result == os.path.join(save_dir, filename)

def test_download_with_extract_zip_file():
    url = "https://example.com/archive.zip"
    save_dir = "/tmp/test"
    filename = "archive.zip"
    result = download(url, save_dir, filename, extract=True)
    assert result == os.path.join(save_dir, filename)

def test_download_with_progress_bar():
    url = "https://example.com/large.bin"
    save_dir = "/tmp/test"
    result = download(url, save_dir, progress=True)
    assert result == os.path.join(save_dir, "large.bin")

def test_download_with_custom_bar_fn():
    class MockBar:
        def __init__(self):
            self.total = None
        def update(self, n):
            pass
        def close(self):
            pass
        def refresh(self):
            pass
    def mock_bar_fn():
        return MockBar()
    url = "https://example.com/file.bin"
    save_dir = "/tmp/test"
    result = download(url, save_dir, bar_fn=mock_bar_fn)
    assert result == os.path.join(save_dir, "file.bin")


# LLM-generated content at query #10
#--------------------------

def test_download_direct_url_with_default_filename():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "data.txt")
    assert os.path.exists(result)

def test_download_direct_url_with_custom_filename():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test_download"
    filename = "custom.txt"
    result = download(url, save_dir=save_dir, filename=filename)
    assert result == os.path.join(save_dir, "custom.txt")
    assert os.path.exists(result)

def test_download_google_drive_url():
    url = "https://drive.google.com/file/d/abc123/view"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "abc123")
    assert os.path.exists(result)

def test_download_github_raw_url():
    url = "https://github.com/user/repo/raw/main/file.txt?raw=true"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir)
    assert result == os.path.join(save_dir, "file.txt")
    assert os.path.exists(result)

def test_download_with_progress_bar():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir, progress=True)
    assert result == os.path.join(save_dir, "data.txt")
    assert os.path.exists(result)

def test_download_with_extraction_tar():
    url = "https://example.com/archive.tar.gz"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir, extract=True)
    assert result == os.path.join(save_dir, "archive.tar.gz")
    assert os.path.exists(result)

def test_download_with_extraction_zip():
    url = "https://example.com/archive.zip"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir, extract=True)
    assert result == os.path.join(save_dir, "archive.zip")
    assert os.path.exists(result)

def test_download_existing_file_skips():
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test_download"
    filepath = os.path.join(save_dir, "data.txt")
    open(filepath, 'w').close()
    result = download(url, save_dir=save_dir)
    assert result == filepath
    assert os.path.exists(result)

def test_download_with_temporary_directory():
    url = "https://example.com/data.txt"
    result = download(url)
    assert os.path.exists(result)
    assert result.startswith(tempfile.gettempdir())

def test_download_with_custom_bar_fn():
    class MockBar:
        def __init__(self):
            self.total = None
            self.count = 0
        def update(self, n):
            self.count += n
        def close(self):
            pass
        def refresh(self):
            pass
    bar_instance = MockBar()
    def bar_fn():
        return bar_instance
    url = "https://example.com/data.txt"
    save_dir = "/tmp/test_download"
    result = download(url, save_dir=save_dir, bar_fn=bar_fn)
    assert result == os.path.join(save_dir, "data.txt")
    assert os.path.exists(result)


