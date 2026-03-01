####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', save_dir=None, filename='test.txt')
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with custom save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='test.txt')
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Extract filename from URL
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/document.pdf', None)
        result = download('http://example.com/path/to/document.pdf')
        assert 'document.pdf' in result
    
    # Test 4: Remove ?raw=true suffix from GitHub URLs
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/file.txt', None)
        result = download('http://github.com/user/repo/file.txt?raw=true')
        assert 'file.txt' in result
        assert '?raw=true' not in result
    
    # Test 5: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 6: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('tarfile.open') as mock_tar_open:
            mock_retrieve.return_value = (tar_path, None)
            mock_is_tarfile.return_value = True
            mock_tar = Mock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('zipfile.ZipFile') as mock_zip_open:
            mock_retrieve.return_value = (zip_path, None)
            mock_is_zipfile.return_value = True
            mock_zip = Mock()
            mock_zip_open.return_value.__enter__.return_value = mock_zip
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 8: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        unknown_path = os.path.join(tmpdir, 'unknown.rar')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('flutes.worker.download.log') as mock_log:
            mock_retrieve.return_value = (unknown_path, None)
            mock_is_tarfile.return_value = False
            mock_is_zipfile.return_value = False
            
            result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
            mock_log.assert_called_once()
    
    # Test 9: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('flutes.worker.download.tqdm') as mock_tqdm:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        result = download('http://example.com/test.txt', progress=True)
        mock_tqdm.assert_called_once()
        mock_progress.close.assert_called_once()
    
    # Test 10: Custom progress bar function
    mock_bar_fn = Mock()
    mock_bar_instance = Mock()
    mock_bar_fn.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        def hook(count, block_size, total_size):
            pass
        
        mock_retrieve.side_effect = lambda url, filename, reporthook: (filename, None) if reporthook is None else (filename, hook)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar_instance.close.assert_called_once()
    
    # Test 11: Google Drive download
    with patch('flutes.worker.download._download_from_google_drive') as mock_gdrive:
        mock_gdrive.return_value = '/tmp/gdrive_file.txt'
        result = download('https://drive.google.com/file/d/DRIVE_ID/view', filename='gdrive_file.txt')
        mock_gdrive.assert_called_once()
        assert result == '/tmp/gdrive_file.txt'
    
    # Test 12: Google Drive filename extraction
    with patch('flutes.worker.download._download_from_google_drive') as mock_gdrive:
        mock_gdrive.return_value = '/tmp/DRIVE_ID'
        result = download('https://drive.google.com/file/d/DRIVE_ID/view')
        assert 'DRIVE_ID' in result


# LLM-generated content at query #2
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
        assert mock_retrieve.called
    
    # Test 2: Download with custom save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            expected_path = os.path.join(tmpdir, 'custom.txt')
            assert result == expected_path
            assert mock_retrieve.called
    
    # Test 3: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, progress=False)
            assert result == existing_file
            assert not mock_retrieve.called
    
    # Test 4: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar) as mock_open:
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_open.called
                    assert mock_tar.extractall.called
    
    # Test 5: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip) as mock_open:
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_open.called
                    assert mock_zip.extractall.called
    
    # Test 6: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir, progress=False)
            expected_path = os.path.join(tmpdir, 'DRIVE_ID')
            assert result == expected_path
            assert mock_session.get.called
    
    # Test 7: Progress bar with custom bar_fn
    mock_bar = MagicMock()
    mock_bar_instance = MagicMock()
    mock_bar.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        def progress_hook(count, block_size, total_size):
            pass
        
        mock_retrieve.side_effect = lambda url, path, hook: (path, None) and hook(1, 1024, 2048) and hook(2, 1024, 2048)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        assert mock_bar.called
        assert mock_bar_instance.update.called
        assert mock_bar_instance.close.called
    
    # Test 8: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true', progress=False)
        assert result == '/tmp/test.txt'
    
    # Test 9: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'unknown.rar')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    with patch('.log.log') as mock_log:
                        result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True, progress=False)
                        assert mock_log.called
                        assert 'warning' in mock_log.call_args[0] or 'warning' in mock_log.call_args[1].get('level', '')


# LLM-generated content at query #3
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        result = download('http://example.com/test.txt', save_dir=None)
        assert result == '/tmp/test_file.txt'
    
    # Test 2: Download with specified directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.tar.gz'), None)
            
            # Create a mock tarfile
            mock_tar = MagicMock()
            with patch('tarfile.is_tarfile', return_value=True):
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.zip'), None)
            
            # Create a mock zipfile
            mock_zip = MagicMock()
            with patch('zipfile.is_zipfile', return_value=True):
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'existing.txt')
        with open(filepath, 'w') as f:
            f.write('test')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            mock_retrieve.assert_not_called()
            assert result == filepath
    
    # Test 7: Progress bar with tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_tqdm = MagicMock()
            mock_tqdm_instance = MagicMock()
            mock_tqdm.return_value = mock_tqdm_instance
            
            with patch.dict('sys.modules', {'tqdm': MagicMock(tqdm=mock_tqdm)}):
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 8: Custom progress bar function
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_bar_fn = MagicMock()
            mock_bar_instance = MagicMock()
            mock_bar_fn.return_value = mock_bar_instance
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()
    
    # Test 9: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_session.get.return_value = mock_response
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session_class.return_value = mock_session
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        result = download('http://github.com/test.py?raw=true', save_dir=None)
        assert result == '/tmp/test.py'


# LLM-generated content at query #4
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no extraction)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt')
                assert 'test.txt' in result
                mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom save_dir and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/custom/path/custom_name.txt', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt', 
                                 save_dir='/custom/path',
                                 filename='custom_name.txt')
                assert result == '/custom/path/custom_name.txt'
    
    # Test 3: Skip download if file already exists
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('os.path.exists', return_value=True):
            result = download('http://example.com/existing.txt', 
                             save_dir='/tmp',
                             filename='existing.txt')
            mock_retrieve.assert_not_called()
            assert result == '/tmp/existing.txt'
    
    # Test 4: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                mock_tqdm = Mock()
                with patch('tqdm.tqdm', return_value=mock_tqdm):
                    result = download('http://example.com/test.txt', progress=True)
                    mock_tqdm.assert_called()
                    mock_tqdm.close.assert_called()
    
    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/archive.tar.gz', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                mock_tarfile = Mock()
                mock_tarfile.is_tarfile.return_value = True
                with patch('tarfile.open') as mock_open:
                    with patch('tarfile.is_tarfile', return_value=True):
                        result = download('http://example.com/archive.tar.gz', 
                                         extract=True)
                        mock_open.assert_called_once_with('/tmp/archive.tar.gz', 'r')
    
    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/archive.zip', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('zipfile.is_zipfile', return_value=True):
                    mock_zip = Mock()
                    with patch('zipfile.ZipFile', return_value=mock_zip):
                        result = download('http://example.com/archive.zip', 
                                         extract=True)
                        mock_zip.extractall.assert_called_once_with('/tmp')
    
    # Test 7: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.return_value.get.return_value = mock_response
        
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('builtins.open', Mock()):
                    result = download('https://drive.google.com/file/d/DRIVE_ID/view',
                                     filename='drive_file.txt')
                    assert 'drive_file.txt' in result
    
    # Test 8: Google Drive download with confirmation token
    with patch('requests.Session') as mock_session:
        mock_response1 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        mock_response2 = Mock()
        mock_response2.cookies = {}
        mock_response2.iter_content.return_value = [b'chunk1']
        
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = [mock_response1, mock_response2]
        mock_session.return_value = mock_session_instance
        
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('builtins.open', Mock()):
                    result = download('https://drive.google.com/file/d/DRIVE_ID/view')
                    assert mock_session_instance.get.call_count == 2
    
    # Test 9: Custom progress bar function
    mock_bar_fn = Mock()
    mock_bar_instance = Mock()
    mock_bar_fn.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        def urlretrieve_side_effect(url, filename, reporthook=None):
            if reporthook:
                reporthook(1, 1024, 2048)
                reporthook(2, 1024, 2048)
            return (filename, None)
        
        mock_retrieve.side_effect = urlretrieve_side_effect
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt',
                                 bar_fn=mock_bar_fn,
                                 progress=True)
                mock_bar_fn.assert_called_once()
                mock_bar_instance.update.assert_called()
                mock_bar_instance.close.assert_called_once()
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                result = download('https://github.com/user/repo/blob/main/test.py?raw=true')
                assert 'test.py' in result
                assert '?raw=true' not in result


# LLM-generated content at query #5
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = download('http://example.com/test.txt')
            assert result == '/tmp/test.txt'
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom save_dir and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/custom/path/custom.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs') as mock_makedirs:
                result = download('http://example.com/test.txt', save_dir='/custom/path', filename='custom.txt')
                assert result == '/custom/path/custom.txt'
                mock_makedirs.assert_called_once_with('/custom/path', exist_ok=True)
    
    # Test 3: File already exists - skip download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = download('http://example.com/test.txt', save_dir='/tmp')
            mock_retrieve.assert_not_called()
            assert result == '/tmp/test.txt'
    
    # Test 4: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
            assert mock_retrieve.call_args[0][2] is not None  # _progress_hook should be set
    
    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = Mock()
                with patch('tarfile.open') as mock_open:
                    mock_open.return_value.__enter__.return_value = mock_tar
                    result = download('http://example.com/test.tar.gz', extract=True)
                    mock_tar.extractall.assert_called_once_with('/tmp')
    
    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = Mock()
                with patch('zipfile.ZipFile') as mock_zip_class:
                    mock_zip_class.return_value.__enter__.return_value = mock_zip
                    result = download('http://example.com/test.zip', extract=True)
                    mock_zip.extractall.assert_called_once_with('/tmp')
    
    # Test 7: Google Drive download
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_session.get.return_value = mock_response
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('builtins.open', Mock()) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                result = download('https://drive.google.com/file/d/DRIVE_ID/view')
                assert 'DRIVE_ID' in result
    
    # Test 8: Google Drive with confirmation token
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # First response with token
        mock_response1 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        
        # Second response after confirmation
        mock_response2 = Mock()
        mock_response2.cookies = {}
        mock_response2.iter_content.return_value = [b'data']
        
        mock_session.get.side_effect = [mock_response1, mock_response2]
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('builtins.open', Mock()):
                result = download('https://drive.google.com/file/d/DRIVE_ID/view')
                assert mock_session.get.call_count == 2
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = download('https://github.com/user/repo/blob/main/test.py?raw=true')
            assert result == '/tmp/test.py'
    
    # Test 10: Unknown compression type with extract=True
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.unknown', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = False
                with patch('zipfile.is_zipfile') as mock_is_zip:
                    mock_is_zip.return_value = False
                    with patch('.log.log') as mock_log:
                        result = download('http://example.com/test.unknown', extract=True)
                        mock_log.assert_called_once()


# LLM-generated content at query #6
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt', '/tmp')
                assert result == '/tmp/test.txt'
                mock_retrieve.assert_called_once()
    
    # Test 2: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                mock_tqdm = Mock()
                mock_tqdm_instance = Mock()
                mock_tqdm.return_value = mock_tqdm_instance
                with patch('tqdm.tqdm', mock_tqdm):
                    result = download('http://example.com/test.txt', '/tmp', progress=True)
                    assert result == '/tmp/test.txt'
                    mock_tqdm_instance.close.assert_called_once()
    
    # Test 3: File already exists - skip download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt', '/tmp')
                assert result == '/tmp/test.txt'
                mock_retrieve.assert_not_called()
    
    # Test 4: Custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt', '/tmp', filename='custom.txt')
                assert result == '/tmp/custom.txt'
    
    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                with patch('tarfile.is_tarfile') as mock_is_tar:
                    mock_is_tar.return_value = True
                    mock_tar = Mock()
                    with patch('tarfile.open', return_value=mock_tar):
                        result = download('http://example.com/test.tar.gz', '/tmp', extract=True)
                        mock_tar.extractall.assert_called_once_with('/tmp')
    
    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                with patch('zipfile.is_zipfile') as mock_is_zip:
                    mock_is_zip.return_value = True
                    mock_zip = Mock()
                    with patch('zipfile.ZipFile', return_value=mock_zip):
                        result = download('http://example.com/test.zip', '/tmp', extract=True)
                        mock_zip.extractall.assert_called_once_with('/tmp')
    
    # Test 7: Unknown compression type with extract=True
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                with patch('tarfile.is_tarfile', return_value=False):
                    with patch('zipfile.is_zipfile', return_value=False):
                        with patch('log') as mock_log:
                            result = download('http://example.com/test.rar', '/tmp', extract=True)
                            mock_log.assert_called_once()
    
    # Test 8: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                with patch('builtins.open', Mock()):
                    result = download('https://drive.google.com/file/d/12345/view', '/tmp')
                    assert '12345' in result
    
    # Test 9: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('tempfile.gettempdir', return_value='/tmp'):
                result = download('http://example.com/test.txt')
                assert result.startswith('/tmp/')
    
    # Test 10: Custom bar_fn
    mock_bar_fn = Mock()
    mock_bar_instance = Mock()
    mock_bar_fn.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with patch('os.makedirs'):
                result = download('http://example.com/test.txt', '/tmp', progress=True, bar_fn=mock_bar_fn)
                mock_bar_fn.assert_called_once()
                mock_bar_instance.close.assert_called_once()


# LLM-generated content at query #7
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with default parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            mock_bar.total = None
            mock_bar.refresh = Mock()
            mock_bar.update = Mock()
            mock_bar.close = Mock()
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            assert mock_bar.update.called or mock_bar.close.called
    
    # Test 4: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'existing.txt')
        with open(test_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == test_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'archive.tar.gz')
        with open(test_file, 'w') as f:
            f.write('dummy content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile', return_value=True) as mock_is_tar, \
             patch('tarfile.open') as mock_tar_open:
            
            mock_retrieve.return_value = (test_file, None)
            mock_tar = Mock()
            mock_tar.extractall = Mock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'archive.zip')
        with open(test_file, 'w') as f:
            f.write('dummy content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.is_zipfile', return_value=True) as mock_is_zip, \
             patch('zipfile.ZipFile') as mock_zip_open:
            
            mock_retrieve.return_value = (test_file, None)
            mock_zip = Mock()
            mock_zip.extractall = Mock()
            mock_zip_open.return_value.__enter__.return_value = mock_zip
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 8: Temporary directory usage
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        
        result = download('http://example.com/test.txt', save_dir=None)
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: GitHub raw URL filename cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'unknown.rar')
        with open(test_file, 'w') as f:
            f.write('dummy content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile', return_value=False), \
             patch('zipfile.is_zipfile', return_valueFalse), \
             patch('flutes.console.log') as mock_log:
            
            mock_retrieve.return_value = (test_file, None)
            
            result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
            mock_log.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile

    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()

    # Test 2: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                assert result == os.path.join(tmpdir, 'test.txt')
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()

    # Test 3: Download with custom bar_fn
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()

    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir, filename='drive_file.txt')
            assert result == os.path.join(tmpdir, 'drive_file.txt')
            mock_session_instance.get.assert_called()

    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, 'test.tar.gz')
            with open(test_file, 'wb') as f:
                f.write(b'test')
            with patch('tarfile.is_tarfile', return_value=True):
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)

    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, 'test.zip')
            with open(test_file, 'wb') as f:
                f.write(b'test')
            with patch('zipfile.is_zipfile', return_value=True):
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)

    # Test 7: File already exists (skip download)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()

    # Test 8: Use temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result == os.path.join(tempfile.gettempdir(), 'test.txt')
        mock_retrieve.assert_called_once()

    # Test 9: Remove ?raw=true suffix from GitHub URLs
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/user/repo/test.txt?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')

    # Test 10: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.log') as mock_log:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        test_file = os.path.join(tmpdir, 'test.rar')
                        with open(test_file, 'wb') as f:
                            f.write(b'test')
                        result = download('http://example.com/test.rar', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once_with(
                            "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported",
                            "warning"
                        )


# LLM-generated content at query #9
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
        assert mock_retrieve.called
    
    # Test 2: Download with custom save directory and filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Skip download if file already exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            assert not mock_retrieve.called
    
    # Test 4: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('tarfile.open') as mock_tar_open:
            
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.tar.gz'), None)
            mock_is_tarfile.return_value = True
            mock_tar_instance = MagicMock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar_instance
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            assert mock_tar_instance.extractall.called
    
    # Test 5: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('zipfile.ZipFile') as mock_zip_open:
            
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.zip'), None)
            mock_is_zipfile.return_value = True
            mock_zip_instance = MagicMock()
            mock_zip_open.return_value.__enter__.return_value = mock_zip_instance
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            assert mock_zip_instance.extractall.called
    
    # Test 6: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 7: Progress bar with bar_fn
    mock_bar = MagicMock()
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        def urlretrieve_side_effect(url, filename, reporthook=None):
            if reporthook:
                reporthook(1, 1024, 2048)
                reporthook(2, 1024, 2048)
            return (filename, None)
        
        mock_retrieve.side_effect = urlretrieve_side_effect
        result = download('http://example.com/test.txt', progress=True, bar_fn=lambda: mock_bar)
        assert mock_bar.update.called
    
    # Test 8: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true')
        assert result == '/tmp/test.txt'
    
    # Test 9: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('.log.log') as mock_log:
            
            mock_retrieve.return_value = (os.path.join(tmpdir, 'unknown.rar'), None)
            mock_is_tarfile.return_value = False
            mock_is_zipfile.return_value = False
            
            result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
            assert mock_log.called
            assert 'warning' in mock_log.call_args[0] or 'warning' in mock_log.call_args[1]


# LLM-generated content at query #10
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
        assert mock_retrieve.called
    
    # Test 2: Download with specified save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            assert not mock_retrieve.called
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'test.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'test.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'test.rar')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    with patch('flutes.log') as mock_log:
                        result = download('http://example.com/test.rar', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once()
    
    # Test 8: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        mock_bar = MagicMock()
        mock_bar_fn = MagicMock(return_value=mock_bar)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar.close.assert_called_once()
    
    # Test 9: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = MagicMock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        
        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('https://github.com/user/repo/blob/main/test.txt?raw=true')
        assert result == '/tmp/test.txt'


# LLM-generated content at query #11
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no extraction)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        with patch('tqdm.tqdm', return_value=mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.update.assert_called()
                mock_tqdm.close.assert_called()
    
    # Test 4: Download with extraction (tar file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_tar = Mock()
            with patch('tarfile.is_tarfile', return_value=True):
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Download with extraction (zip file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zip = Mock()
            with patch('zipfile.is_zipfile', return_value=True):
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: File already exists (no download)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 7: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.return_value.get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 8: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: Custom progress bar function
    mock_bar = Mock()
    mock_bar_fn = Mock(return_value=mock_bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, bar_fn=mock_bar_fn)
            mock_bar.update.assert_called()
            mock_bar.close.assert_called()
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://github.com/user/repo/raw/main/test.txt?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')


# LLM-generated content at query #12
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 4: Download with custom bar_fn
    mock_bar_fn = Mock()
    mock_bar_instance = Mock()
    mock_bar_fn.return_value = mock_bar_instance
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()
    
    # Test 5: File already exists - skip download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the file first
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock tar file
            tar_path = os.path.join(tmpdir, 'test.tar.gz')
            with tarfile.open(tar_path, 'w:gz') as tar:
                # Add a dummy file to the tar
                dummy_path = os.path.join(tmpdir, 'dummy.txt')
                with open(dummy_path, 'w') as f:
                    f.write('test')
                tar.add(dummy_path, arcname='dummy.txt')
            
            with patch('tarfile.is_tarfile', return_value=True):
                result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                # Check that extraction happened
                assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock zip file
            zip_path = os.path.join(tmpdir, 'test.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.writestr('dummy.txt', 'test content')
            
            with patch('zipfile.is_zipfile', return_value=True):
                result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                # Check that extraction happened
                assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 8: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
    
    # Test 9: GitHub URL with raw=true suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/user/repo/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')
    
    # Test 10: No save_dir (uses temp directory)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
    
    # Test 11: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.download.log') as mock_log:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        result = download('http://example.com/test.rar', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    import urllib.request
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert 'test.txt' in result
        mock_retrieve.assert_called_once()
    
    # Test 2: Download with specified save_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
            mock_retrieve.assert_called_once()
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            mock_bar.total = None
            mock_bar.refresh = Mock()
            mock_bar.update = Mock()
            mock_bar.close = Mock()
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            mock_retrieve.assert_called_once()
            mock_bar.update.assert_called()
            mock_bar.close.assert_called_once()
    
    # Test 6: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('tarfile.open') as mock_tar_open:
            
            mock_retrieve.return_value = (tar_path, None)
            mock_is_tarfile.return_value = True
            mock_tar = Mock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('zipfile.ZipFile') as mock_zip_open:
            
            mock_retrieve.return_value = (zip_path, None)
            mock_is_zipfile.return_value = True
            mock_zip = Mock()
            mock_zip_open.return_value.__enter__.return_value = mock_zip
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 8: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class, \
             patch('os.path.join') as mock_join:
            
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_response = Mock()
            mock_session.get.return_value = mock_response
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            mock_join.return_value = os.path.join(tmpdir, 'gdrive_file.txt')
            
            result = download('https://drive.google.com/file/d/12345/view', save_dir=tmpdir)
            mock_session.get.assert_called()
    
    # Test 9: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true')
        assert 'test.txt' in result
        assert '?raw=true' not in result
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('log') as mock_log:
            
            mock_retrieve.return_value = (os.path.join(tmpdir, 'unknown.rar'), None)
            mock_is_tarfile.return_value = False
            mock_is_zipfile.return_value = False
            
            result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
            mock_log.assert_called_once_with(
                "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported",
                "warning"
            )


# LLM-generated content at query #2
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', save_dir=None)
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with specified directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.tar.gz'), None)
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'archive.zip'), None)
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'existing.txt')
        with open(filepath, 'w') as f:
            f.write('test')
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            mock_retrieve.assert_not_called()
            assert result == filepath
    
    # Test 7: Progress bar with tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_tqdm = Mock()
            mock_tqdm_instance = Mock()
            mock_tqdm.return_value = mock_tqdm_instance
            with patch.dict('sys.modules', {'tqdm': Mock(tqdm=mock_tqdm)}):
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 8: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
    
    # Test 9: Custom progress bar function
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_bar_fn = Mock()
            mock_bar_instance = Mock()
            mock_bar_fn.return_value = mock_bar_instance
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/user/repo/test.txt?raw=true', save_dir=None)
        assert result == '/tmp/test.txt'


# LLM-generated content at query #3
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request

    # Test 1: Download with default parameters (no progress bar)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()

    # Test 2: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()

    # Test 3: Download with existing file (should skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()

    # Test 4: Download from Google Drive
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session.return_value.get.return_value = mock_response
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')

    # Test 5: Download with extraction (tar file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write('content')
                tmp.close()
                tar.add(tmp.name, arcname='file.txt')
                os.unlink(tmp.name)
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'file.txt'))

    # Test 6: Download with extraction (zip file)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write('content')
                tmp.close()
                zipf.write(tmp.name, arcname='file.txt')
                os.unlink(tmp.name)
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'file.txt'))

    # Test 7: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')

    # Test 8: Download with no save_dir (uses temp directory)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        temp_dir = tempfile.gettempdir()
        mock_retrieve.return_value = (os.path.join(temp_dir, 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result.startswith(temp_dir)

    # Test 9: Download GitHub raw URL (removes ?raw=true suffix)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')

    # Test 10: Download with unknown compression type (warning logged)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'unknown.rar'), None)
            with patch('flutes.download.log') as mock_log:
                result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
                mock_log.assert_called_once()


# LLM-generated content at query #4
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with specified save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tarfile.is_tarfile') as mock_is_tarfile, \
             patch('tarfile.open') as mock_tar_open:
            
            mock_retrieve.return_value = (tar_path, None)
            mock_is_tarfile.return_value = True
            mock_tarfile = Mock()
            mock_tar_open.return_value.__enter__.return_value = mock_tarfile
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            mock_tarfile.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('zipfile.is_zipfile') as mock_is_zipfile, \
             patch('zipfile.ZipFile') as mock_zip_open:
            
            mock_retrieve.return_value = (zip_path, None)
            mock_is_zipfile.return_value = True
            mock_zipfile = Mock()
            mock_zip_open.return_value.__enter__.return_value = mock_zipfile
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            mock_zipfile.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Progress bar with tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
             patch('tqdm.tqdm') as mock_tqdm:
            
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_progress = Mock()
            mock_tqdm.return_value = mock_progress
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
            mock_tqdm.assert_called_once()
            mock_progress.close.assert_called_once()
    
    # Test 8: Custom progress bar function
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_bar_fn = Mock(return_value=Mock())
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
    
    # Test 9: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class, \
             patch('builtins.open', Mock()):
            
            mock_session = Mock()
            mock_response = Mock()
            mock_session_class.return_value = mock_session
            
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session.get.return_value = mock_response
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 10: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/user/repo/file.txt?raw=true')
        assert result == '/tmp/test.txt'


# LLM-generated content at query #5
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                assert mock_tqdm.called
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 3: Download with custom bar_fn
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        mock_bar_fn = Mock(return_value=Mock())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
    
    # Test 4: Download with extraction (tar file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('tarfile.is_tarfile', return_value=True):
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Download with extraction (zip file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('zipfile.is_zipfile', return_value=True):
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: File already exists (no download)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, 'existing.txt')
            with open(test_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            mock_retrieve.assert_not_called()
            assert result == test_file
    
    # Test 7: Google Drive download
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_session.get.return_value = mock_response
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session_class.return_value = mock_session
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 8: Google Drive download with confirmation token
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_response1 = Mock()
        mock_response2 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        mock_response2.cookies = {}
        mock_response2.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.get.side_effect = [mock_response1, mock_response2]
        mock_session_class.return_value = mock_session
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert mock_session.get.call_count == 2
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')
    
    # Test 10: No save_dir (uses temp directory)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/tempfile', None)
        with patch('tempfile.gettempdir', return_value='/tmp'):
            result = download('http://example.com/test.txt')
            assert result.startswith('/tmp')
    
    # Test 11: Custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 12: Unknown compression type with extraction
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    result = download('http://example.com/test.rar', save_dir=tmpdir, extract=True)
                    # Should log warning but still return filepath
                    assert result == os.path.join(tmpdir, 'test.rar')


# LLM-generated content at query #6
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                assert result == os.path.join(tmpdir, 'test.txt')
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 3: File already exists - should skip download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 4: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        
        mock_tarfile = Mock()
        mock_tarfile.is_tarfile.return_value = True
        mock_tar_instance = Mock()
        
        with patch('tarfile.is_tarfile', mock_tarfile.is_tarfile):
            with patch('tarfile.open') as mock_tar_open:
                mock_tar_open.return_value.__enter__.return_value = mock_tar_instance
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar_instance.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        
        mock_zipfile = Mock()
        mock_zipfile.is_zipfile.return_value = True
        mock_zip_instance = Mock()
        
        with patch('zipfile.is_zipfile', mock_zipfile.is_zipfile):
            with patch('zipfile.ZipFile') as mock_zip_open:
                mock_zip_open.return_value.__enter__.return_value = mock_zip_instance
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip_instance.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Google Drive download
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
            mock_session.get.assert_called()
    
    # Test 7: Custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 8: No save_dir (uses temp directory)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        with patch('tempfile.gettempdir', return_value='/tmp'):
            result = download('http://example.com/test.txt')
            assert result == '/tmp/test.txt'
    
    # Test 9: GitHub URL with ?raw=true suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/user/repo/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')
    
    # Test 10: Custom bar_fn
    mock_custom_bar = Mock()
    mock_custom_bar_instance = Mock()
    mock_custom_bar.return_value = mock_custom_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        def mock_hook(count, block_size, total_size):
            pass
        
        mock_retrieve.side_effect = lambda url, filename, reporthook=None: (filename, None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_custom_bar)
            mock_custom_bar.assert_called_once()
            mock_custom_bar_instance.close.assert_called_once()


# LLM-generated content at query #7
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.txt'), None)
            result = download('http://example.com/file.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/original.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Create a dummy file in the tar
            dummy_path = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_path, 'w') as f:
                f.write('content')
            tar.add(dummy_path, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr('test.txt', 'content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'test.txt'))
    
    # Test 7: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        mock_bar = Mock()
        mock_bar.return_value = mock_bar
        mock_bar.total = None
        mock_bar.refresh = Mock()
        mock_bar.update = Mock()
        mock_bar.close = Mock()
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        assert mock_bar.called
        mock_bar.update.assert_called()
        mock_bar.close.assert_called()
    
    # Test 8: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_response.stream = True
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir='/tmp')
        assert 'DRIVE_ID' in result
    
    # Test 9: Google Drive with confirmation token
    with patch('requests.Session') as mock_session:
        mock_response1 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        mock_response1.stream = True
        
        mock_response2 = Mock()
        mock_response2.cookies = {}
        mock_response2.iter_content = Mock(return_value=[b'data'])
        mock_response2.stream = True
        
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = [mock_response1, mock_response2]
        mock_session.return_value = mock_session_instance
        
        result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir='/tmp')
        assert mock_session_instance.get.call_count == 2
    
    # Test 10: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/file.txt', None)
        result = download('http://github.com/file.txt?raw=true')
        assert result == '/tmp/file.txt'
    
    # Test 11: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('.log') as mock_log:
            mock_retrieve.return_value = ('/tmp/unknown.rar', None)
            result = download('http://example.com/unknown.rar', save_dir='/tmp', extract=True)
            mock_log.assert_called_with(
                "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported",
                "warning"
            )


# LLM-generated content at query #8
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with specified save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=False)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, progress=False)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                with patch('tarfile.open') as mock_tar_open:
                    mock_tar = Mock()
                    mock_tar_open.return_value.__enter__.return_value = mock_tar
                    
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                with patch('zipfile.ZipFile') as mock_zip_open:
                    mock_zip = Mock()
                    mock_zip_open.return_value.__enter__.return_value = mock_zip
                    
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'unknown.rar')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = False
                with patch('zipfile.is_zipfile') as mock_is_zip:
                    mock_is_zip.return_value = False
                    with patch('flutes.download.log') as mock_log:
                        result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True, progress=False)
                        mock_log.assert_called_once_with(
                            "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported",
                            "warning"
                        )
    
    # Test 8: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        
        with patch.dict('sys.modules', {'tqdm': Mock(tqdm=mock_tqdm)}):
            result = download('http://example.com/test.txt', progress=True)
            mock_tqdm.assert_called_once()
            mock_tqdm_instance.close.assert_called_once()
    
    # Test 9: Custom progress bar function
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar_instance.close.assert_called_once()
    
    # Test 10: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('flutes.download._download_from_google_drive') as mock_gdrive:
            mock_gdrive.return_value = os.path.join(tmpdir, 'gdrive_file.txt')
            
            result = download('https://drive.google.com/file/d/abc123/view', save_dir=tmpdir, progress=False)
            mock_gdrive.assert_called_once()
    
    # Test 11: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/file.txt', None)
        
        result = download('http://github.com/file.txt?raw=true', progress=False)
        assert result == '/tmp/file.txt'
    
    # Test 12: Create directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, 'new_subdir')
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(new_dir, 'test.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=new_dir, progress=False)
            assert os.path.exists(new_dir)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    import urllib.request
    
    # Test 1: Download with default parameters (no extraction)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_bar = Mock()
        mock_bar.return_value = mock_bar
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            assert mock_bar.called
    
    # Test 4: Download from Google Drive
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'test data']
        
        with patch('requests.Session') as mock_session:
            mock_sess = Mock()
            mock_sess.get.return_value = mock_response
            mock_session.return_value = mock_sess
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'test.tar.gz')
        with open(tar_path, 'wb') as f:
            f.write(b'test tar')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = Mock()
                with patch('tarfile.open') as mock_open:
                    mock_open.return_value.__enter__.return_value = mock_tar
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'test.zip')
        with open(zip_path, 'wb') as f:
            f.write(b'test zip')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = Mock()
                with patch('zipfile.ZipFile') as mock_zip_class:
                    mock_zip_class.return_value.__enter__.return_value = mock_zip
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('existing content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 8: Use temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: GitHub URL with raw=true suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')
    
    # Test 10: Unknown compression type with extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        unknown_path = os.path.join(tmpdir, 'test.unknown')
        with open(unknown_path, 'wb') as f:
            f.write(b'test data')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (unknown_path, None)
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = False
                with patch('zipfile.is_zipfile') as mock_is_zip:
                    mock_is_zip.return_value = False
                    with patch('flutes.log') as mock_log:
                        result = download('http://example.com/test.unknown', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once()


# LLM-generated content at query #2
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Basic download with default parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            mock_bar.total = None
            mock_bar.refresh = Mock()
            mock_bar.update = Mock()
            mock_bar.close = Mock()
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 4: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Create a dummy file in the tar
            dummy_path = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_path, 'w') as f:
                f.write('content')
            tar.add(dummy_path, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Create a dummy file in the zip
            dummy_path = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_path, 'w') as f:
                f.write('content')
            zipf.write(dummy_path, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            mock_response.stream = True
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            result = download('https://drive.google.com/file/d/DRIVE_FILE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_FILE_ID' in result
    
    # Test 8: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: GitHub URL with raw=true suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        unknown_file = os.path.join(tmpdir, 'unknown.rar')
        with open(unknown_file, 'w') as f:
            f.write('not a tar or zip')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (unknown_file, None)
            
            with patch('flutes.download.log') as mock_log:
                result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
                mock_log.assert_called_once()


# LLM-generated content at query #3
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no extraction)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar = Mock()
        mock_bar.return_value = mock_bar
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
            assert 'tqdm' in str(type(mock_bar)) or mock_bar.called
    
    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'data']
        mock_session.return_value.get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result.endswith('DRIVE_ID')
    
    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('tarfile.is_tarfile', return_value=True) as mock_is_tar, \
         patch('tarfile.open') as mock_tar_open:
        
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        mock_tar = Mock()
        mock_tar_open.return_value.__enter__.return_value = mock_tar
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
            mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('zipfile.is_zipfile', return_value=True) as mock_is_zip, \
         patch('zipfile.ZipFile') as mock_zip_open:
        
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        mock_zip = Mock()
        mock_zip_open.return_value.__enter__.return_value = mock_zip
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
            mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: File already exists (skip download)
    with patch('os.path.exists', return_value=True) as mock_exists, \
         patch('urllib.request.urlretrieve') as mock_retrieve:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            mock_retrieve.assert_not_called()
            assert result == os.path.join(tmpdir, 'existing.txt')
    
    # Test 8: GitHub URL with raw=true suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')
    
    # Test 9: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'temp.txt'), None)
        result = download('http://example.com/temp.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 10: Custom progress bar function
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()


# LLM-generated content at query #4
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with custom save directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 3: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        result = download('http://example.com/test.txt', filename='custom.txt')
        assert result == '/tmp/custom.txt'
    
    # Test 4: Skip download if file exists
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/test.txt', save_dir='/tmp')
            mock_retrieve.assert_not_called()
            assert result == '/tmp/test.txt'
    
    # Test 5: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        with patch('tqdm.tqdm', return_value=mock_tqdm):
            result = download('http://example.com/test.txt', progress=True)
            mock_tqdm.close.assert_called_once()
    
    # Test 6: Download with custom bar_fn
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar = Mock()
        mock_bar_fn = Mock(return_value=mock_bar)
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar.close.assert_called_once()
    
    # Test 7: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.tar.gz'), None)
            with patch('tarfile.is_tarfile', return_value=True):
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 8: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.zip'), None)
            with patch('zipfile.is_zipfile', return_value=True):
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 9: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('log') as mock_log:
                    result = download('http://example.com/test.rar', save_dir='/tmp', extract=True)
                    mock_log.assert_called_once()
    
    # Test 10: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.return_value.get.return_value = mock_response
        
        result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir='/tmp')
        assert result == '/tmp/DRIVE_ID'
    
    # Test 11: Google Drive download with confirm token
    with patch('requests.Session') as mock_session:
        mock_response1 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        mock_response2 = Mock()
        mock_response2.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_session.return_value.get.side_effect = [mock_response1, mock_response2]
        
        result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir='/tmp')
        assert result == '/tmp/DRIVE_ID'
    
    # Test 12: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('https://github.com/user/repo/blob/main/test.txt?raw=true')
        assert result == '/tmp/test.txt'
    
    # Test 13: Progress hook functionality
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        def side_effect(url, filename, reporthook=None):
            if reporthook:
                reporthook(1, 1024, 2048)
                reporthook(2, 1024, 2048)
            return (filename, None)
        
        mock_retrieve.side_effect = side_effect
        mock_bar = Mock()
        mock_bar_fn = Mock(return_value=mock_bar)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        assert mock_bar.update.called
        mock_bar.close.assert_called_once()


# LLM-generated content at query #5
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with custom save directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 3: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_bar = Mock()
        mock_bar.return_value = mock_bar
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        assert result == '/tmp/test.txt'
    
    # Test 5: Skip download if file exists
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'existing.txt')
            with open(filepath, 'w') as f:
                f.write('test')
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == filepath
            mock_retrieve.assert_not_called()
    
    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'test.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            with patch('tarfile.is_tarfile', return_value=True):
                with patch('tarfile.open') as mock_tar:
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.return_value.__enter__.assert_called_once()
    
    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'test.zip')
            mock_retrieve.return_value = (zip_path, None)
            with patch('zipfile.is_zipfile', return_value=True):
                with patch('zipfile.ZipFile') as mock_zip:
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.return_value.__enter__.assert_called_once()
    
    # Test 8: Google Drive download
    with patch('requests.Session') as mock_session:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'test data']
            mock_session.return_value.get.return_value = mock_response
            result = download('https://drive.google.com/file/d/12345/view', save_dir=tmpdir)
            assert '12345' in result
    
    # Test 9: Google Drive download with confirmation token
    with patch('requests.Session') as mock_session:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response1 = Mock()
            mock_response1.cookies = {'download_warning_token': 'abc123'}
            mock_response2 = Mock()
            mock_response2.cookies = {}
            mock_response2.iter_content.return_value = [b'test data']
            mock_session.return_value.get.side_effect = [mock_response1, mock_response2]
            result = download('https://drive.google.com/file/d/12345/view', save_dir=tmpdir)
            assert '12345' in result
    
    # Test 10: Remove ?raw=true suffix from GitHub URLs
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true')
        assert result == '/tmp/test.txt'


# LLM-generated content at query #6
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Download with default parameters (no extraction)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_bar = Mock()
        mock_bar.return_value = mock_bar
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            mock_bar.assert_called_once()
    
    # Test 4: Download from Google Drive
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_response = Mock()
        mock_response.cookies = {'download_warning_token': 'test_token'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        
        with patch('requests.Session') as mock_session:
            mock_sess = Mock()
            mock_sess.get.return_value = mock_response
            mock_session.return_value = mock_sess
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'DRIVE_ID')
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'test.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Create a dummy file in the tar
            dummy_file = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_file, 'w') as f:
                f.write('test')
            tar.add(dummy_file, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            dummy_file = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_file, 'w') as f:
                f.write('test')
            zipf.write(dummy_file, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 7: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 8: GitHub URL with raw=true suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')
    
    # Test 9: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        unknown_file = os.path.join(tmpdir, 'test.unknown')
        with open(unknown_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (unknown_file, None)
            with patch('flutes.download.log') as mock_log:
                result = download('http://example.com/test.unknown', save_dir=tmpdir, extract=True)
                mock_log.assert_called_once()


# LLM-generated content at query #7
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    import urllib.request
    
    # Test 1: Download with default parameters (no extraction)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 4: Download with extraction (tar file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_tarfile = Mock()
            mock_tarfile.is_tarfile.return_value = True
            mock_tar = Mock()
            with patch('tarfile.is_tarfile', mock_tarfile.is_tarfile):
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Download with extraction (zip file)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_zipfile = Mock()
            mock_zipfile.is_zipfile.return_value = True
            mock_zip = Mock()
            with patch('zipfile.is_zipfile', mock_zipfile.is_zipfile):
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: File already exists (no download)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 7: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_sess = Mock()
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_sess.get.return_value = mock_response
        mock_session.return_value = mock_sess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 8: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt', save_dir=None)
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/user/repo/test.txt?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 10: Custom progress bar function
    mock_bar_fn = Mock()
    mock_bar_instance = Mock()
    mock_bar_fn.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with default parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_bar = Mock()
            mock_bar.return_value = mock_bar
            mock_bar.total = None
            mock_bar.refresh = Mock()
            mock_bar.update = Mock()
            mock_bar.close = Mock()
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar)
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 4: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'archive.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Create a dummy file in the tar
            dummy_path = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_path, 'w') as f:
                f.write('content')
            tar.add(dummy_path, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            
            result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
            assert result == tar_path
            # Check extraction happened
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'archive.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Create a dummy file in the zip
            dummy_path = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_path, 'w') as f:
                f.write('content')
            zipf.write(dummy_path, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            
            result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
            assert result == zip_path
            # Check extraction happened
            assert os.path.exists(os.path.join(tmpdir, 'dummy.txt'))
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            
            mock_session_instance = Mock()
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            result = download('https://drive.google.com/file/d/DRIVE_FILE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_FILE_ID' in result
    
    # Test 8: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: GitHub URL with ?raw=true suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        unknown_path = os.path.join(tmpdir, 'unknown.rar')
        with open(unknown_path, 'w') as f:
            f.write('not a valid archive')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (unknown_path, None)
            
            with patch('flutes.log') as mock_log:
                result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
                assert result == unknown_path
                mock_log.assert_called_once()


# LLM-generated content at query #9
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom save directory
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('os.makedirs') as mock_makedirs:
        mock_retrieve.return_value = ('/custom/path/test.txt', None)
        result = download('http://example.com/test.txt', save_dir='/custom/path')
        assert result == '/custom/path/test.txt'
        mock_makedirs.assert_called_once_with('/custom/path', exist_ok=True)
    
    # Test 3: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        result = download('http://example.com/test.txt', filename='custom.txt')
        assert result == '/tmp/custom.txt'
    
    # Test 4: Skip download if file exists
    with patch('os.path.exists') as mock_exists, \
         patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_exists.return_value = True
        result = download('http://example.com/test.txt', save_dir='/tmp')
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_not_called()
    
    # Test 5: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('tqdm.tqdm') as mock_tqdm:
        mock_bar = Mock()
        mock_tqdm.return_value = mock_bar
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        result = download('http://example.com/test.txt', progress=True)
        assert result == '/tmp/test.txt'
        mock_tqdm.assert_called_once()
    
    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('tarfile.is_tarfile') as mock_is_tar, \
         patch('tarfile.open') as mock_tar_open:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        mock_is_tar.return_value = True
        mock_tar = Mock()
        mock_tar_open.return_value.__enter__.return_value = mock_tar
        
        result = download('http://example.com/test.tar.gz', extract=True)
        mock_tar.extractall.assert_called_once_with('/tmp')
    
    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('zipfile.is_zipfile') as mock_is_zip, \
         patch('zipfile.ZipFile') as mock_zip_open:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        mock_is_zip.return_value = True
        mock_zip = Mock()
        mock_zip_open.return_value.__enter__.return_value = mock_zip
        
        result = download('http://example.com/test.zip', extract=True)
        mock_zip.extractall.assert_called_once_with('/tmp')
    
    # Test 8: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve, \
         patch('tarfile.is_tarfile') as mock_is_tar, \
         patch('zipfile.is_zipfile') as mock_is_zip, \
         patch('flutes.download.log') as mock_log:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        mock_is_tar.return_value = False
        mock_is_zip.return_value = False
        
        result = download('http://example.com/test.rar', extract=True)
        mock_log.assert_called_once_with(
            "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported",
            "warning"
        )
    
    # Test 9: Google Drive download
    with patch('flutes.download._download_from_google_drive') as mock_gdrive:
        mock_gdrive.return_value = '/tmp/gdrive_file.txt'
        result = download('https://drive.google.com/file/d/12345/view')
        mock_gdrive.assert_called_once()
    
    # Test 10: Custom bar_fn
    custom_bar = Mock()
    custom_bar_fn = Mock(return_value=custom_bar)
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', bar_fn=custom_bar_fn)
        assert result == '/tmp/test.txt'
    
    # Test 11: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/user/repo/test.txt?raw=true')
        assert result == '/tmp/test.txt'
    
    # Test 12: Progress hook functionality
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_bar = Mock()
        mock_bar.total = None
        mock_bar_fn = Mock(return_value=mock_bar)
        
        def urlretrieve_side_effect(url, filename, progress_hook):
            # Simulate progress updates
            progress_hook(1, 1024, 10240)
            progress_hook(2, 1024, 10240)
            return (filename, None)
        
        mock_retrieve.side_effect = urlretrieve_side_effect
        
        result = download('http://example.com/test.txt', bar_fn=mock_bar_fn)
        assert mock_bar.update.called
        mock_bar.close.assert_called_once()


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile

    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert 'test.txt' in result
        mock_retrieve.assert_called_once()

    # Test 2: Download with custom save directory and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/custom/path/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
            mock_retrieve.assert_called_once()

    # Test 3: Skip download if file already exists
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir='/tmp')
            mock_retrieve.assert_not_called()
            assert result == '/tmp/existing.txt'

    # Test 4: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm.return_value = Mock(total=None, refresh=Mock(), update=Mock(), close=Mock())
        with patch('tqdm.tqdm', mock_tqdm):
            result = download('http://example.com/test.txt', progress=True)
            mock_tqdm.assert_called_once()
            mock_retrieve.assert_called_once()

    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/archive.tar.gz', None)
        with patch('tarfile.is_tarfile') as mock_is_tar:
            mock_is_tar.return_value = True
            mock_tar = Mock()
            mock_tar.extractall = Mock()
            with patch('tarfile.open', return_value=mock_tar):
                result = download('http://example.com/archive.tar.gz', extract=True)
                mock_tar.extractall.assert_called_once()

    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/archive.zip', None)
        with patch('zipfile.is_zipfile') as mock_is_zip:
            mock_is_zip.return_value = True
            mock_zip = Mock()
            mock_zip.extractall = Mock()
            with patch('zipfile.ZipFile', return_value=mock_zip):
                result = download('http://example.com/archive.zip', extract=True)
                mock_zip.extractall.assert_called_once()

    # Test 7: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/unknown.rar', None)
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('.log.log') as mock_log:
                    result = download('http://example.com/unknown.rar', extract=True)
                    mock_log.assert_called_once()

    # Test 8: Google Drive download
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_session.return_value.get.return_value = mock_response
        with patch('builtins.open', Mock()):
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', filename='drive_file.txt')
            assert 'drive_file.txt' in result

    # Test 9: Google Drive download with confirm token
    with patch('requests.Session') as mock_session:
        mock_response1 = Mock()
        mock_response1.cookies = {'download_warning_token': 'abc123'}
        mock_response2 = Mock()
        mock_response2.cookies = {}
        mock_response2.iter_content = Mock(return_value=[b'data'])
        mock_session.return_value.get.side_effect = [mock_response1, mock_response2]
        with patch('builtins.open', Mock()):
            result = download('https://drive.google.com/file/d/DRIVE_ID/view')
            assert mock_session.return_value.get.call_count == 2

    # Test 10: Custom progress bar function
    mock_bar = Mock()
    mock_bar.return_value = Mock(update=Mock(), close=Mock())
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        mock_bar.assert_called_once()


# LLM-generated content at query #2
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
        assert mock_retrieve.called
    
    # Test 2: Download with specified directory and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            assert result == os.path.join(tmpdir, 'custom.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Skip download if file already exists
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='existing.txt', progress=False)
            assert result == existing_file
            assert not mock_retrieve.called
    
    # Test 4: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_tar.extractall.called
    
    # Test 5: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_zip.extractall.called
    
    # Test 6: Google Drive download
    with patch('requests.Session') as mock_session:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session.return_value.get.return_value = mock_response
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir, progress=False)
            assert 'DRIVE_ID' in result
    
    # Test 7: Progress bar with custom bar_fn
    mock_bar = Mock()
    mock_bar_instance = Mock()
    mock_bar.return_value = mock_bar_instance
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        def urlretrieve_side_effect(url, filename, reporthook=None):
            if reporthook:
                reporthook(1, 1024, 2048)
                reporthook(2, 1024, 2048)
            return (filename, None)
        
        mock_retrieve.side_effect = urlretrieve_side_effect
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        assert mock_bar.called
        assert mock_bar_instance.update.called
        assert mock_bar_instance.close.called
    
    # Test 8: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true', progress=False)
        assert result == '/tmp/test.txt'
    
    # Test 9: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.log') as mock_log:
                    mock_retrieve.return_value = ('/tmp/test.rar', None)
                    result = download('http://example.com/test.rar', extract=True, progress=False)
                    assert mock_log.called
                    assert 'warning' in mock_log.call_args[0][1]


# LLM-generated content at query #3
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test_file.txt'
    
    # Test 2: Download with custom save directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='test.txt')
            assert result == os.path.join(tmpdir, 'test.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            result = download('http://example.com/test.txt', progress=True)
            mock_tqdm_instance.close.assert_called_once()
    
    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'data1', b'data2']
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = download('https://drive.google.com/file/d/abc123/view', 
                         save_dir='/tmp', filename='drive_file.txt')
        assert 'abc123' in result
    
    # Test 5: File already exists - skip download
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', 
                             save_dir=tmpdir, filename='existing.txt')
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            mock_tar = Mock()
            mock_tar.is_tarfile.return_value = True
            mock_tar_instance = Mock()
            
            with patch('tarfile.is_tarfile', return_value=True):
                with patch('tarfile.open', return_value=mock_tar_instance):
                    result = download('http://example.com/archive.tar.gz', 
                                     save_dir=tmpdir, extract=True)
                    mock_tar_instance.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            mock_zip = Mock()
            mock_zip.is_zipfile.return_value = True
            mock_zip_instance = Mock()
            
            with patch('zipfile.is_zipfile', return_value=True):
                with patch('zipfile.ZipFile', return_value=mock_zip_instance):
                    result = download('http://example.com/archive.zip', 
                                     save_dir=tmpdir, extract=True)
                    mock_zip_instance.extractall.assert_called_once_with(tmpdir)
    
    # Test 8: Custom progress bar function
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        
        result = download('http://example.com/test.txt', 
                         progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar_instance.close.assert_called_once()
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        result = download('http://github.com/user/repo/test.py?raw=true', 
                         save_dir='/tmp', filename=None)
        assert '?raw=true' not in result
    
    # Test 10: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.log') as mock_log:
                    mock_retrieve.return_value = ('/tmp/unknown.rar', None)
                    result = download('http://example.com/unknown.rar', 
                                     save_dir='/tmp', extract=True)
                    mock_log.assert_called_with(
                        "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported", 
                        "warning"
                    )


# LLM-generated content at query #4
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
        assert mock_retrieve.called
    
    # Test 2: Download with custom save directory and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            assert result == os.path.join(tmpdir, 'custom.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Skip download if file already exists
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='existing.txt', progress=False)
            assert result == existing_file
            assert not mock_retrieve.called
    
    # Test 4: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar) as mock_open:
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_open.called
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 5: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip) as mock_zip_open:
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    assert mock_zip_open.called
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Unknown compression type warning
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.download.log') as mock_log:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        file_path = os.path.join(tmpdir, 'unknown.rar')
                        mock_retrieve.return_value = (file_path, None)
                        result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True, progress=False)
                        mock_log.assert_called_with("Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported", "warning")
    
    # Test 7: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('flutes.download.tqdm') as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            mock_retrieve.return_value = ('/tmp/test.txt', None)
            
            def hook(count, block_size, total_size):
                pass
            
            mock_retrieve.side_effect = lambda url, path, reporthook: (path, None) if reporthook is None else (path, None)
            
            result = download('http://example.com/test.txt', progress=True)
            assert mock_tqdm.called
            mock_bar.close.assert_called_once()
    
    # Test 8: Custom progress bar function
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_bar = MagicMock()
        mock_bar_fn = Mock(return_value=mock_bar)
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        def hook(count, block_size, total_size):
            pass
        
        mock_retrieve.side_effect = lambda url, path, reporthook: (path, None) if reporthook is None else (path, None)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar.close.assert_called_once()
    
    # Test 9: Google Drive download
    with patch('flutes.download._download_from_google_drive') as mock_gdrive:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_path = os.path.join(tmpdir, 'gdrive_file.txt')
            mock_gdrive.return_value = expected_path
            result = download('https://drive.google.com/file/d/12345/view', save_dir=tmpdir, progress=False)
            assert result == expected_path
            mock_gdrive.assert_called_once()
    
    # Test 10: Remove GitHub raw suffix from filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_path = os.path.join(tmpdir, 'file.txt')
            mock_retrieve.return_value = (expected_path, None)
            result = download('http://github.com/user/repo/file.txt?raw=true', save_dir=tmpdir, progress=False)
            assert result == expected_path
            args, kwargs = mock_retrieve.call_args
            assert args[0] == 'http://github.com/user/repo/file.txt?raw=true'


# LLM-generated content at query #5
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with custom save directory and filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            assert result == os.path.join(tmpdir, 'custom.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar = Mock()
        mock_bar.return_value = Mock()
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar)
        assert result == '/tmp/test.txt'
    
    # Test 4: File already exists - should skip download
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='existing.txt', progress=False)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = Mock()
                with patch('tarfile.open') as mock_open:
                    mock_open.return_value.__enter__.return_value = mock_tar
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = Mock()
                with patch('zipfile.ZipFile') as mock_zip_class:
                    mock_zip_class.return_value.__enter__.return_value = mock_zip
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Unknown compression type with extract=True
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'unknown.xyz')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                with patch('zipfile.is_zipfile') as mock_is_zip:
                    mock_is_tar.return_value = False
                    mock_is_zip.return_value = False
                    with patch('flutes.download.log') as mock_log:
                        result = download('http://example.com/unknown.xyz', save_dir=tmpdir, extract=True, progress=False)
                        mock_log.assert_called_once_with(
                            "Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported", 
                            "warning"
                        )
    
    # Test 8: Google Drive download
    with patch('flutes.download._download_from_google_drive') as mock_gdrive:
        mock_gdrive.return_value = '/tmp/gdrive_file.txt'
        result = download('https://drive.google.com/file/d/12345/view', progress=False)
        mock_gdrive.assert_called_once()
        assert result == '/tmp/gdrive_file.txt'
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        result = download('http://github.com/user/repo/test.py?raw=true', progress=False)
        assert result == '/tmp/test.py'
    
    # Test 10: Custom progress bar function with kwargs
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn, desc='Downloading')
        mock_bar_fn.assert_called_once()
        assert result == '/tmp/test.txt'


# LLM-generated content at query #6
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=False)
        assert result == '/tmp/test.txt'
    
    # Test 2: Download with specified save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=False)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert os.path.exists(tmpdir)
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt', progress=False)
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, progress=False)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True, progress=False)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True, progress=False)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'unknown.rar')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    with patch('flutes.log') as mock_log:
                        result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True, progress=False)
                        mock_log.assert_called_once_with("Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported", "warning")
    
    # Test 8: Progress bar with tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_bar = MagicMock()
        mock_bar_fn = Mock(return_value=mock_bar)
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            def urlretrieve_side_effect(url, filename, reporthook=None):
                if reporthook:
                    reporthook(1, 1024, 2048)
                    reporthook(2, 1024, 2048)
                return (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_retrieve.side_effect = urlretrieve_side_effect
            
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar_fn.assert_called_once()
            assert mock_bar.update.call_count == 2
            mock_bar.close.assert_called_once()
    
    # Test 9: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = MagicMock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            
            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir, progress=False)
            assert 'DRIVE_ID' in result
    
    # Test 10: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt?raw=true', progress=False)
        assert result == '/tmp/test.txt'


# LLM-generated content at query #7
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    import requests

    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()

    # Test 2: Download with custom save directory and filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
            mock_retrieve.assert_called_once()

    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', progress=True)
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()

    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'data1', b'data2']
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = download('https://drive.google.com/file/d/12345/view', save_dir='/tmp')
        assert result == '/tmp/12345'
        mock_session_instance.get.assert_called()

    # Test 5: File already exists - skip download
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/test.txt', save_dir='/tmp')
            assert result == '/tmp/test.txt'
            mock_retrieve.assert_not_called()

    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('tarfile.is_tarfile') as mock_is_tar:
            with patch('tarfile.open') as mock_tar_open:
                mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
                mock_is_tar.return_value = True
                mock_tar_instance = Mock()
                mock_tar_open.return_value.__enter__.return_value = mock_tar_instance
                
                result = download('http://example.com/test.tar.gz', extract=True)
                assert result == '/tmp/test.tar.gz'
                mock_tar_instance.extractall.assert_called_once()

    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with patch('zipfile.is_zipfile') as mock_is_zip:
            with patch('zipfile.ZipFile') as mock_zip:
                mock_retrieve.return_value = ('/tmp/test.zip', None)
                mock_is_zip.return_value = True
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                
                result = download('http://example.com/test.zip', extract=True)
                assert result == '/tmp/test.zip'
                mock_zip_instance.extractall.assert_called_once()

    # Test 8: Custom progress bar function
    mock_bar = Mock()
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt', bar_fn=mock_bar)
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()

    # Test 9: GitHub URL with raw=true suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://github.com/test.txt?raw=true')
        assert result == '/tmp/test.txt'
        mock_retrieve.assert_called_once()

    # Test 10: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt', save_dir=None)
        assert result == os.path.join(tempfile.gettempdir(), 'test.txt')
        mock_retrieve.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no progress bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance
        with patch('tqdm.tqdm', mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                assert result == os.path.join(tmpdir, 'test.txt')
                mock_tqdm.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 3: Download with custom bar_fn
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_bar_fn = Mock()
        mock_bar_instance = Mock()
        mock_bar_fn.return_value = mock_bar_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()
    
    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir, filename='drive_file.txt')
            expected_path = os.path.join(tmpdir, 'drive_file.txt')
            assert result == expected_path
            assert os.path.exists(expected_path)
    
    # Test 5: File already exists - skip download
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, 'existing.txt')
            with open(existing_file, 'w') as f:
                f.write('content')
            
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 6: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, 'test.tar.gz')
            with patch('tarfile.is_tarfile', return_value=True):
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'test.zip')
            with patch('zipfile.is_zipfile', return_value=True):
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 8: Unknown compression type with extract=True
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.rar', None)
        with patch('tarfile.is_tarfile', return_value=False):
            with patch('zipfile.is_zipfile', return_value=False):
                with patch('flutes.log') as mock_log:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        result = download('http://example.com/test.rar', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once()
    
    # Test 9: No save_dir - use temp directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('tempfile.gettempdir', return_value='/tmp'):
            result = download('http://example.com/test.txt')
            assert result == '/tmp/test.txt'
    
    # Test 10: GitHub URL with ?raw=true suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/user/repo/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')


# LLM-generated content at query #9
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Basic download with temporary directory
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        result = download('http://example.com/test.txt')
        assert 'test.txt' in result
        mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom save directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.txt'), None)
            result = download('http://example.com/file.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.txt')
            mock_retrieve.assert_called_once()
    
    # Test 3: Download with custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/file.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 4: Skip download if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir)
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            tar_path = os.path.join(tmpdir, 'archive.tar.gz')
            mock_retrieve.return_value = (tar_path, None)
            
            with patch('tarfile.is_tarfile') as mock_is_tar:
                mock_is_tar.return_value = True
                mock_tar = MagicMock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/archive.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            zip_path = os.path.join(tmpdir, 'archive.zip')
            mock_retrieve.return_value = (zip_path, None)
            
            with patch('zipfile.is_zipfile') as mock_is_zip:
                mock_is_zip.return_value = True
                mock_zip = MagicMock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/archive.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_response = MagicMock()
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session.return_value.get.return_value = mock_response
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 8: Progress bar with tqdm
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        
        mock_bar = MagicMock()
        mock_bar_fn = MagicMock(return_value=mock_bar)
        
        result = download('http://example.com/test.txt', progress=True, bar_fn=mock_bar_fn)
        mock_bar_fn.assert_called_once()
        mock_bar.close.assert_called_once()
    
    # Test 9: Remove GitHub raw suffix
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/file.txt', None)
        result = download('http://github.com/user/repo/file.txt?raw=true')
        assert 'file.txt' in result
        assert '?raw=true' not in result
    
    # Test 10: Unknown compression type warning
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            file_path = os.path.join(tmpdir, 'unknown.rar')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    with patch('.log.log') as mock_log:
                        result = download('http://example.com/unknown.rar', save_dir=tmpdir, extract=True)
                        mock_log.assert_called_once()


# LLM-generated content at query #10
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Download with default parameters (no progress bar)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            assert mock_retrieve.called
    
    # Test 2: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_tqdm = Mock()
            mock_tqdm_instance = Mock()
            mock_tqdm.return_value = mock_tqdm_instance
            with patch('tqdm.tqdm', mock_tqdm):
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                assert mock_tqdm.called
                assert mock_tqdm_instance.close.called
    
    # Test 3: Download with custom bar_fn
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            mock_bar_fn = Mock()
            mock_bar_instance = Mock()
            mock_bar_fn.return_value = mock_bar_instance
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            assert mock_bar_fn.called
            assert mock_bar_instance.close.called
    
    # Test 4: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'test.tar.gz')
        with open(tar_path, 'wb') as f:
            f.write(b'test content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            with patch('tarfile.is_tarfile', return_value=True):
                mock_tar = Mock()
                with patch('tarfile.open', return_value=mock_tar):
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    assert mock_tar.extractall.called
    
    # Test 5: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'test.zip')
        with open(zip_path, 'wb') as f:
            f.write(b'test content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            with patch('zipfile.is_zipfile', return_value=True):
                mock_zip = Mock()
                with patch('zipfile.ZipFile', return_value=mock_zip):
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    assert mock_zip.extractall.called
    
    # Test 6: File already exists (skip download)
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('existing content')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            result = download('http://example.com/existing.txt', save_dir=tmpdir, filename='existing.txt')
            assert not mock_retrieve.called
            assert result == existing_file
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_session.get.return_value = mock_response
            mock_response.cookies = {}
            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
            mock_session_class.return_value = mock_session
            
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert os.path.exists(result)
    
    # Test 8: Temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        result = download('http://example.com/test.txt')
        assert result.startswith(tempfile.gettempdir())
    
    # Test 9: Custom filename
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'custom.txt'), None)
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 10: GitHub raw URL filename cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'file.py'), None)
            result = download('http://github.com/user/repo/file.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'file.py')


# LLM-generated content at query #11
#--------------------------

```python
def test_download():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    import tarfile
    import zipfile
    
    # Test 1: Download with default parameters (no extraction)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test_file.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with custom filename
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/custom.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, filename='custom.txt')
            assert result == os.path.join(tmpdir, 'custom.txt')
    
    # Test 3: Download with progress bar
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        mock_tqdm = Mock()
        with patch('tqdm.tqdm', return_value=mock_tqdm):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True)
                mock_tqdm.update.assert_called()
                mock_tqdm.close.assert_called_once()
    
    # Test 4: Download from Google Drive
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.cookies = {}
        mock_response.iter_content.return_value = [b'data']
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('https://drive.google.com/file/d/DRIVE_ID/view', save_dir=tmpdir)
            assert 'DRIVE_ID' in result
    
    # Test 5: Extract tar file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.tar.gz', None)
        with patch('tarfile.is_tarfile', return_value=True):
            mock_tar = Mock()
            with patch('tarfile.open', return_value=mock_tar):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = download('http://example.com/test.tar.gz', save_dir=tmpdir, extract=True)
                    mock_tar.extractall.assert_called_once_with(tmpdir)
    
    # Test 6: Extract zip file
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.zip', None)
        with patch('zipfile.is_zipfile', return_value=True):
            mock_zip = Mock()
            with patch('zipfile.ZipFile', return_value=mock_zip):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = download('http://example.com/test.zip', save_dir=tmpdir, extract=True)
                    mock_zip.extractall.assert_called_once_with(tmpdir)
    
    # Test 7: File already exists (skip download)
    with patch('os.path.exists', return_value=True):
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = download('http://example.com/existing.txt', save_dir=tmpdir)
                mock_retrieve.assert_not_called()
                assert result == os.path.join(tmpdir, 'existing.txt')
    
    # Test 8: Use temporary directory when save_dir is None
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with patch('tempfile.gettempdir', return_value='/tmp'):
            result = download('http://example.com/test.txt')
            assert result.startswith('/tmp')
    
    # Test 9: GitHub raw URL filename cleanup
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.py', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://github.com/test.py?raw=true', save_dir=tmpdir)
            assert result == os.path.join(tmpdir, 'test.py')
    
    # Test 10: Custom progress bar function
    mock_bar = Mock()
    mock_bar_fn = Mock(return_value=mock_bar)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = ('/tmp/test.txt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download('http://example.com/test.txt', save_dir=tmpdir, progress=True, bar_fn=mock_bar_fn)
            mock_bar.update.assert_called()
            mock_bar.close.assert_called_once()


# LLM-generated content at query #12
#--------------------------

```python
def test_download():
    import tempfile
    import os
    import tarfile
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    import urllib.request
    
    # Test 1: Download with default parameters (no progress bar)
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            # Mock a simple URL
            url = 'http://example.com/test.txt'
            result = download(url, save_dir=tmpdir, filename='test.txt')
            
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
    
    # Test 2: Download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            # Mock tqdm progress bar
            mock_tqdm = Mock()
            mock_tqdm_instance = Mock()
            mock_tqdm.return_value = mock_tqdm_instance
            mock_tqdm_instance.total = None
            mock_tqdm_instance.refresh = Mock()
            mock_tqdm_instance.update = Mock()
            mock_tqdm_instance.close = Mock()
            
            with patch('tqdm.tqdm', mock_tqdm):
                url = 'http://example.com/test.txt'
                result = download(url, save_dir=tmpdir, filename='test.txt', progress=True)
                
                assert result == os.path.join(tmpdir, 'test.txt')
                mock_retrieve.assert_called_once()
                mock_tqdm_instance.close.assert_called_once()
    
    # Test 3: Download with custom bar_fn
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.txt'), None)
            
            mock_bar_fn = Mock()
            mock_bar_instance = Mock()
            mock_bar_fn.return_value = mock_bar_instance
            mock_bar_instance.total = None
            mock_bar_instance.refresh = Mock()
            mock_bar_instance.update = Mock()
            mock_bar_instance.close = Mock()
            
            url = 'http://example.com/test.txt'
            result = download(url, save_dir=tmpdir, filename='test.txt', 
                            progress=True, bar_fn=mock_bar_fn)
            
            assert result == os.path.join(tmpdir, 'test.txt')
            mock_retrieve.assert_called_once()
            mock_bar_fn.assert_called_once()
            mock_bar_instance.close.assert_called_once()
    
    # Test 4: File already exists - skip download
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy file
        existing_file = os.path.join(tmpdir, 'existing.txt')
        with open(existing_file, 'w') as f:
            f.write('test')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            url = 'http://example.com/existing.txt'
            result = download(url, save_dir=tmpdir, filename='existing.txt')
            
            assert result == existing_file
            mock_retrieve.assert_not_called()
    
    # Test 5: Extract tar file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock tar file
        tar_path = os.path.join(tmpdir, 'test.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Add a dummy file to the tar
            dummy_file = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_file, 'w') as f:
                f.write('test')
            tar.add(dummy_file, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (tar_path, None)
            
            url = 'http://example.com/test.tar.gz'
            result = download(url, save_dir=tmpdir, extract=True)
            
            assert result == tar_path
            # Check that extraction happened
            extracted_file = os.path.join(tmpdir, 'dummy.txt')
            assert os.path.exists(extracted_file)
    
    # Test 6: Extract zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock zip file
        zip_path = os.path.join(tmpdir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add a dummy file to the zip
            dummy_file = os.path.join(tmpdir, 'dummy.txt')
            with open(dummy_file, 'w') as f:
                f.write('test')
            zipf.write(dummy_file, arcname='dummy.txt')
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (zip_path, None)
            
            url = 'http://example.com/test.zip'
            result = download(url, save_dir=tmpdir, extract=True)
            
            assert result == zip_path
            # Check that extraction happened
            extracted_file = os.path.join(tmpdir, 'dummy.txt')
            assert os.path.exists(extracted_file)
    
    # Test 7: Google Drive download
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_sess_instance = Mock()
            mock_session.return_value = mock_sess_instance
            
            # Mock the response
            mock_response = Mock()
            mock_sess_instance.get.return_value = mock_response
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            
            url = 'https://drive.google.com/file/d/12345/view'
            result = download(url, save_dir=tmpdir, filename='gdrive.txt')
            
            assert result == os.path.join(tmpdir, 'gdrive.txt')
            mock_session.assert_called_once()
    
    # Test 8: Google Drive download with progress bar
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.Session') as mock_session:
            mock_sess_instance = Mock()
            mock_session.return_value = mock_sess_instance
            
            mock_response = Mock()
            mock_sess_instance.get.return_value = mock_response
            mock_response.cookies = {}
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            
            mock_bar_fn = Mock()
            mock_bar_instance = Mock()
            mock_bar_fn.return_value = mock_bar_instance
            mock_bar_instance.update = Mock()
            mock_bar_instance.close = Mock()
            
            url = 'https://drive.google.com/file/d/12345/view'
            result = download(url, save_dir=tmpdir, filename='gdrive.txt', 
                            progress=True, bar_fn=mock_bar_fn)
            
            assert result == os.path.join(tmpdir, 'gdrive.txt')
            mock_bar_instance.update.assert_called()
            mock_bar_instance.close.assert_called_once()
    
    # Test 9: No save_dir specified (uses temp directory)
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        mock_retrieve.return_value = (os.path.join(tempfile.gettempdir(), 'test.txt'), None)
        
        url = 'http://example.com/test.txt'
        result = download(url, filename='test.txt')
        
        assert result == os.path.join(tempfile.gettempdir(), 'test.txt')
        mock_retrieve.assert_called_once()
    
    # Test 10: GitHub URL with ?raw=true suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.return_value = (os.path.join(tmpdir, 'test.py'), None)
            
            url = 'https://github.com/user/repo/blob/main/test.py?raw=true'
            result = download(url, save_dir=tmpdir)
            
            # Should remove ?raw=true from filename
            assert result == os.path.join(tmpdir, 'test.py')
            mock_retrieve.assert_called_once()
    
    # Test 11: Unknown compression type with extract=True
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            # Create a non-compressed file
            file_path = os.path.join(tmpdir, 'test.unknown')
            with open(file_path, 'w') as f:
                f.write('test')
            mock_retrieve.return_value = (file_path, None)
            
            with patch('tarfile.is_tarfile', return_value=False):
                with patch('zipfile.is_zipfile', return_value=False):
                    url = 'http://example.com/test.unknown'
                    result = download(url, save_dir=tmpdir, extract=True)
                    
                    assert result == file_path


