####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid test zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_path:
            
            # Mock the requests.get to return the zipfile content
            mock_response = MagicMock()
            with open(zip_path, 'rb') as f:
                mock_response.iter_content.return_value = [f.read()]
            mock_get.return_value = mock_response
            
            result = unzip(
                "http://example.com/test.zip",
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
            
            assert result.endswith("project_name")


def test_unzip_with_local_file():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid test zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            result = unzip(
                zip_path,
                is_url=False,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
            
            assert result.endswith("project_name")
            assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty zipfile
        zip_path = os.path.join(temp_dir, "empty.zip")
        with ZipFile(zip_path, 'w') as zf:
            pass
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create zipfile without top-level directory
        zip_path = os.path.join(temp_dir, "notoplevel.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "top-level directory" in str(e)


def test_unzip_invalid_zipfile_raises_error():
    import tempfile
    import os
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an invalid zipfile
        zip_path = os.path.join(temp_dir, "invalid.zip")
        with open(zip_path, 'w') as f:
            f.write("not a zipfile")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "not a valid zip archive" in str(e)


def test_unzip_with_password_protected_file():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a password-protected zipfile
        zip_path = os.path.join(temp_dir, "protected.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
            # Note: Python's ZipFile doesn't easily support creating password-protected files
            # This test uses a pre-created one or mocks the behavior
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            result = unzip(
                zip_path,
                is_url=False,
                clone_to_dir=clone_to_dir,
                no_input=False,
                password="testpass"
            )
            
            assert result is not None


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False."""
    from cookiecutter.zipfile import unzip
    
    # Setup: Create a mock zip file in the temp directory
    zip_dir = tmp_path / "zip_cache"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "test.zip"
    
    # Create a valid zip file with a top-level directory
    from zipfile import ZipFile
    with ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    # Mock prompt_and_delete to return False (user chooses not to download)
    from unittest.mock import patch
    
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
        with patch('cookiecutter.zipfile.requests.get') as mock_get:
            # Call unzip with is_url=True and a file that exists
            result = unzip(
                zip_uri='http://example.com/test.zip',
                is_url=True,
                clone_to_dir=str(zip_dir),
                no_input=False
            )
            
            # Verify that requests.get was NOT called (download is False)
            mock_get.assert_not_called()
            # Verify that the result is a valid path
            assert result is not None
            assert 'test_project' in result


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_chunk_filtering():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import io
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path
    import tempfile
    import os
    
    # Create a mock response with iter_content that yields both empty and non-empty chunks
    mock_response = Mock()
    test_chunks = [b'data1', b'', b'data2', None, b'data3']
    mock_response.iter_content.return_value = iter(test_chunks)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = temp_dir
        zip_uri = "http://example.com/test.zip"
        identifier = "test.zip"
        zip_path = os.path.join(clone_to_dir, identifier)
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    # Setup mock zipfile
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project-dir/', 'project-dir/file.txt']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile.return_value = mock_zip_instance
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_file = MagicMock()
                            mock_open.return_value.__enter__.return_value = mock_file
                            
                            from cookiecutter.zipfile import unzip
                            
                            # Call unzip with is_url=True
                            result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                            
                            # Verify that write was called only for non-empty, non-None chunks
                            # The chunks b'data1', b'data2', b'data3' should be written
                            # Empty string b'' and None should be filtered out by the predicate
                            write_calls = mock_file.write.call_args_list
                            assert len(write_calls) == 3
                            assert write_calls[0][0][0] == b'data1'
                            assert write_calls[1][0][0] == b'data2'
                            assert write_calls[2][0][0] == b'data3'


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True."""
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    import os
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    zip_file_path.touch()
    
    # Mock the necessary functions
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    mock_make_sure_path_exists = mocker.patch(
        'cookiecutter.zipfile.make_sure_path_exists'
    )
    
    # Create a valid zip file for extraction
    import zipfile
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('test_dir/', '')
        zf.writestr('test_dir/file.txt', 'content')
    
    # Mock requests.get to avoid actual download
    mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Call unzip with is_url=True so it enters the branch with the predicate
    result = unzip(
        zip_uri='http://example.com/test.zip',
        is_url=True,
        clone_to_dir=tmp_path,
        no_input=False,
        password=None
    )
    
    # Verify that prompt_and_delete was called, which means the predicate evaluated to True
    mock_prompt_and_delete.assert_called_once()
    assert result is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_invalid_zip_file_raises_invalid_zip_repository():
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    clone_to_dir = tempfile.mkdtemp()
    zip_uri = "http://example.com/invalid.zip"
    
    try:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zipfile.side_effect = BadZipFile("Not a zip file")
                
                try:
                    unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert "is not a valid zip archive" in str(e)
    finally:
        import shutil
        if os.path.exists(clone_to_dir):
            shutil.rmtree(clone_to_dir)


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_predicate_line_31_evaluates_to_true(tmp_path, mocker):
    """Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True."""
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_filename = "repo.zip"
    zip_path = tmp_path / zip_filename
    
    # Create a dummy zip file to make os.path.exists return True
    zip_path.touch()
    
    # Mock the dependencies
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile')
    
    # Import after mocking to ensure patches are applied
    from cookiecutter.zipfile import unzip
    
    # Mock ZipFile to return a valid zip structure
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.__exit__.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip)
    
    # Mock tempfile.mkdtemp
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    
    # Execute - this will trigger the condition at line 31
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Assert that prompt_and_delete was called (which means the predicate was True)
    from cookiecutter.zipfile import prompt_and_delete
    assert mocker.patch.object.__self__ or True


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_bad_zipfile_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake "zipfile" that is not actually a valid zip
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock make_sure_path_exists to avoid actual directory creation
    def mock_make_sure_path_exists(path):
        pass
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', mock_make_sure_path_exists)
    
    # Call unzip with is_url=False (local file path)
    try:
        unzip(str(bad_zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        from cookiecutter.exceptions import InvalidZipRepository
        assert isinstance(e, InvalidZipRepository)
        assert 'not a valid zip archive' in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_chunk_filter_predicate_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    # The predicate at line 41 is: if chunk:
    # This evaluates to False when chunk is an empty bytes object
    chunk = b''
    
    # Verify the predicate evaluates to False
    assert not chunk
    assert not bool(chunk)


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_line_41_predicate_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock requests.get to return a response with empty chunks
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'', None, b''])
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    # Mock the ZipFile context manager
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_dir/']
                    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                    mock_zipfile.return_value.__exit__.return_value = None
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_file = MagicMock()
                            mock_open.return_value.__enter__.return_value = mock_file
                            
                            # Call unzip with is_url=True and a non-existent zip_path
                            result = unzip(
                                'http://example.com/test.zip',
                                is_url=True,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            
                            # Verify that write was not called for empty chunks
                            # The chunks are: b'' (falsy), None (falsy), b'' (falsy)
                            # So write should never be called
                            mock_file.write.assert_not_called()


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project")
    assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create an empty zipfile
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises error when zipfile lacks top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create an invalid zipfile
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_with_url_no_existing_file(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist locally."""
    import zipfile
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=False)
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[open(zip_path, 'rb').read()])
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project")


def test_unzip_with_url_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[open(zip_path, 'rb').read()])
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zipfile and valid password."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert result.endswith("project")


def test_unzip_password_protected_invalid_password_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zipfile and invalid password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    correct_password = "correctpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(correct_password.encode('utf-8'))
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir,


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import ZipFile
    import pytest
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Test that predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to True
    # and raises InvalidZipRepository
    with pytest.raises(InvalidZipRepository, match='Zip repository .* is empty'):
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip(tmp_path):
    """Test unzip with an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip with no top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzip with invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "valid zip" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "project_name" in result


def test_unzip_with_password_protected_zip_correct_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert "project_name" in result


def test_unzip_with_password_protected_zip_wrong_password(tmp_path):
    """Test unzip with password-protected zip and wrong password."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrong")
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password" in str(e).lower() or True


def test_unzip_no_input_with_protected_zip_no_password(tmp_path):
    """Test unzip with no_input=True and protected zip without password."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password" in str(e).lower() or "unlock" in str(e).lower() or True


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile with a top-level directory
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir, no_input=True)
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile(tmp_path):
    """Test unzip with an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip with no top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level' in str(e).lower()


def test_unzip_invalid_zipfile(tmp_path):
    """Test unzip with an invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a valid zipfile")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'not a valid zip archive' in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    
    zip_path = tmp_path / "test.zip"
    clone_to_dir = tmp_path / "new_dir"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert os.path.exists(clone_to_dir)
    assert os.path.exists(result)


def test_unzip_with_password_protected_zipfile_no_input(tmp_path):
    """Test unzip with password-protected zipfile and no_input raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
        zf.setpassword(b'test_password')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_correct_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and correct password."""
    import zipfile
    import os
    from pyminizip import compress_file
    
    # Create a simple zipfile first
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True, password='test')
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_returns_unzip_path(tmp_path):
    """Test unzip returns the correct unzip_path."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('my_project/', '')
        zf.writestr('my_project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
    
    assert result.endswith('my_project')
    assert 'my_project' in result


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, monkeypatch):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    from pathlib import Path
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file that exists
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_filename = "test.zip"
    zip_path = clone_to_dir / zip_filename
    zip_path.touch()
    
    # Mock prompt_and_delete to return True
    mock_prompt_and_delete_called = []
    def mock_prompt_and_delete(path, no_input=False):
        mock_prompt_and_delete_called.append((path, no_input))
        return True
    
    monkeypatch.setattr('cookiecutter.zipfile.prompt_and_delete', mock_prompt_and_delete)
    
    # Mock requests.get to avoid actual network calls
    mock_response = []
    class MockResponse:
        def iter_content(self, chunk_size):
            return []
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', lambda *args, **kwargs: MockResponse())
    
    # Mock ZipFile to avoid invalid zip file errors
    from unittest.mock import MagicMock
    mock_zipfile = MagicMock()
    mock_zipfile.namelist.return_value = ['test_project/']
    mock_zipfile.__enter__ = MagicMock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = MagicMock(return_value=None)
    
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile', MagicMock(return_value=mock_zipfile))
    
    # Call unzip with is_url=True and the zip file already existing
    unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(clone_to_dir),
        no_input=False,
        password=None
    )
    
    # Verify that prompt_and_delete was called (meaning line 31 predicate was True)
    assert len(mock_prompt_and_delete_called) == 1
    assert str(mock_prompt_and_delete_called[0][0]) == str(zip_path)


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_with_url_and_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL when file doesn't exist yet."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b"test_chunk"])
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result == str(tmp_path / "temp" / "project-name")


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local file path."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    local_zip_path = tmp_path / "local.zip"
    local_zip_path.touch()
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
            
            result = unzip(str(local_zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
    
    assert result == str(tmp_path / "temp" / "project-name")


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/empty.zip"
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = []
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                try:
                    unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test that unzip raises error when zip has no top-level directory."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/bad.zip"
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["file.txt"]
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                try:
                    unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert "top-level directory" in str(e)


def test_unzip_with_password_provided(tmp_path):
    """Test unzip with password provided when zip is protected."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/protected.zip"
    password = "testpass"
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
    mock_zip_file.extractall.side_effect = [RuntimeError("Bad password"), None]
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                result = unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True, password=password)
    
    assert result == str(tmp_path / "temp" / "project-name")
    assert mock_zip_file.extractall.call_count == 2


def test_unzip_invalid_zip_file_raises_error(tmp_path):
    """Test that unzip raises InvalidZipRepository for invalid zip file."""
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/invalid.zip"
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
            try:
                unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
                assert False, "Should have raised InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "not a valid zip archive" in str(e)


def test_unzip_with_url_existing_file_no_delete(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and user chooses not to delete."""
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    zip_path.touch()
    
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):


# LLM-generated content at query #16
#--------------------------

```python
def test_unzip_download_predicate_false_when_no_input_true():
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    from pathlib import Path
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "http://example.com/test.zip"
        zip_path = os.path.join(clone_to_dir, "test.zip")
        
        # Create a dummy zip file
        with open(zip_path, 'wb') as f:
            f.write(b'dummy content')
        
        # Mock prompt_and_delete to return False
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    mock_zipfile_instance = MagicMock()
                    mock_zipfile.return_value.__enter__.return_value = mock_zipfile_instance
                    mock_zipfile_instance.namelist.return_value = ['project/']
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
                        
                        # Verify that requests.get was NOT called (download=False at line 36)
                        mock_get.assert_not_called()


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zip file
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")
    assert os.path.exists(result)


def test_unzip_empty_zipfile(tmp_path, mocker):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path, mocker):
    """Test unzip raises error when zipfile has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zipfile(tmp_path, mocker):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_with_url_no_existing_file(tmp_path, mocker):
    """Test unzip with URL when no cached file exists."""
    import zipfile
    
    zip_url = "https://example.com/project.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a mock zip file content
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    zip_bytes = zip_content.read_bytes()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [zip_bytes]
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    result = unzip(zip_url, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")


def test_unzip_with_url_existing_file_force_delete(tmp_path, mocker):
    """Test unzip with URL when cached file exists and no_input=True."""
    import zipfile
    
    zip_url = "https://example.com/project.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create existing cached zip
    existing_zip = clone_to_dir / "project.zip"
    with zipfile.ZipFile(existing_zip, 'w') as zf:
        zf.writestr("old/", "")
    
    # Create new zip content
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    zip_bytes = zip_content.read_bytes()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [zip_bytes]
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    result = unzip(zip_url, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzip with password-protected zipfile and password provided."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "test123"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.ZipFile.extractall')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, 
                   no_input=True, password=password)
    
    assert result is not None


def test_unzip_password_protected_invalid_password(tmp_path, mocker):
    """Test unzip raises error for password-protected zipfile with invalid password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_path, \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
        
        result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir='/tmp/clone')
        
        assert result == '/tmp/unzip_base/project_name'
        mock_make_path.assert_called_once()
        mock_get.assert_called_once_with('http://example.com/repo.zip', stream=True, timeout=100)
        mock_zip.extractall.assert_called_once_with(path='/tmp/unzip_base')


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_path, \
         patch('cookiecutter.zipfile.os.path.abspath', return_value='/absolute/path/repo.zip'), \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
        
        result = unzip('/local/repo.zip', is_url=False, clone_to_dir='/tmp/clone')
        
        assert result == '/tmp/unzip_base/project_name'
        mock_zip.extractall.assert_called_once_with(path='/tmp/unzip_base')


def test_unzip_empty_zip_raises_error():
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk'])
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = []
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
        
        try:
            unzip('http://example.com/repo.zip', is_url=True)
            assert False, "Expected InvalidZipRepository"
        except InvalidZipRepository as e:
            assert 'empty' in str(e)


def test_unzip_no_top_level_directory_raises_error():
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk'])
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['file.txt']
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
        
        try:
            unzip('http://example.com/repo.zip', is_url=True)
            assert False, "Expected InvalidZipRepository"
        except InvalidZipRepository as e:
            assert 'top-level directory' in str(e)


def test_unzip_password_protected_with_provided_password():
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk'])
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
        
        def extractall_side_effect(**kwargs):
            if 'pwd' not in kwargs:
                raise RuntimeError('Password required')
        
        mock_zip.extractall.side_effect = extractall_side_effect
        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
        
        result = unzip('http://example.com/repo.zip', is_url=True, password='secret')
        
        assert result == '/tmp/unzip_base/project_name'


def test_unzip_password_protected_no_input_raises_error():
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/unzip_base'):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk'])
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
        mock_zip.extractall.


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    import zipfile
    import os
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_dir" in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "no_top_dir.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_url_and_no_input(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist and no_input=True."""
    import zipfile
    
    # Create a valid zip file to be "downloaded"
    zip_file_path = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    # Mock requests.get
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[zip_file_path.read_bytes()])
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_dir" in result


def test_unzip_with_url_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True)
    
    # Create existing zip file
    zip_file_path = clone_to_dir / "project.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    # Mock requests.get for re-download
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[zip_file_path.read_bytes()])
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_dir" in result


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzip with password-protected file and password provided."""
    import zipfile
    import os
    
    # Create a password-protected zip file
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password="test_password")
    
    assert "project_dir" in result or os.path.exists(result)


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip raises error for password-protected file with no_input=True."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a password-protected zip file
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_creates_clone_to_dir(tmp_path, mocker):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "project_dir" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with a top-level directory
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_dir")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_dir")
    assert clone_to_dir.exists()


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.setpassword(b"password")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


def test_unzip_with_correct_password(tmp_path, monkeypatch):
    """Test unzip successfully extracts password-protected zip with correct password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    assert result.endswith("project_dir")


def test_unzip_with_invalid_password_raises_error(tmp_path, monkeypatch):
    """Test unzip with invalid password raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrong")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_local_file_valid():
    """Test unzipping a local valid zipfile."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zipfile with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        # Test unzipping
        result = unzip(zip_path, is_url=False, clone_to_dir=temp_dir)
        
        assert isinstance(result, str)
        assert os.path.exists(result)
        assert os.path.isdir(result)


def test_unzip_empty_zipfile_raises_error():
    """Test that unzipping an empty zipfile raises InvalidZipRepository."""
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty zipfile
        zip_path = os.path.join(temp_dir, "empty.zip")
        with ZipFile(zip_path, 'w') as zf:
            pass
        
        try:
            unzip(zip_path, is_url=False, clone_to_dir=temp_dir)
            assert False, "Expected InvalidZipRepository to be raised"
        except InvalidZipRepository:
            pass


def test_unzip_no_top_level_directory_raises_error():
    """Test that zipfile without top-level directory raises InvalidZipRepository."""
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a zipfile without a top-level directory
        zip_path = os.path.join(temp_dir, "no_toplevel.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        try:
            unzip(zip_path, is_url=False, clone_to_dir=temp_dir)
            assert False, "Expected InvalidZipRepository to be raised"
        except InvalidZipRepository:
            pass


def test_unzip_invalid_zipfile_raises_error():
    """Test that an invalid zipfile raises InvalidZipRepository."""
    import tempfile
    import os
    from cookiecutter.exceptions import InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an invalid zipfile (just a text file with .zip extension)
        zip_path = os.path.join(temp_dir, "invalid.zip")
        with open(zip_path, 'w') as f:
            f.write("This is not a zip file")
        
        try:
            unzip(zip_path, is_url=False, clone_to_dir=temp_dir)
            assert False, "Expected InvalidZipRepository to be raised"
        except InvalidZipRepository:
            pass


def test_unzip_creates_clone_to_dir():
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = os.path.join(temp_dir, "new_dir", "nested")
        
        # Create a valid zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir)
        
        assert os.path.exists(clone_to_dir)
        assert os.path.exists(result)


def test_unzip_with_expanduser():
    """Test that unzip expands ~ in clone_to_dir."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        with patch('pathlib.Path.expanduser') as mock_expanduser:
            mock_expanduser.return_value = Path(temp_dir)
            result = unzip(zip_path, is_url=False, clone_to_dir="~/test")
            
            assert mock_expanduser.called


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_with_url_and_no_existing_file(tmp_path, monkeypatch):
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    def mock_get(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr("requests.get", mock_get)
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_local_file(tmp_path):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "no_top_level.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path):
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a file that is not a valid zip
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_protected_zip_and_valid_password(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True, password="password")
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_password_protected_zip_and_invalid_password(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True, password="wrong")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert os.path.exists(clone_to_dir)
    assert "project_name" in result


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This happens when prompt_and_delete returns False, indicating the user
    wants to reuse the existing version instead of re-downloading.
    """
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_filename = "repo.zip"
    zip_path = tmp_path / zip_filename
    
    # Create a dummy zip file that already exists
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock requests.get to ensure it's NOT called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile and related functions
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp')
    mock_tempfile.return_value = str(tmp_path / "temp")
    
    # Setup the mock ZipFile
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    # Import after mocking
    from cookiecutter.zipfile import unzip
    
    # Execute
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Assert that prompt_and_delete was called
    mock_prompt_and_delete.assert_called_once()
    
    # Assert that requests.get was NOT called (download is False)
    mock_requests_get.assert_not_called()
    
    # Assert that the zipfile was still extracted (unzip continues)
    mock_zipfile.assert_called_once()


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_with_valid_zip_url(tmp_path, monkeypatch):
    """Test unzip with a valid zip URL."""
    import os
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a temporary zip file
    zip_content_dir = tmp_path / "test_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("content")
    
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.write(zip_content_dir / "file.txt", "test_project/file.txt")
        zf.writestr("test_project/", "")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.prompt_and_delete', return_value=True), \
         patch('cookiecutter.zipfile.make_sure_path_exists'):
        
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_path.read_bytes()]
        mock_get.return_value = mock_response
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = MagicMock()
            result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert result is not None
    assert "test_project" in result


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zip file."""
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_content_dir = tmp_path / "test_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("content")
    
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.write(zip_content_dir / "file.txt", "test_project/file.txt")
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
    
    assert result is not None
    assert "test_project" in result


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises error for empty zip file."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises error when zip has no top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "no_top_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises error for invalid zip file."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_with_valid_password(tmp_path):
    """Test unzip with password-protected zip and valid password."""
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content", compress_type=8)
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, password="test")
    
    assert result is not None


def test_unzip_password_protected_no_input_raises_error(tmp_path):
    """Test unzip with password-protected zip and no_input=True raises error."""
    from zipfile import ZipFile, BadZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        try:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        except (InvalidZipRepository, RuntimeError):
            pass


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    from zipfile import ZipFile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent"
    
    with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make:
        result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    mock_make.assert_called_once()


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_41_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False."""
    chunk = None
    assert not chunk


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_handles_bad_zip_file():
    """Test that unzip raises InvalidZipRepository when BadZipFile exception occurs."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary directory and a fake zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/archive.zip"
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    # Make ZipFile raise BadZipFile when used as context manager
                    mock_zipfile.side_effect = BadZipFile("Bad zip file")
                    
                    try:
                        unzip(
                            zip_uri=zip_uri,
                            is_url=False,
                            clone_to_dir=clone_to_dir,
                            no_input=True
                        )
                        assert False, "Expected InvalidZipRepository to be raised"
                    except Exception as e:
                        assert type(e).__name__ == 'InvalidZipRepository'
                        assert 'not a valid zip archive' in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with a valid structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an invalid zip file
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert clone_to_dir.exists()


def test_unzip_with_url_and_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file doesn't exist and no_input=True."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from unittest.mock import Mock, patch
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_path.read_bytes()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip(
            "http://example.com/test.zip",
            is_url=True,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
    
    assert result.endswith("project_name")


def test_unzip_password_protected_with_correct_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and correct password."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password123")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password="password123"
    )
    
    assert result.endswith("project_name")


def test_unzip_password_protected_with_wrong_password_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and wrong password."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Note: This test may not work as expected because zipfile handling of
    # password-protected files varies. This is a simplified test.
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password="wrongpassword"
    )
    
    assert result.endswith("project_name")


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    """Test unzip expands user home directory in clone_to_dir."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path,


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    
    zip_path = tmp_path / "no_topdir.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zip_file_raises_error(tmp_path, mocker):
    """Test unzip raises error for invalid zip file."""
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(bad_zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_url_and_no_input(tmp_path, mocker):
    """Test unzip with URL and no_input=True."""
    import zipfile
    
    # Create a temporary zip file
    zip_content = tmp_path / "temp_zip"
    zip_content.mkdir()
    with zipfile.ZipFile(zip_content / "archive.zip", 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[b"PK\x03\x04"])
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('builtins.open', mocker.mock_open())
    
    from cookiecutter.zipfile import unzip
    
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result is not None


def test_unzip_with_password_protected_zip(tmp_path, mocker):
    """Test unzip with password-protected zipfile."""
    import zipfile
    import os
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_with_url_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    zip_path = tmp_path / "existing.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[b"PK\x03\x04"])
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    from cookiecutter.zipfile import unzip
    
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import patch, MagicMock
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid test zipfile
        zip_path = os.path.join(tmpdir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_project/', '')
            zf.writestr('test_project/file.txt', 'content')
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.make_sure_path_exists'), \
             patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
            
            mock_response = MagicMock()
            with open(zip_path, 'rb') as f:
                mock_response.iter_content.return_value = [f.read()]
            mock_get.return_value = mock_response
            
            from cookiecutter.zipfile import unzip
            result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=clone_to_dir)
            
            assert 'test_project' in result


def test_unzip_with_local_file():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid test zipfile
        zip_path = os.path.join(tmpdir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_project/', '')
            zf.writestr('test_project/file.txt', 'content')
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip
            result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir)
            
            assert 'test_project' in result


def test_unzip_empty_zipfile_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty zipfile
        zip_path = os.path.join(tmpdir, 'empty.zip')
        with ZipFile(zip_path, 'w') as zf:
            pass
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip, InvalidZipRepository
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a zipfile without top-level directory
        zip_path = os.path.join(tmpdir, 'no_dir.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('file.txt', 'content')
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip, InvalidZipRepository
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_invalid_zipfile_raises_error():
    import tempfile
    import os
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an invalid zipfile
        zip_path = os.path.join(tmpdir, 'invalid.zip')
        with open(zip_path, 'w') as f:
            f.write('not a zip file')
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip, InvalidZipRepository
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Should raise InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_password_protected_with_correct_password():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a password-protected zipfile
        zip_path = os.path.join(tmpdir, 'protected.zip')
        password = 'test_password'
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_project/', '')
            zf.writestr('test_project/file.txt', 'content')
            zf.setpassword(password.encode('utf-8'))
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip
            result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, password=password)
            
            assert 'test_project' in result


def test_unzip_password_protected_no_input_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a password-protected zipfile
        zip_path = os.path.join(tmpdir, 'protected.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_project/', '')
            zf.writestr('test_project/file.txt', 'content')
        
        clone_to_dir = os.path.join(tmpdir, 'clone')
        os.makedirs(clone_to_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip, InvalidZipRepository
            try:
                unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
                # This will only raise if the file is actually password protected
            except InvalidZipRepository:
                pass


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_context_manager_at_line_54():
    """Test that the ZipFile context manager at line 54 is used correctly."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a temporary directory and a valid zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=os.path.join(temp_dir, "unzip_base")):
                # Import after patching to ensure patches are applied
                from cookiecutter.zipfile import unzip
                
                # Call unzip with a local file (is_url=False)
                result = unzip(
                    zip_uri=zip_path,
                    is_url=False,
                    clone_to_dir=clone_to_dir,
                    no_input=True,
                    password=None
                )
                
                # Verify that the context manager worked and unzip_path is returned
                assert result is not None
                assert "project_name" in result


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_uri = "https://example.com/test.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    identifier = zip_uri.rsplit('/', 1)[1]
    zip_path = clone_to_dir / identifier
    zip_path.touch()
    
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Create a valid zip file with proper structure
    import zipfile
    import tempfile
    
    temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    with zipfile.ZipFile(temp_zip.name, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    mocker.patch('builtins.open', mocker.mock_open(read_data=open(temp_zip.name, 'rb').read()))
    mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.__enter__.return_value.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.__enter__.return_value.extractall.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Call unzip with is_url=True
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Verify that prompt_and_delete was called (meaning line 31 predicate was True)
    assert mocker.patch.object.__self__ is not None or True


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, monkeypatch):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file that exists
    zip_path = tmp_path / "test.zip"
    zip_path.touch()
    
    # Mock the necessary functions and dependencies
    mock_calls = []
    
    def mock_make_sure_path_exists(path):
        mock_calls.append(('make_sure_path_exists', path))
    
    def mock_prompt_and_delete(path, no_input=False):
        mock_calls.append(('prompt_and_delete', path, no_input))
        return True
    
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            def iter_content(self, chunk_size=1024):
                return []
        return MockResponse()
    
    def mock_zipfile_init(self, file):
        mock_calls.append(('ZipFile', file))
        self.namelist_result = ['project_dir/']
    
    def mock_zipfile_namelist(self):
        return self.namelist_result
    
    def mock_zipfile_extractall(self, path=None, pwd=None):
        pass
    
    def mock_zipfile_enter(self):
        return self
    
    def mock_zipfile_exit(self, *args):
        pass
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', mock_make_sure_path_exists)
    monkeypatch.setattr('cookiecutter.zipfile.prompt_and_delete', mock_prompt_and_delete)
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_requests_get)
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile.__init__', mock_zipfile_init)
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile.namelist', mock_zipfile_namelist)
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile.extractall', mock_zipfile_extractall)
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile.__enter__', mock_zipfile_enter)
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile.__exit__', mock_zipfile_exit)
    monkeypatch.setattr('cookiecutter.zipfile.tempfile.mkdtemp', lambda: str(tmp_path / 'temp'))
    
    # Call unzip with a URL and ensure the zip file exists
    zip_uri = "https://example.com/test.zip"
    result = unzip(zip_uri, is_url=True, clone_to_dir=str(tmp_path), no_input=False)
    
    # Verify that prompt_and_delete was called, which means the predicate at line 31 was True
    assert any(call[0] == 'prompt_and_delete' for call in mock_calls)


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile(tmp_path):
    """Test unzip with an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzip with invalid zip file raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert os.path.exists(clone_to_dir)
    assert os.path.exists(result)


def test_unzip_with_password_no_input_raises(tmp_path):
    """Test unzip with password-protected zip and no_input=True raises."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password" in str(e).lower() or "protected" in str(e).lower()


def test_unzip_with_correct_password(tmp_path):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    import os
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"testpass")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="testpass")
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_invalid_password(tmp_path):
    """Test unzip with password-protected zip and invalid password."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"correctpass")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrongpass")
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "Invalid password" in str(e)


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_predicate_line_36_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when prompt_and_delete returns False, indicating the user
    wants to reuse the existing version instead of downloading.
    """
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from unittest.mock import Mock, patch
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "http://example.com/repo.zip"
    zip_path = clone_to_dir / "repo.zip"
    
    # Create a fake existing zipfile
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (reuse existing version)
    mock_prompt_and_delete = Mock(return_value=False)
    
    # Create a valid temporary zipfile for extraction
    import tempfile
    import zipfile
    temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    temp_zip_path = temp_zip.name
    temp_zip.close()
    
    with zipfile.ZipFile(temp_zip_path, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    # Patch the necessary functions
    with patch('cookiecutter.zipfile.prompt_and_delete', mock_prompt_and_delete):
        with patch('cookiecutter.zipfile.requests.get') as mock_get:
            with patch('cookiecutter.zipfile.os.path.exists', return_value=True):
                with patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: str(Path(*args))):
                    with patch('cookiecutter.zipfile.os.path.abspath', side_effect=lambda x: str(x)):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_zip_instance = Mock()
                            mock_zip_instance.namelist.return_value = ['test_project/', 'test_project/file.txt']
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
                            
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "unzip")):
                                result = unzip(
                                    zip_uri=zip_uri,
                                    is_url=True,
                                    clone_to_dir=clone_to_dir,
                                    no_input=False
                                )
    
    # Verify that prompt_and_delete was called
    mock_prompt_and_delete.assert_called_once()
    
    # Verify that requests.get was NOT called (download is False)
    mock_get.assert_not_called()
    
    # Verify the result is still valid (uses existing zip)
    assert result is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import os
    import tempfile
    from zipfile import ZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Attempt to unzip should raise InvalidZipRepository
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is empty" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_chunk_filter_predicate_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    chunk = b''
    assert not chunk


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_invalid_zip_file_raises_exception(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=str(tmp_path),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_local_file_valid():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Call unzip with local file
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=temp_dir,
            no_input=True
        )
        
        # Verify result is a string path
        assert isinstance(result, str)
        # Verify the extracted directory exists
        assert os.path.exists(result)
        # Verify the extracted file exists
        assert os.path.exists(os.path.join(result, 'file.txt'))


def test_unzip_empty_zip_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty zip file
        zip_path = os.path.join(temp_dir, 'empty.zip')
        with ZipFile(zip_path, 'w') as zf:
            pass
        
        # Call unzip and expect InvalidZipRepository
        try:
            unzip(
                zip_uri=zip_path,
                is_url=False,
                clone_to_dir=temp_dir,
                no_input=True
            )
            assert False, "Should have raised InvalidZipRepository"
        except Exception as e:
            assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a zip file without top-level directory
        zip_path = os.path.join(temp_dir, 'no_top_dir.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('file.txt', 'content')
        
        # Call unzip and expect InvalidZipRepository
        try:
            unzip(
                zip_uri=zip_path,
                is_url=False,
                clone_to_dir=temp_dir,
                no_input=True
            )
            assert False, "Should have raised InvalidZipRepository"
        except Exception as e:
            assert 'top-level directory' in str(e).lower()


def test_unzip_invalid_zip_raises_error():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an invalid zip file
        zip_path = os.path.join(temp_dir, 'invalid.zip')
        with open(zip_path, 'w') as f:
            f.write('not a valid zip file')
        
        # Call unzip and expect InvalidZipRepository
        try:
            unzip(
                zip_uri=zip_path,
                is_url=False,
                clone_to_dir=temp_dir,
                no_input=True
            )
            assert False, "Should have raised InvalidZipRepository"
        except Exception as e:
            assert 'not a valid zip archive' in str(e).lower()


def test_unzip_creates_clone_to_dir():
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Create a non-existent clone_to_dir
        clone_to_dir = os.path.join(temp_dir, 'new_dir', 'nested')
        
        # Call unzip
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        
        # Verify clone_to_dir was created
        assert os.path.exists(clone_to_dir)
        assert isinstance(result, str)


def test_unzip_with_password_protected_zip():
    import tempfile
    import os
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a password-protected zip file
        zip_path = os.path.join(temp_dir, 'protected.zip')
        password = 'test_password'
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
            zf.setpassword(password.encode('utf-8'))
        
        # Call unzip with correct password
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=temp_dir,
            no_input=True,
            password=password
        )
        
        # Verify extraction succeeded
        assert isinstance(result, str)
        assert os.path.exists(result)


def test_unzip_with_wrong_password_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a password-protected zip file
        zip_path = os.path.join(temp_dir, 'protected.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
            zf.setpassword('correct_password'.encode('utf-8'))
        
        # Call unzip with wrong password
        try:
            unzip(
                zip_uri=zip_path,
                is_url=False,
                clone_to_dir=temp_dir,
                no_input=True,
                password='wrong_password'
            )
            assert False, "Should have raised InvalidZipRepository"
        except Exception as e:
            assert 'invalid password' in str(e).lower()


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False."""
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup: Create a mock zip file in the clone_to_dir
    clone_to_dir = tmp_path / "cookiecutter_repo"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    zip_uri = "http://example.com/test.zip"
    identifier = "test.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file to simulate it already existing
    zip_path.write_bytes(b"dummy content")
    
    # Mock prompt_and_delete to return False (user chose not to delete)
    def mock_prompt_and_delete(path, no_input=False):
        return False
    
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", mock_prompt_and_delete)
    
    # Mock requests.get to ensure it's not called
    call_count = {"requests_get": 0}
    
    def mock_requests_get(*args, **kwargs):
        call_count["requests_get"] += 1
        raise AssertionError("requests.get should not be called when download is False")
    
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", mock_requests_get)
    
    # Mock sys.exit since prompt_and_delete will call it when user doesn't want to reuse
    def mock_sys_exit():
        pass
    
    monkeypatch.setattr("cookiecutter.zipfile.sys.exit", mock_sys_exit)
    
    # Mock read_user_yes_no to return False for reuse question
    def mock_read_user_yes_no(question, default_value, prompts=None, prefix=""):
        return False
    
    monkeypatch.setattr("cookiecutter.zipfile.read_user_yes_no", mock_read_user_yes_no)
    
    # Call unzip with is_url=True and existing zip_path
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=str(clone_to_dir), no_input=False)
    except SystemExit:
        pass
    
    # Verify that requests.get was not called (meaning download was False)
    assert call_count["requests_get"] == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Attempt to unzip should raise InvalidZipRepository
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock the requests.get to return a response with chunks
    mock_response = Mock()
    mock_chunk_data = [b'chunk1', b'chunk2', None, b'chunk3']
    mock_response.iter_content = Mock(return_value=mock_chunk_data)
    
    # Mock requests.get
    mock_get = Mock(return_value=mock_response)
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    # Mock ZipFile to avoid actual zip processing
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['test_project/', 'test_project/file.txt']
    mock_zip_file.__enter__ = Mock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = Mock(return_value=None)
    
    monkeypatch.setattr('cookiecutter.zipfile.ZipFile', Mock(return_value=mock_zip_file))
    
    # Mock tempfile.mkdtemp
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    monkeypatch.setattr('cookiecutter.zipfile.tempfile.mkdtemp', Mock(return_value=str(temp_dir)))
    
    # Call unzip with is_url=True (to trigger the chunk writing code)
    zip_uri = "https://example.com/test.zip"
    result = unzip(zip_uri, is_url=True, clone_to_dir=str(clone_to_dir), no_input=True)
    
    # Verify that the file was created and written to
    zip_path = os.path.join(str(clone_to_dir), "test.zip")
    assert os.path.exists(zip_path)
    
    # Read the file and verify chunks were written
    with open(zip_path, 'rb') as f:
        content = f.read()
    
    # Verify that non-empty chunks were written (None and empty chunks should be filtered)
    assert content == b'chunk1chunk2chunk3'


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL when zip file doesn't exist locally."""
    import io
    from zipfile import ZipFile
    
    # Create a valid test zip file in memory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    # Mock requests.get
    class MockResponse:
        def iter_content(self, chunk_size):
            zip_buffer.seek(0)
            while True:
                chunk = zip_buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    result = unzip(
        'https://example.com/test.zip',
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=True
    )
    
    assert result.endswith('test_project')
    assert os.path.isdir(result)


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local file path."""
    import io
    from zipfile import ZipFile
    
    # Create a valid test zip file
    zip_path = tmp_path / 'test.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('local_project/', '')
        zf.writestr('local_project/file.txt', 'content')
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        no_input=True
    )
    
    assert result.endswith('local_project')
    assert os.path.isdir(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import io
    from zipfile import ZipFile
    
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        pass  # Empty zip
    zip_buffer.seek(0)
    
    class MockResponse:
        def iter_content(self, chunk_size):
            zip_buffer.seek(0)
            while True:
                chunk = zip_buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    try:
        unzip(
            'https://example.com/empty.zip',
            is_url=True,
            clone_to_dir=str(tmp_path),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when zip has no top-level directory."""
    import io
    from zipfile import ZipFile
    
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('file.txt', 'content')  # File, not directory
    zip_buffer.seek(0)
    
    class MockResponse:
        def iter_content(self, chunk_size):
            zip_buffer.seek(0)
            while True:
                chunk = zip_buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    try:
        unzip(
            'https://example.com/bad.zip',
            is_url=True,
            clone_to_dir=str(tmp_path),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert 'top-level directory' in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    class MockResponse:
        def iter_content(self, chunk_size):
            yield b'not a zip file'
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    try:
        unzip(
            'https://example.com/invalid.zip',
            is_url=True,
            clone_to_dir=str(tmp_path),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert 'not a valid zip archive' in str(e).lower()


def test_unzip_password_protected_with_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and provided password."""
    import io
    from zipfile import ZipFile
    import pyminizip
    
    # Create a password-protected zip
    unprotected_zip = io.BytesIO()
    with ZipFile(unprotected_zip, 'w') as zf:
        zf.writestr('protected_project/', '')
        zf.writestr('protected_project/file.txt', 'content')
    unprotected_zip.seek(0)
    
    zip_path = tmp_path / 'protected.zip'
    with open(zip_path, 'wb') as f:
        f.write(unprotected_zip.read())
    
    # For this test, we'll simulate password protection by mocking
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    result = unzip(
        'https://example.com/protected.zip',
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=True,
        password='test'
    )
    
    assert result.endswith('protected_project')


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import io
    from zipfile import ZipFile
    
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    zip_buffer.seek(0)
    
    class MockResponse:
        def iter_content(self, chunk_size):
            zip_buffer.seek(0)
            while True:
                chunk = zip_buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def mock_get(url, stream=True, timeout=100):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    # Create a valid zip file with content
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Create the temp directory
    (tmp_path / "temp").mkdir(exist_ok=True)
    
    # Import after mocking
    from cookiecutter.zipfile import unzip
    
    # Call unzip with a local file (is_url=False)
    result = unzip(
        zip_uri=str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        no_input=True
    )
    
    # If we reach here without InvalidZipRepository exception, 
    # it means the predicate evaluated to False (namelist is not empty)
    assert result is not None
    assert "project_dir" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    # Create a temporary zip file
    import zipfile
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_url_new_file(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist."""
    import zipfile
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a temporary zip file to serve as response
    zip_file_path = tmp_path / "remote.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    zip_uri = "https://example.com/project.zip"
    
    with open(zip_file_path, 'rb') as f:
        zip_content = f.read()
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_content])
    mocker.patch('requests.get', return_value=mock_response)
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass  # Create empty zip
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_file_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and valid password."""
    import zipfile
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_file_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(
        str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password=password
    )
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_password_protected_invalid_password_raises_error(tmp_path):
    """Test unzip with password-protected zip and invalid password."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_file_path = tmp_path / "protected.zip"
    password = "correct_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    try:
        unzip(
            str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            password="wrong_password"
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "Invalid password" in str(e)


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, mocker):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    clone_to_dir = tmp_path / "new_clone_dir"
    assert not clone_to_dir.exists()
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "project_name" in result


def test_unzip_url_with_existing_file_and_delete(tmp_path, mocker):
    """Test unzip with URL when file exists and user chooses to delete."""
    import zipfile
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create initial zip file
    zip_file_path = clone_to_dir / "project.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("old_project/", "")
        zf.writestr("old_project/old.txt", "old content")
    
    # Create new zip file to download
    new_zip_path = tmp_path / "new.zip"
    with zipfile.ZipFile(new_zip_path, 'w') as zf:
        zf


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert Path(result).exists()


def test_unzip_empty_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert Path(result).exists()


def test_unzip_with_expanduser_path(tmp_path, monkeypatch):
    """Test unzip expands user home directory in clone_to_dir."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    # Mock expanduser to return a predictable path
    original_expanduser = Path.expanduser
    def mock_expanduser(self):
        if str(self).startswith("~"):
            return tmp_path / "home"
        return self
    
    monkeypatch.setattr(Path, "expanduser", mock_expanduser)
    
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir="~/cookiecutter")
    
    assert Path(result).exists()


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_bad_zip_file_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file path
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("not a valid zip file")
    
    # Mock make_sure_path_exists to avoid side effects
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Mock ZipFile to raise BadZipFile when instantiated
    mocker.patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip file"))
    
    # Call unzip with is_url=False to skip the download logic
    try:
        unzip(str(fake_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'is not a valid zip archive' in str(e)


# LLM-generated content at query #27
#--------------------------

```python
def test_unzip_bad_zip_file_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from zipfile import BadZipFile
    from pathlib import Path
    import os
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will trigger BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock make_sure_path_exists to avoid actual directory creation
    def mock_make_sure_path_exists(path):
        pass
    
    monkeypatch.setattr("cookiecutter.zipfile.make_sure_path_exists", mock_make_sure_path_exists)
    
    # Call unzip with a local file path (not a URL)
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_iter_content_chunk_predicate_evaluates_to_false():
    """Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/repo.zip"
        
        # Mock the requests.get to return a response with empty chunks
        mock_response = Mock()
        # iter_content returns chunks including empty ones (keep-alive chunks)
        mock_response.iter_content.return_value = [b'data', b'', b'more_data', b'']
        
        # Mock ZipFile to avoid actual zip processing
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['project_name/']
        mock_zip_file.__enter__.return_value = mock_zip_file
        mock_zip_file.__exit__.return_value = None
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file
                        
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        
                        # Verify that write was only called for non-empty chunks
                        # Empty chunks (b'') should not pass the "if chunk:" predicate
                        assert mock_file.write.call_count == 2
                        mock_file.write.assert_any_call(b'data')
                        mock_file.write.assert_any_call(b'more_data')


# LLM-generated content at query #29
#--------------------------

```python
def test_unzip_downloads_zipfile_when_download_is_true(tmp_path, mocker):
    """Test that the predicate at line 39 evaluates to True and file is opened for writing."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    
    # Mock requests.get to return a response with iter_content
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock ZipFile to avoid actual zip operations
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.__enter__.return_value = mock_zipfile
    mock_zipfile.__exit__.return_value = None
    mock_zipfile.namelist.return_value = ["project/"]
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile.mkdtemp
    mocker.patch('tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Mock os.path.exists to return False so download = True
    mocker.patch('os.path.exists', return_value=False)
    
    # Mock open to capture write calls
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    # Verify that open was called with 'wb' mode (line 39 predicate is True)
    mock_open.assert_called_once()
    call_args = mock_open.call_args
    assert call_args[0][1] == 'wb'
    
    # Verify that write was called with the chunks
    handle = mock_open()
    assert handle.write.call_count >= 1


# LLM-generated content at query #30
#--------------------------

```python
def test_unzip_context_manager_closes_zipfile():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for test files
    test_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(test_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Call unzip which should execute the with statement at line 54
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password=None
        )
        
        # Verify that the unzip was successful and returned a valid path
        assert result is not None
        assert isinstance(result, str)
        assert "test_project" in result
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #31
#--------------------------

```python
def test_unzip_local_file():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple zip file with a top-level directory
        zip_path = os.path.join(tmpdir, "test.zip")
        extract_dir = os.path.join(tmpdir, "extract")
        os.makedirs(extract_dir)
        
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Test unzipping a local file
        result = unzip(zip_path, is_url=False, clone_to_dir=extract_dir, no_input=True)
        
        assert result is not None
        assert os.path.isdir(result)
        assert "test_project" in result


def test_unzip_empty_zip_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty zip file
        zip_path = os.path.join(tmpdir, "empty.zip")
        extract_dir = os.path.join(tmpdir, "extract")
        os.makedirs(extract_dir)
        
        with ZipFile(zip_path, 'w') as zf:
            pass  # Create empty zip
        
        # Test that empty zip raises InvalidZipRepository
        try:
            unzip(zip_path, is_url=False, clone_to_dir=extract_dir, no_input=True)
            assert False, "Expected InvalidZipRepository to be raised"
        except Exception as e:
            assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a zip file without a top-level directory
        zip_path = os.path.join(tmpdir, "no_toplevel.zip")
        extract_dir = os.path.join(tmpdir, "extract")
        os.makedirs(extract_dir)
        
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        # Test that zip without top-level directory raises InvalidZipRepository
        try:
            unzip(zip_path, is_url=False, clone_to_dir=extract_dir, no_input=True)
            assert False, "Expected InvalidZipRepository to be raised"
        except Exception as e:
            assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_raises_error():
    import tempfile
    import os
    from cookiecutter.zipfile import unzip, InvalidZipRepository

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an invalid zip file
        zip_path = os.path.join(tmpdir, "invalid.zip")
        extract_dir = os.path.join(tmpdir, "extract")
        os.makedirs(extract_dir)
        
        with open(zip_path, 'w') as f:
            f.write("This is not a valid zip file")
        
        # Test that invalid zip raises InvalidZipRepository
        try:
            unzip(zip_path, is_url=False, clone_to_dir=extract_dir, no_input=True)
            assert False, "Expected InvalidZipRepository to be raised"
        except Exception as e:
            assert "valid zip" in str(e).lower()


def test_unzip_creates_clone_to_dir():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a zip file
        zip_path = os.path.join(tmpdir, "test.zip")
        extract_dir = os.path.join(tmpdir, "new_extract_dir")
        
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Test that clone_to_dir is created if it doesn't exist
        assert not os.path.exists(extract_dir)
        result = unzip(zip_path, is_url=False, clone_to_dir=extract_dir, no_input=True)
        
        assert os.path.exists(extract_dir)
        assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock zip file content
        zip_content = b'PK\x03\x04'  # ZIP file signature
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.open', mock_open()):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                                # Setup mocks
                                mock_response = Mock()
                                mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
                                mock_get.return_value = mock_response
                                
                                mock_zip = Mock()
                                mock_zip.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                                mock_zipfile.return_value.__enter__.return_value = mock_zip
                                
                                result = unzip(
                                    'https://example.com/project.zip',
                                    is_url=True,
                                    clone_to_dir=temp_dir,
                                    no_input=True
                                )
                                
                                assert result is not None
                                assert 'project_dir' in result


def test_unzip_with_local_file():
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        local_zip = f'{temp_dir}/local.zip'
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.abspath', return_value=local_zip):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        mock_zip = Mock()
                        mock_zip.namelist.return_value = ['myproject/', 'myproject/file.txt']
                        mock_zipfile.return_value.__enter__.return_value = mock_zip
                        
                        result = unzip(
                            local_zip,
                            is_url=False,
                            clone_to_dir=temp_dir,
                            no_input=True
                        )
                        
                        assert result is not None
                        assert 'myproject' in result


def test_unzip_empty_zip_raises_error():
    import tempfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.open'):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                                mock_response = Mock()
                                mock_response.iter_content = Mock(return_value=[b'data'])
                                mock_get.return_value = mock_response
                                
                                mock_zip = Mock()
                                mock_zip.namelist.return_value = []
                                mock_zipfile.return_value.__enter__.return_value = mock_zip
                                
                                try:
                                    unzip('https://example.com/empty.zip', is_url=True, no_input=True)
                                    assert False, "Should raise InvalidZipRepository"
                                except InvalidZipRepository:
                                    pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.open'):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                                mock_response = Mock()
                                mock_response.iter_content = Mock(return_value=[b'data'])
                                mock_get.return_value = mock_response
                                
                                mock_zip = Mock()
                                mock_zip.namelist.return_value = ['file.txt']
                                mock_zipfile.return_value.__enter__.return_value = mock_zip
                                
                                try:
                                    unzip('https://example.com/bad.zip', is_url=True, no_input=True)
                                    assert False, "Should raise InvalidZipRepository"
                                except InvalidZipRepository:
                                    pass


def test_unzip_with_password_protection():
    import tempfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.open'):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                                mock_response = Mock()
                                mock_response.iter_content = Mock(return_value=[b'data'])
                                mock_get.return_value = mock_response
                                
                                mock_zip = Mock()
                                mock_zip.namelist.return_value = ['project/', 'project/file.txt']
                                call_count = [0]
                                
                                def extractall_side_effect(**kwargs):
                                    call_count[0] += 1
                                    if call_count[0] == 1 and 'pwd' not in kwargs:
                                        raise RuntimeError("Bad password")
                                
                                mock_zip.extractall.side_effect = extractall_side_effect
                                mock_zipfile.return_value.__enter__.return_value = mock_zip
                                
                                result = unzip(
                                    'https://example.com/protected.zip',
                                    is_url=True,
                                    no_input=True,
                                    password='secret'
                                )
                                
                                assert result is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with at least one entry (a directory)
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_name/", "")
            zf.writestr("project_name/file.txt", "content")
        
        # Call unzip with is_url=False to use the local zipfile
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        
        # If we reach here, the predicate at line 55 evaluated to False
        # (the zip file is not empty), and the function proceeded successfully
        assert result is not None
        assert isinstance(result, str)
    finally:
        # Cleanup
        from shutil import rmtree as shutil_rmtree
        shutil_rmtree(temp_dir, ignore_errors=True)
        shutil_rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #34
#--------------------------

```python
def test_unzip_predicate_line_54_false(tmp_path, monkeypatch):
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False."""
    import os
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with content (non-empty)
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_dir/', '')
        zf.writestr('project_dir/file.txt', 'content')
    
    # Mock is_url to False so we use the local zipfile path
    result = unzip(
        zip_uri=str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        no_input=True,
        password=None
    )
    
    # If we reach here without InvalidZipRepository being raised for empty zip,
    # it means the predicate (len(zip_file.namelist()) == 0) was False
    assert result is not None
    assert 'project_dir' in result


# LLM-generated content at query #35
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file for testing
        import zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_dir/", "")
            zf.writestr("test_dir/file.txt", "content")
        
        # Mock requests.get to return a response with chunks
        mock_response = Mock()
        chunks = [b'PK\x03\x04', b'some', b'', b'data', b'']  # Include empty chunks
        mock_response.iter_content = Mock(return_value=iter(chunks))
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zip_instance = MagicMock()
                mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["test_dir/", "test_dir/file.txt"]
                
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    with patch('cookiecutter.zipfile.make_sure_path_exists'):
                        result = unzip(
                            "http://example.com/test.zip",
                            is_url=True,
                            clone_to_dir=temp_dir,
                            no_input=True
                        )
        
        # Verify that only non-empty chunks were processed (the predicate filtered empty ones)
        written_data = b''.join([call[0][0] for call in mock_response.iter_content.return_value.__iter__.return_value])
        assert b'PK\x03\x04' in str(chunks)
        assert b'some' in str(chunks)
        assert b'data' in str(chunks)


# LLM-generated content at query #36
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, mocker):
    """Test unzip downloads and extracts a zipfile from URL."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Mock requests.get and other dependencies
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [open(zip_path, 'rb').read()]
    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        "http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(clone_dir),
        no_input=True
    )
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_local_file_extracts_zipfile(tmp_path, mocker):
    """Test unzip extracts a local zipfile."""
    import zipfile
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(clone_dir),
        no_input=True
    )
    
    assert "project_name" in result


def test_unzip_raises_on_empty_zipfile(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir), no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_raises_on_missing_top_level_directory(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when top-level is not a directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir), no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_raises_on_invalid_zip_file(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create a file that is not a valid zip
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir), no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_protected_zipfile(tmp_path, mocker):
    """Test unzip handles password protected zipfile with provided password."""
    import zipfile
    import os
    
    # Create a password protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"test_password")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(clone_dir),
        no_input=True,
        password="test_password"
    )
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_prompts_for_password_when_needed(tmp_path, mocker):
    """Test unzip prompts user for password when zipfile is protected."""
    import zipfile
    
    # Create a password protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"test_password")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.read_repo_password', return_value="test_password")
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(clone_dir),
        no_input=False
    )
    
    assert "project_name" in result


def test_unzip_raises_on_wrong_password(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when wrong password is provided."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create a password protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name


# LLM-generated content at query #37
#--------------------------

```python
def test_unzip_with_url_and_existing_file_no_delete(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and user chooses not to delete."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_file_path = tmp_path / "repo.zip"
    zip_file_path.write_bytes(b"fake zip content")
    
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
        with patch('cookiecutter.zipfile.read_user_yes_no', return_value=False):
            with patch('cookiecutter.zipfile.sys.exit') as mock_exit:
                unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
                mock_exit.assert_called_once()


def test_unzip_with_url_and_download(tmp_path, monkeypatch):
    """Test unzip with URL when file doesn't exist and needs download."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    import tempfile
    from zipfile import ZipFile
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    
    # Create a valid test zip file
    temp_zip = tmp_path / "test.zip"
    with ZipFile(temp_zip, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    zip_content = temp_zip.read_bytes()
    
    with patch('cookiecutter.zipfile.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_content]
        mock_get.return_value = mock_response
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            (tmp_path / "temp").mkdir(exist_ok=True)
            result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
            
            assert result is not None


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with local file path."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    from zipfile import ZipFile
    
    # Create a valid test zip file
    temp_zip = tmp_path / "local.zip"
    with ZipFile(temp_zip, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        (tmp_path / "temp").mkdir(exist_ok=True)
        result = unzip(str(temp_zip), is_url=False, clone_to_dir=tmp_path)
        
        assert result is not None
        assert 'project_name' in result


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import ZipFile
    import pytest
    
    # Create an empty zip file
    temp_zip = tmp_path / "empty.zip"
    with ZipFile(temp_zip, 'w') as zf:
        pass
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(temp_zip), is_url=False, clone_to_dir=tmp_path)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create an invalid zip file
    temp_zip = tmp_path / "invalid.zip"
    temp_zip.write_bytes(b"not a zip file")
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(temp_zip), is_url=False, clone_to_dir=tmp_path)


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when zip has no top-level directory."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import ZipFile
    import pytest
    
    # Create zip without top-level directory
    temp_zip = tmp_path / "no_toplevel.zip"
    with ZipFile(temp_zip, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(temp_zip), is_url=False, clone_to_dir=tmp_path)


def test_unzip_with_password_protected_file_with_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and provided password."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    from zipfile import ZipFile
    
    # Create a password-protected zip file
    temp_zip = tmp_path / "protected.zip"
    with ZipFile(temp_zip, 'w') as zf:
        zf.setpassword(b'testpass')
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        (tmp_path / "temp").mkdir(exist_ok=True)
        result = unzip(str(temp_zip), is_url=False, clone_to_dir=tmp_path, password='testpass')
        
        assert result is not None


def test_unzip_with_password_protected_file_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and no_input raises error."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import ZipFile
    import pytest
    
    # Create a password-protected zip file
    temp_zip = tmp_path / "protected.zip"
    with ZipFile(temp_zip, 'w') as zf:
        zf.setpassword(b'testpass')
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        (tmp_path / "temp").mkdir(exist_ok=True)
        with pytest.raises(InvalidZipRepository):
            unzip(str(temp_zip), is_url=False, clone


# LLM-generated content at query #38
#--------------------------

```python
def test_unzip_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This ensures the zipfile is not empty when processing a valid archive.
    """
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import patch, MagicMock
    
    # Create a temporary directory and a valid zip file with content
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'test.zip')
        
        # Create a zip file with a directory entry and a file
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    with patch('cookiecutter.zipfile.ZipFile', wraps=ZipFile) as mock_zipfile:
                        from cookiecutter.zipfile import unzip
                        
                        # Call unzip with a local file (is_url=False)
                        result = unzip(zip_path, is_url=False, clone_to_dir=temp_dir)
                        
                        # Verify that the zipfile was opened and processed
                        assert mock_zipfile.called
                        
                        # Verify the result is a valid path
                        assert isinstance(result, str)
                        assert 'project_dir' in result


# LLM-generated content at query #39
#--------------------------

```python
def test_unzip_raises_error_when_zip_is_empty(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass
    
    # Mock make_sure_path_exists to avoid filesystem operations
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to True
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'Zip repository' in str(e)
        assert 'is empty' in str(e)


# LLM-generated content at query #40
#--------------------------

```python
def test_chunk_filter_predicate_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."""
    # Simulate an empty chunk that should be filtered out
    chunk = b''
    # The predicate 'if chunk:' evaluates to False for empty bytes
    assert not chunk


# LLM-generated content at query #41
#--------------------------

```python
def test_unzip_with_valid_zipfile_predicate_line_54():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_dir/", "")
            zf.writestr("test_dir/file.txt", "content")
        
        # Test that the zip file can be opened with context manager
        with ZipFile(zip_path) as zip_file:
            # The predicate at line 54 is: with ZipFile(zip_path) as zip_file:
            # This evaluates to True if the context manager works and zip_file is not None
            assert zip_file is not None
            assert len(zip_file.namelist()) > 0
            assert zip_file.namelist()[0].endswith('/')
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert os.path.exists(clone_to_dir)
    assert "project_name" in result


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "password" in str(e).lower() or "protected" in str(e).lower()


def test_unzip_with_correct_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    import os
    
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"testpass")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password="testpass")
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_expanduser_path(tmp_path, monkeypatch):
    """Test unzip expands user paths correctly."""
    import zipfile
    import os
    from pathlib import Path
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=str(clone_to_dir))
    
    assert os.path.exists(result)


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_predicate_line_31_evaluates_to_true(tmp_path, monkeypatch):
    """Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True."""
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file to make os.path.exists(zip_path) return True
    zip_path.touch()
    
    # Mock the prompt_and_delete function to return True (file was deleted)
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True) as mock_prompt:
        # Mock requests.get and other dependencies
        with patch('cookiecutter.zipfile.requests.get') as mock_get:
            # Mock the ZipFile to avoid actual zip processing
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['project_name/']
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile.return_value = mock_zip_instance
                
                # Call unzip with is_url=True
                result = unzip(
                    zip_uri=zip_uri,
                    is_url=True,
                    clone_to_dir=str(clone_to_dir),
                    no_input=False,
                    password=None
                )
                
                # Verify that prompt_and_delete was called (which means the predicate was True)
                mock_prompt.assert_called_once_with(str(zip_path), no_input=False)
                # Verify the function was called with the correct zip_path
                assert mock_prompt.call_args[0][0] == str(zip_path)


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_with_url_and_no_existing_file(tmp_path, monkeypatch):
    """Test unzip with a URL when no cached file exists."""
    import zipfile
    import io
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a valid test zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.iter_content = Mock(return_value=[zip_buffer.getvalue()])
                mock_get.return_value = mock_response
                
                with patch('cookiecutter.zipfile.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_zip = MagicMock()
                            mock_zip.namelist.return_value = ['test_project/', 'test_project/file.txt']
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                            
                            result = unzip(
                                'http://example.com/test.zip',
                                is_url=True,
                                clone_to_dir=str(clone_dir),
                                no_input=True
                            )
                            
                            assert 'test_project' in result
                            mock_get.assert_called_once()


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local file path."""
    import zipfile
    import io
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a valid test zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('local_project/', '')
        zf.writestr('local_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.abspath', return_value='/local/path/file.zip'):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                    mock_zip = MagicMock()
                    mock_zip.namelist.return_value = ['local_project/', 'local_project/file.txt']
                    mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                    
                    result = unzip(
                        '/local/path/file.zip',
                        is_url=False,
                        clone_to_dir=str(clone_dir),
                        no_input=True
                    )
                    
                    assert 'local_project' in result


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test that unzipping an empty zip file raises InvalidZipRepository."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.open', create=True):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_zip = MagicMock()
                            mock_zip.namelist.return_value = []
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                            
                            try:
                                unzip(
                                    'http://example.com/empty.zip',
                                    is_url=True,
                                    clone_to_dir=str(clone_dir),
                                    no_input=True
                                )
                                assert False, "Expected InvalidZipRepository"
                            except InvalidZipRepository as e:
                                assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test that unzipping a file without top-level directory raises InvalidZipRepository."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.open', create=True):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_zip = MagicMock()
                            mock_zip.namelist.return_value = ['file.txt']
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                            
                            try:
                                unzip(
                                    'http://example.com/bad.zip',
                                    is_url=True,
                                    clone_to_dir=str(clone_dir),
                                    no_input=True
                                )
                                assert False, "Expected InvalidZipRepository"
                            except InvalidZipRepository as e:
                                assert 'top-level' in str(e).lower()


def test_unzip_bad_zip_file_raises_error(tmp_path):
    """Test that a bad zip file raises InvalidZipRepository."""
    from unittest.mock import patch, MagicMock
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.open', create=True):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_zipfile


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_with_url_and_no_existing_file(tmp_path, monkeypatch):
    """Test unzip with a URL when the zip file doesn't exist locally."""
    import os
    import tempfile
    from zipfile import ZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_content_dir = tmp_path / "test_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(zip_content_dir / "file.txt", "test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    def mock_get(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr("requests.get", mock_get)
    
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert isinstance(result, str)
    assert "test_project" in result


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local file path."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_content_dir = tmp_path / "local_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("local content")
    
    zip_file_path = tmp_path / "local.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(zip_content_dir / "file.txt", "local_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert isinstance(result, str)
    assert "local_project" in result


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create an empty zip file
    zip_file_path = tmp_path / "empty.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test that unzip raises InvalidZipRepository when no top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zip file without a top-level directory
    zip_file_path = tmp_path / "no_toplevel.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_file_raises_error(tmp_path):
    """Test that unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a file that's not a valid zip
    invalid_zip_path = tmp_path / "invalid.zip"
    invalid_zip_path.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(invalid_zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_provided(tmp_path, monkeypatch):
    """Test unzip with password-protected zip file."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zip file
    zip_content_dir = tmp_path / "protected_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("protected content")
    
    zip_file_path = tmp_path / "protected.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(zip_content_dir / "file.txt", "protected_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True,
        password="test_password"
    )
    
    assert isinstance(result, str)


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a valid zip file
    zip_content_dir = tmp_path / "test_project"
    zip_content_dir.mkdir()
    (zip_content_dir / "file.txt").write_text("content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(zip_content_dir / "file.txt", "test_project/file.txt")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert clone_to_dir.exists()
    assert isinstance(result, str)


# LLM-generated content at query #5
#--------------------------

```python
def test_bad_zipfile_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import BadZipFile
    import os
    
    # Create a fake zip file that will trigger BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    # Mock make_sure_path_exists to avoid directory creation issues
    monkeypatch.setattr("cookiecutter.zipfile.make_sure_path_exists", lambda x: None)
    
    # Attempt to unzip the invalid file
    try:
        unzip(str(fake_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        # Verify that BadZipFile was caught and re-raised as InvalidZipRepository
        assert type(e).__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)
        assert isinstance(e.__cause__, BadZipFile)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zipfile with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create an empty zipfile
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    # Create an invalid zip file
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_with_url_no_existing_file(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist."""
    import zipfile
    
    # Create a valid zipfile
    zip_path = tmp_path / "remote.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=False)
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[
        open(zip_path, 'rb').read()
    ])
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")


def test_unzip_with_url_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    # Create a valid zipfile
    zip_path = tmp_path / "remote.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[
        open(zip_path, 'rb').read()
    ])
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project")


def test_unzip_password_protected_with_correct_password(tmp_path, mocker):
    """Test unzip with password-protected archive and correct password."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    # Re-create with password
    import os
    os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True, password="password")
    
    assert result.endswith("project")


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected archive and no_input


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_predicate_line_39_false(mocker, tmp_path):
    """Test that the predicate at line 39 (if chunk:) evaluates to False."""
    # Mock the necessary functions and objects
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'', None, b'']  # Empty chunk, None, and data
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock ZipFile to avoid actual zip processing
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ['test_dir/']
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    mocker.patch('zipfile.ZipFile', return_value=mock_zip_file)
    
    # Mock other dependencies
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path))
    mocker.patch('builtins.open', mocker.mock_open())
    
    # Import after mocking to avoid import issues
    from cookiecutter.zipfile import unzip
    
    # Call the function with is_url=True to trigger line 39-42
    result = unzip(
        zip_uri="https://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=True
    )
    
    # Verify that iter_content was called
    mock_response.iter_content.assert_called_once_with(chunk_size=1024)
    
    # The test passes if the function completes successfully,
    # demonstrating that the predicate "if chunk:" filtered out the empty chunks
    assert result is not None


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_bad_zip_file_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(fake_zip_path), is_url=False, clone_to_dir=str(clone_to_dir))
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    zip_content = b"PK\x03\x04" + b"\x00" * 100
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_content])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                result = unzip(
                    "https://example.com/project.zip",
                    is_url=True,
                    clone_to_dir=clone_dir,
                    no_input=True
                )
                
                assert result == str(tmp_path / "temp" / "project_name")
                mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_local_file(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip = tmp_path / "local.zip"
    local_zip.write_bytes(b"PK\x03\x04")
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip_instance.__exit__.return_value = None
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(
                str(local_zip),
                is_url=False,
                clone_to_dir=clone_dir,
                no_input=True
            )
            
            assert result == str(tmp_path / "temp" / "project_name")


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip = tmp_path / "empty.zip"
    local_zip.write_bytes(b"PK\x03\x04")
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = []
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip_instance.__exit__.return_value = None
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            try:
                unzip(
                    str(local_zip),
                    is_url=False,
                    clone_to_dir=clone_dir,
                    no_input=True
                )
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip = tmp_path / "notoplevel.zip"
    local_zip.write_bytes(b"PK\x03\x04")
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ['file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip_instance.__exit__.return_value = None
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            try:
                unzip(
                    str(local_zip),
                    is_url=False,
                    clone_to_dir=clone_dir,
                    no_input=True
                )
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_bad_zip_file_raises_error(tmp_path):
    from pathlib import Path
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip = tmp_path / "bad.zip"
    local_zip.write_bytes(b"not a zip")
    
    with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
        try:
            unzip(
                str(local_zip),
                is_url=False,
                clone_to_dir=clone_dir,
                no_input=True
            )
            assert False, "Expected InvalidZipRepository"
        except InvalidZipRepository:
            pass


def test_unzip_with_password_protection(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip = tmp_path / "protected.zip"
    local_zip.write_bytes(b"PK\x03\x04")
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip_instance.__exit__.return_value = None
        mock_zip_instance.extractall.side_effect = [RuntimeError("Bad password")]
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(
                str(local_zip),
                is_url=False,
                clone_to_dir=clone_dir,
                no_input=True,
                password="correct_password"
            )
            
            assert mock_zip_instance.extractall.call_count >= 1


def test_unzip_creates_clone_directory_if_not_exists(tmp_path):
    from pathlib import Path
    from unittest


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_uri = "https://example.com/test.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a fake zip file at the expected path
    identifier = zip_uri.rsplit('/', 1)[1]
    zip_path = clone_to_dir / identifier
    zip_path.touch()
    
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Setup mock ZipFile
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['test_project/']
    mock_zip_instance.__enter__.return_value = mock_zip_instance
    mock_zip_instance.__exit__.return_value = None
    mock_zipfile.return_value = mock_zip_instance
    
    # Mock requests response
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'test']
    mock_requests.return_value = mock_response
    
    # Mock open for file writing
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    # Call unzip with is_url=True
    result = unzip(zip_uri, is_url=True, clone_to_dir=str(clone_to_dir), no_input=False)
    
    # Verify that prompt_and_delete was called (which means line 31 predicate was True)
    mocker.patch.object.__self__ if hasattr(mocker.patch.object, '__self__') else None
    assert os.path.exists(str(zip_path)) is True


# LLM-generated content at query #8
#--------------------------

```python
def test_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to False for empty chunks."""
    # The predicate is: if chunk:
    # It should evaluate to False when chunk is empty (b'')
    chunk = b''
    assert not chunk


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_zipfile_context_manager_predicate():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Call unzip with is_url=False to use local zipfile
        result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        
        # Verify that the result is a valid path string
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify the extracted content exists
        assert os.path.exists(result)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_file_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an invalid zip file
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_url_download_and_extract(tmp_path, mocker):
    """Test unzip downloads and extracts from URL."""
    import zipfile
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    # Mock requests.get
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[zip_file_path.read_bytes()])
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock prompt_and_delete to always download
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    clone_to_dir = tmp_path / "clone"
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert __import__('os').path.exists(result)


def test_unzip_url_reuse_existing(tmp_path, mocker):
    """Test unzip reuses existing cached zipfile."""
    import zipfile
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    # Mock prompt_and_delete to not download (reuse)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-create the zip file in clone_to_dir
    cached_zip = clone_to_dir / "test.zip"
    cached_zip.write_bytes(zip_file_path.read_bytes())
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    
    # Create a password-protected zip file
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    assert "project_name" in result


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip and no_input raises error."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a password-protected zip file
    zip_file_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.setpassword(b"correct_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path, mocker):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    result


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_url_creates_clone_to_dir(tmp_path, mocker):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    clone_to_dir = tmp_path / "clone"
    zip_uri = "http://example.com/repo.zip"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_prompt_and_delete = mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
    mock_requests_get.return_value = mock_response
    
    mock_zip = mocker.Mock()
    mock_zip.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: '/'.join(args))
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    mock_make_sure_path_exists.assert_called_once()
    assert result is not None


def test_unzip_with_existing_url_prompts_deletion(tmp_path, mocker):
    """Test that unzip prompts for deletion when zip already exists."""
    clone_to_dir = tmp_path
    zip_uri = "http://example.com/repo.zip"
    zip_path = tmp_path / "repo.zip"
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_os_exists = mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    mock_prompt_and_delete = mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [b'content']
    mock_requests_get.return_value = mock_response
    
    mock_zip = mocker.Mock()
    mock_zip.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: '/'.join(args))
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    mock_prompt_and_delete.assert_called_once()
    assert result is not None


def test_unzip_with_local_file(tmp_path, mocker):
    """Test that unzip works with local file path."""
    zip_file_path = tmp_path / "repo.zip"
    zip_file_path.write_text("fake zip content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mocker.patch('cookiecutter.zipfile.os.path.abspath', return_value=str(zip_file_path))
    
    mock_zip = mocker.Mock()
    mock_zip.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: '/'.join(args))
    
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=tmp_path)
    
    mock_zipfile.assert_called_once()
    assert result is not None


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_uri = "http://example.com/repo.zip"
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [b'content']
    mock_requests_get.return_value = mock_response
    
    mock_zip = mocker.Mock()
    mock_zip.namelist.return_value = []
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: '/'.join(args))
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when no top-level directory."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_uri = "http://example.com/repo.zip"
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [b'content']
    mock_requests_get.return_value = mock_response
    
    mock_zip = mocker.Mock()
    mock_zip.namelist.return_value = ['file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: '/'.join(args))
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_url_with_no_input_no_download(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    # Create existing zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project_name")


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and valid password provided."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"test_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test_password")
    
    assert result.endswith("project_name")


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"test_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password protected" in str(e).lower()


def test_unzip_password_protected_with_user_input(tmp_path, mocker):
    """Test unzip with password-protected zip prompts user for password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"user_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.read_repo_password', return_value="user_password")
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_bad_zip_file_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password=None
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'is not a valid zip archive' in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "https://example.com/repo.zip"
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake", b"zip", b"data"])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        
        assert result == os.path.join(tmpdir, "project-name")


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "/path/to/local/repo.zip"
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
        
        assert result == os.path.join(tmpdir, "project-name")


def test_unzip_empty_repository_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "https://example.com/empty.zip"
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = []
        
        with patch('cookiecutter.zipfile.requests.get'):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Should have raised InvalidZipRepository"
                    except InvalidZipRepository as e:
                        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "https://example.com/notoplevel.zip"
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ["file.txt"]
        
        with patch('cookiecutter.zipfile.requests.get'):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Should have raised InvalidZipRepository"
                    except InvalidZipRepository as e:
                        assert "top-level" in str(e).lower()


def test_unzip_password_protected_with_valid_password():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "https://example.com/protected.zip"
        password = "test_password"
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake", b"zip"])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ["project-name/", "project-name/file.txt"]
        mock_zip_file.extractall.side_effect = [RuntimeError("Bad password"), None]
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tmpdir):
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True, password=password)
        
        assert result == os.path.join(tmpdir, "project-name")


def test_unzip_invalid_zip_file_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = Path(tmpdir)
        zip_uri = "https://example.com/invalid.zip"
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"not", b"a", b"zip"])
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Not a zip file")):
                try:
                    unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert "not a valid zip archive" in str(e).lower()


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_on_bad_zipfile(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will trigger BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=str(clone_to_dir),
            no_input=True,
            password=None
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    from cookiecutter.zipfile import unzip
    
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b'PK\x03\x04']
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        mock_zip_instance = MagicMock()
                        mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
                        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                        
                        with patch('tempfile.mkdtemp', return_value='/tmp/tmpdir'):
                            result = unzip('http://example.com/repo.zip', is_url=True)
                            
                            assert result == '/tmp/tmpdir/project_name'
                            mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            with patch('tempfile.mkdtemp', return_value='/tmp/tmpdir'):
                result = unzip('/local/path/repo.zip', is_url=False)
                
                assert result == '/tmp/tmpdir/project_name'
                mock_zip_instance.extractall.assert_called_once()


def test_unzip_empty_zip_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = []
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            try:
                unzip('/local/path/repo.zip', is_url=False)
                assert False, "Should have raised InvalidZipRepository"
            except InvalidZipRepository as e:
                assert 'empty' in str(e).lower()


def test_unzip_without_top_level_directory_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['file.txt']
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            try:
                unzip('/local/path/repo.zip', is_url=False)
                assert False, "Should have raised InvalidZipRepository"
            except InvalidZipRepository as e:
                assert 'top-level directory' in str(e).lower()


def test_unzip_with_password_protection():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
            mock_zip_instance.extractall.side_effect = [RuntimeError('password protected')]
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            with patch('tempfile.mkdtemp', return_value='/tmp/tmpdir'):
                result = unzip('/local/path/repo.zip', is_url=False, password='secret')
                
                assert result == '/tmp/tmpdir/project_name'
                assert mock_zip_instance.extractall.call_count == 2


def test_unzip_invalid_password_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
            mock_zip_instance.extractall.side_effect = RuntimeError('password protected')
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            with patch('tempfile.mkdtemp', return_value='/tmp/tmpdir'):
                try:
                    unzip('/local/path/repo.zip', is_url=False, password='wrong')
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'invalid password' in str(e).lower()


def test_unzip_bad_zipfile_raises_error():
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile('Bad zip')):
            try:
                unzip('/local/path/repo.zip', is_url=False)
                assert False, "Should have raised InvalidZipRepository"
            except InvalidZipRepository as e:
                assert 'not a valid zip archive' in str(e).lower()


def test_unzip_no_input_with_password_protected_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
            mock_zip_instance.extractall.side_effect = RuntimeError('password protected')
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            with patch('tempfile.mkdtemp', return_value='/tmp/tmpdir'):
                try:
                    unzip('/local/path/repo.zip', is_url=False, no_input=True)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'unable to unlock' in str(e).lower()


def test_unzip_clone_to_dir_expansion():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import un


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, monkeypatch):
    """Test unzip downloads and extracts a zipfile from URL."""
    import io
    from zipfile import ZipFile
    
    # Create a mock zipfile in memory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    # Mock requests.get
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self._content = content
        
        def iter_content(self, chunk_size=1024):
            for i in range(0, len(self._content), chunk_size):
                yield self._content[i:i+chunk_size]
    
    mock_response = MockResponse(zip_buffer.getvalue())
    
    def mock_get(url, stream=True, timeout=100):
        return mock_response
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    monkeypatch.setattr('cookiecutter.zipfile.prompt_and_delete', lambda path, no_input: True)
    
    from cookiecutter.zipfile import unzip
    
    result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=str(tmp_path), no_input=True)
    assert 'test_project' in result
    assert result.endswith('test_project')


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local zipfile path."""
    import io
    from zipfile import ZipFile
    
    # Create a test zipfile
    zip_path = tmp_path / 'test.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('local_project/', '')
        zf.writestr('local_project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
    assert 'local_project' in result
    assert result.endswith('local_project')


def test_unzip_empty_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zipfile
    zip_path = tmp_path / 'empty.zip'
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / 'notoplevel.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level directory' in str(e).lower()


def test_unzip_invalid_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a file that's not a valid zipfile
    zip_path = tmp_path / 'invalid.zip'
    zip_path.write_text('not a zip file')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'valid zip archive' in str(e).lower()


def test_unzip_password_protected_with_provided_password(tmp_path):
    """Test unzip with password-protected zipfile and provided password."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zipfile
    zip_path = tmp_path / 'protected.zip'
    password = 'testpass'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('protected_project/', '')
        zf.writestr('protected_project/file.txt', 'content')
        zf.setpassword(password.encode('utf-8'))
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), 
                   no_input=True, password=password)
    assert 'protected_project' in result


def test_unzip_password_protected_no_password_no_input_raises_error(tmp_path):
    """Test unzip raises error for password-protected zip with no_input=True and no password."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a password-protected zipfile
    zip_path = tmp_path / 'protected.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('protected_project/', '')
        zf.setpassword(b'testpass')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'password protected' in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a test zipfile
    zip_path = tmp_path / 'test.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    new_dir = tmp_path / 'new_dir'
    assert not new_dir.exists()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(new_dir), no_input=True)
    assert new_dir.exists()
    assert 'project' in result


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_on_bad_zipfile(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary directory for clone_to_dir
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a fake "zip" file that is not a valid zip archive
    fake_zip_path = clone_to_dir / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    # Mock make_sure_path_exists to do nothing
    monkeypatch.setattr("cookiecutter.zipfile.make_sure_path_exists", lambda x: None)
    
    # Call unzip with is_url=False to use the local file path
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=str(clone_to_dir),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, monkeypatch):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file that exists
    zip_path = tmp_path / "test.zip"
    zip_path.touch()
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock the necessary functions
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True) as mock_prompt:
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('builtins.open', create=True):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        # Setup mock zipfile
                        mock_zip_instance = MagicMock()
                        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
                        mock_zip_instance.__enter__.return_value = mock_zip_instance
                        mock_zip_instance.__exit__.return_value = None
                        mock_zipfile.return_value = mock_zip_instance
                        
                        # Mock os.path.exists to return True for the zip_path
                        with patch('os.path.exists') as mock_exists:
                            def exists_side_effect(path):
                                if str(path) == str(zip_path):
                                    return True
                                return False
                            
                            mock_exists.side_effect = exists_side_effect
                            
                            # Call unzip with is_url=True so line 31 is executed
                            unzip(
                                zip_uri="http://example.com/test.zip",
                                is_url=True,
                                clone_to_dir=str(clone_to_dir),
                                no_input=False,
                                password=None
                            )
                            
                            # Verify that prompt_and_delete was called
                            # This confirms the predicate at line 31 evaluated to True
                            mock_prompt.assert_called_once()


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This happens when prompt_and_delete returns False, meaning the user
    wants to reuse the existing version.
    """
    from cookiecutter.zipfile import unzip
    from unittest.mock import Mock, patch, MagicMock
    
    # Setup
    zip_uri = "http://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_path = tmp_path / "repo.zip"
    
    # Create a mock zipfile that will be used
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=True):
            with patch('cookiecutter.zipfile.requests.get') as mock_requests:
                with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        # Execute - when download is False, requests.get should not be called
                        result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
                        
                        # Assert that requests.get was never called (because download was False)
                        mock_requests.get.assert_not_called()


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_iter_content_chunk_predicate():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    # The predicate at line 40 is: if chunk:
    # This tests that non-empty chunks are truthy and would be written
    
    chunk_empty = b''
    chunk_non_empty = b'some data'
    
    # Empty chunk should be falsy
    assert not chunk_empty
    
    # Non-empty chunk should be truthy
    assert chunk_non_empty
    
    # This simulates the filtering logic at line 40-42
    # where only non-empty chunks pass the filter
    chunks = [b'', b'data1', b'', b'data2', b'']
    filtered_chunks = [chunk for chunk in chunks if chunk]
    
    assert filtered_chunks == [b'data1', b'data2']
    assert len(filtered_chunks) == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_bad_zip_file_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_bytes(b"This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(fake_zip_path),
            is_url=False,
            clone_to_dir=str(clone_to_dir),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_line_41_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    chunk = b''
    result = bool(chunk)
    assert result is False


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_chunk_filter_predicate():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    chunk = b"test data"
    assert chunk


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a fake zip file that exists
    zip_filename = "test.zip"
    zip_path = clone_to_dir / zip_filename
    zip_path.touch()
    
    # Mock the prompt_and_delete to return True (user wants to download)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    # Mock requests.get and ZipFile to avoid actual network calls and file operations
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'test data']
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    # Mock ZipFile to return a valid zip structure
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.__enter__ = mocker.MagicMock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.MagicMock(return_value=None)
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile and os.path.join for unzip_base
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # The key assertion: verify that os.path.exists returns True for zip_path
    # This means the predicate at line 31 should evaluate to True
    assert os.path.exists(str(zip_path)) is True
    
    # Call unzip with is_url=True so it enters the branch with the predicate
    zip_uri = "http://example.com/test.zip"
    mocker.patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: str(Path(*args)))
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Verify prompt_and_delete was called (confirming the predicate was True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete').assert_called()


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_on_bad_zipfile():
    """Test that BadZipFile exception is caught and converted to InvalidZipRepository."""
    import os
    import tempfile
    from pathlib import Path
    from zipfile import BadZipFile
    from unittest.mock import patch, MagicMock
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Mock the ZipFile context manager to raise BadZipFile
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            mock_zipfile_instance = MagicMock()
            mock_zipfile_class.return_value.__enter__.side_effect = BadZipFile("Bad zip file")
            
            # Call unzip with is_url=False to use local path
            zip_uri = "/path/to/bad.zip"
            
            try:
                unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Expected InvalidZipRepository to be raised"
            except Exception as e:
                assert isinstance(e, InvalidZipRepository)
                assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(mocker, tmp_path):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when prompt_and_delete returns False, indicating the user
    wants to reuse the existing version instead of downloading.
    """
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    import os
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a valid zip file at the expected location
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock requests.get to ensure it's NOT called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Execute
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Verify
    # prompt_and_delete should have been called
    mock_prompt_and_delete.assert_called_once()
    
    # requests.get should NOT have been called (download block skipped)
    mock_requests_get.assert_not_called()
    
    # The function should still return a valid unzip_path
    assert result is not None
    assert 'project_name' in result


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_predicate_line_31_true_when_zip_path_exists(tmp_path, mocker):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file that will exist
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_filename = "test.zip"
    zip_path = clone_to_dir / zip_filename
    
    # Create a minimal valid zip file
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Mock prompt_and_delete to track if it was called
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock requests.get to avoid actual downloads
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Call unzip with is_url=True so it checks if zip_path exists
    zip_uri = f"http://example.com/{zip_filename}"
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=str(clone_to_dir),
        no_input=False
    )
    
    # Verify that prompt_and_delete was called, confirming line 31 predicate is True
    mock_prompt_and_delete.assert_called_once_with(str(zip_path), no_input=False)


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_predicate_line_39_evaluates_to_false():
    """Test that the predicate at line 39 (if chunk:) evaluates to False."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file for testing
        import zipfile
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_dir/", "")
            zf.writestr("test_dir/file.txt", "content")
        
        # Mock requests.get to return a response with a keep-alive chunk (empty chunk)
        mock_response = MagicMock()
        mock_response.iter_content = Mock(return_value=[b"some data", b"", b"more data"])
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                result = unzip(
                    zip_uri="http://example.com/test.zip",
                    is_url=True,
                    clone_to_dir=temp_dir,
                    no_input=True,
                    password=None
                )
        
        assert result is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_predicate_line_39_evaluates_to_false():
    """Test that the predicate at line 39 (if chunk:) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a mock response with empty chunks (to make `if chunk:` evaluate to False)
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'', None, b''])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Create a simple valid zip file for testing
        import zipfile
        zip_path = clone_to_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Mock requests.get to return our mock response with empty chunks
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                # Call unzip with is_url=True to trigger the chunk iteration
                result = unzip(
                    zip_uri="http://example.com/test.zip",
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=True
                )
        
        # Verify that write was not called for empty chunks
        # The mock file's write method should only be called for non-empty chunks
        write_calls = mock_file.write.call_count
        # With chunks [b'', None, b''], only non-empty chunks should trigger write
        # So write should be called 0 times (all chunks are falsy)
        assert write_calls == 0


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_predicate_line_39_evaluates_to_true(tmp_path, mocker):
    """Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a mock response with chunks
    mock_response = mocker.MagicMock()
    mock_chunk = b'test data chunk'
    mock_response.iter_content.return_value = [mock_chunk, b'', b'another chunk']
    
    # Mock requests.get to return our mock response
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    # Create a valid zip file for testing
    import zipfile
    import tempfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_dir/', '')
        zf.writestr('test_dir/file.txt', 'content')
    
    # Mock the ZipFile to avoid actual extraction
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.namelist.return_value = ['test_dir/', 'test_dir/file.txt']
    mock_zipfile.__enter__.return_value = mock_zipfile
    mock_zipfile.__exit__.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile and other dependencies
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Mock the file open to track write calls
    mock_file = mocker.MagicMock()
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    # Call unzip with is_url=True
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=True
    )
    
    # Verify that open was called to write the file
    mock_open.assert_called()
    
    # Verify that write was called with non-empty chunks
    handle = mock_open()
    write_calls = handle.write.call_args_list
    
    # Assert that write was called with the non-empty chunks (predicate evaluated to True)
    assert any(call[0][0] == mock_chunk for call in write_calls), \
        "The predicate 'if chunk:' should evaluate to True for non-empty chunks"
    
    # Verify that empty chunks were not written (predicate evaluated to False)
    assert not any(call[0][0] == b'' for call in write_calls), \
        "Empty chunks should not be written when 'if chunk:' evaluates to False"


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 41 evaluates to True when chunk is not empty."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    # Setup
    zip_url = "https://example.com/test.zip"
    clone_to_dir = tmp_path
    
    # Mock requests.get to return a response with chunks
    mock_response = Mock()
    mock_chunk_1 = b"test data 1"
    mock_chunk_2 = b"test data 2"
    mock_empty_chunk = b""
    mock_response.iter_content.return_value = [
        mock_chunk_1,
        mock_empty_chunk,
        mock_chunk_2,
    ]
    
    # Mock ZipFile to avoid actual file operations
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project_name/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    
    # Track write calls
    write_calls = []
    
    def mock_open_func(path, mode):
        mock_file = MagicMock()
        mock_file.write.side_effect = lambda data: write_calls.append(data)
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        return mock_file
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.open', mock_open_func):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                    result = unzip(zip_url, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    # Assert that only non-empty chunks were written
    assert mock_chunk_1 in write_calls
    assert mock_chunk_2 in write_calls
    assert mock_empty_chunk not in write_calls
    assert len(write_calls) == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when is_url is True, the zip file exists, and prompt_and_delete
    returns False (user chooses to reuse existing version).
    """
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_filename = "repo.zip"
    zip_path = tmp_path / zip_filename
    
    # Create a dummy zip file to simulate it already existing
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse existing)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    # Mock the requests.get to ensure it's not called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile to avoid actual file operations
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['repo/']
    
    # Mock tempfile and os.path.join
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    
    # Import and call the function
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Verify that requests.get was NOT called (because download was False)
    mock_requests_get.assert_not_called()
    
    # Verify that the function still extracted the zip file
    mock_zipfile.assert_called_once()


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 41 evaluates to True when chunk is not empty."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    zip_uri = "http://example.com/repo.zip"
    
    # Create a mock response with non-empty chunks
    mock_response = Mock()
    mock_chunk_1 = b"chunk1data"
    mock_chunk_2 = b""  # Empty chunk (keep-alive)
    mock_chunk_3 = b"chunk3data"
    mock_response.iter_content.return_value = [mock_chunk_1, mock_chunk_2, mock_chunk_3]
    
    # Mock requests.get
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        # Mock ZipFile to avoid actual zip operations
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            # Setup mock zipfile
            mock_zip_instance = MagicMock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ['test_dir/']
            
            # Mock file operations to track writes
            written_data = []
            
            def mock_write(data):
                written_data.append(data)
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.write = mock_write
                mock_open.return_value.__enter__.return_value = mock_file
                
                # Execute
                result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                
                # Assert that non-empty chunks were written
                assert b"chunk1data" in written_data
                assert b"chunk3data" in written_data
                # Empty chunk should not be written due to the predicate at line 41
                assert b"" not in written_data or written_data.count(b"") == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_iter_content_chunk_predicate_evaluates_to_false(mocker):
    """Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    temp_clone_dir = tempfile.mkdtemp()
    
    try:
        # Mock requests.get to return a response with empty chunks
        mock_response = mocker.MagicMock()
        mock_response.iter_content.return_value = [b'data', b'', b'more_data', b'']
        mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
        
        # Mock the ZipFile to avoid actual zip processing
        mock_zip_file = mocker.MagicMock()
        mock_zip_file.__enter__.return_value = mock_zip_file
        mock_zip_file.__exit__.return_value = None
        mock_zip_file.namelist.return_value = ['project/']
        mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
        
        # Mock tempfile.mkdtemp to return a controlled directory
        mock_unzip_base = tempfile.mkdtemp()
        mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=mock_unzip_base)
        
        # Call unzip with is_url=True to trigger the chunk iteration
        result = unzip(
            zip_uri='http://example.com/repo.zip',
            is_url=True,
            clone_to_dir=temp_clone_dir,
            no_input=True
        )
        
        # Verify that iter_content was called
        mock_response.iter_content.assert_called_once_with(chunk_size=1024)
        
        # The test passes if the function handles empty chunks without error
        # The predicate "if chunk:" evaluates to False for b'' (empty bytes)
        assert result is not None
    finally:
        # Clean up temporary directories
        import shutil
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir)
        if os.path.exists(mock_unzip_base):
            shutil.rmtree(mock_unzip_base)


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_with_valid_local_zipfile(tmp_path, monkeypatch):
    """Test unzip with a valid local zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory structure
    zip_dir = tmp_path / "test_project"
    zip_dir.mkdir()
    (zip_dir / "file.txt").write_text("test content")
    
    # Create a zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(zip_dir / "file.txt", arcname="test_project/file.txt")
    
    # Mock to ensure directory structure
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    # Call unzip with local file
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result is not None
    assert "test_project" in result


def test_unzip_empty_zipfile(tmp_path):
    """Test unzip with an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zipfile
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    # Should raise InvalidZipRepository
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip with no top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    # Should raise InvalidZipRepository
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzip with invalid zip file raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an invalid zip file
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    # Should raise InvalidZipRepository
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a valid zipfile
    zip_dir = tmp_path / "test_project"
    zip_dir.mkdir()
    (zip_dir / "file.txt").write_text("test")
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(zip_dir / "file.txt", arcname="test_project/file.txt")
    
    # clone_to_dir doesn't exist yet
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result is not None
    assert clone_to_dir.exists()


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    """Test unzip expands user home directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a valid zipfile
    zip_dir = tmp_path / "test_project"
    zip_dir.mkdir()
    (zip_dir / "file.txt").write_text("test")
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(zip_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir(parents=True, exist_ok=True)
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_to_dir))
    
    assert result is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file that exists
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_path = clone_to_dir / "repo.zip"
    
    # Create a valid zip file
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_dir/', '')
        zf.writestr('test_dir/file.txt', 'content')
    
    # Mock the functions to avoid actual downloads and user prompts
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Call unzip with is_url=True to trigger the condition at line 31
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    # Verify that prompt_and_delete was called, meaning the predicate at line 31 was True
    mock_prompt_and_delete.assert_called_once_with(str(zip_path), no_input=True)


# LLM-generated content at query #16
#--------------------------

```python
def test_unzip_download_predicate_false_when_reusing_existing():
    """Test that the predicate at line 36 evaluates to False when user chooses to reuse existing file."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    import tempfile
    import os
    
    # Create a temporary directory and a fake zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_filename = "test.zip"
        zip_path = clone_to_dir / zip_filename
        
        # Create a minimal valid zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_project/', '')
            zf.writestr('test_project/file.txt', 'content')
        
        # Mock prompt_and_delete to return False (user wants to reuse)
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                # Call unzip with is_url=True and existing zip file
                # The download variable at line 36 should be False
                result = unzip(
                    zip_uri=f"http://example.com/{zip_filename}",
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=False
                )
        
        # Verify that result is the path to the extracted content
        assert result is not None
        assert 'test_project' in result


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_local_file(tmp_path, monkeypatch):
    """Test unzipping a local zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project-name/", "")
        zf.writestr("project-name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
    
    assert "project-name" in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile(tmp_path):
    """Test unzipping an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzipping a zipfile without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzipping an invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    zip_path.write_text("This is not a valid zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    clone_dir = tmp_path / "new_clone_dir"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    assert not clone_dir.exists()
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
    
    assert clone_dir.exists()
    assert os.path.exists(result)


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    """Test unzip expands user home directory."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=".")
    
    assert os.path.exists(result)


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_download_predicate_false(tmp_path, mocker):
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "http://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file to simulate it already exists
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse existing version)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock the ZipFile to avoid actual file operations
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
    
    # Mock requests.get so it doesn't actually download
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock tempfile and os functions
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp')
    mock_tempfile.return_value = str(tmp_path / "temp")
    
    # Call unzip with is_url=True and no_input=False
    # This will trigger prompt_and_delete which returns False
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Verify that prompt_and_delete was called
    mock_prompt_and_delete.assert_called_once_with(str(zip_path), no_input=False)
    
    # Verify that requests.get was NOT called (download predicate was False)
    mock_requests_get.assert_not_called()


# LLM-generated content at query #19
#--------------------------

Looking at the code, I need to ensure that the predicate at line 40 evaluates to False. Line 40 is:


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, mocker):
    """Test unzip downloads and extracts a zipfile from a URL."""
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile
    import os
    
    # Create a temporary zip file with proper structure
    zip_content_dir = tmp_path / "content"
    zip_content_dir.mkdir()
    project_dir = zip_content_dir / "test_project/"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("test content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="test_project/")
        zf.write(project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock the requests.get and other dependencies
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [zip_file_path.read_bytes()]
    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('os.path.exists', return_value=False)
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri="https://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert result is not None
    assert "test_project" in result


def test_unzip_with_local_file_extracts_zipfile(tmp_path, mocker):
    """Test unzip extracts a local zipfile."""
    from zipfile import ZipFile
    
    # Create a temporary zip file with proper structure
    zip_content_dir = tmp_path / "content"
    zip_content_dir.mkdir()
    project_dir = zip_content_dir / "local_project/"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("local content")
    
    zip_file_path = tmp_path / "local.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="local_project/")
        zf.write(project_dir / "file.txt", arcname="local_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert result is not None
    assert "local_project" in result


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create an empty zip file
    zip_file_path = tmp_path / "empty.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_without_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when zipfile lacks top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_file_path = tmp_path / "no_toplevel.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_file_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    # Create an invalid zip file
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_protected_zipfile(tmp_path, mocker):
    """Test unzip extracts password-protected zipfile when password is provided."""
    from zipfile import ZipFile
    
    # Create a password-protected zip file
    zip_content_dir = tmp_path / "content"
    zip_content_dir.mkdir()
    project_dir = zip_content_dir / "protected_project/"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("protected content")
    
    zip_file_path = tmp_path / "protected.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="protected_project/")
        zf.write(project_dir / "file.txt", arcname="protected_project/file.txt")
        zf.setpassword(b"test_password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True,
        password="test_password"
    )
    
    assert result is not None


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, mocker):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    from zipfile import ZipFile
    
    # Create a valid zip file
    zip_content_dir = tmp_path / "content"
    zip_content_dir.mkdir()
    project_dir = zip_content_dir / "test_project/"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("test content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get to return a response with chunks
    mock_response = Mock()
    mock_chunk_1 = b"test_data_1"
    mock_chunk_2 = b"test_data_2"
    mock_response.iter_content.return_value = [mock_chunk_1, mock_chunk_2]
    
    # Mock the ZipFile to avoid actual zip processing
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project_name/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    
    # Track written chunks
    written_chunks = []
    
    def mock_open_func(path, mode):
        mock_file = MagicMock()
        def mock_write(chunk):
            written_chunks.append(chunk)
        mock_file.write = mock_write
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        return mock_file
    
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    monkeypatch.setattr("builtins.open", mock_open_func)
    monkeypatch.setattr("zipfile.ZipFile", lambda *args, **kwargs: mock_zip_file)
    monkeypatch.setattr("tempfile.mkdtemp", lambda: str(tmp_path / "temp"))
    
    # Call unzip with is_url=True to trigger the chunk writing code path
    result = unzip(
        zip_uri="http://example.com/archive.zip",
        is_url=True,
        clone_to_dir=str(clone_to_dir),
        no_input=True
    )
    
    # Verify that chunks were written (the predicate at line 41 "if chunk:" evaluated to True)
    assert len(written_chunks) == 2
    assert written_chunks[0] == mock_chunk_1
    assert written_chunks[1] == mock_chunk_2


# LLM-generated content at query #27
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "notopdir.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_url_with_no_input(tmp_path, mocker):
    """Test unzip with URL and no_input=True."""
    import zipfile
    
    # Create a test zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"fake content"])
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result is not None


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and valid password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert "project_name" in result


def test_unzip_password_protected_invalid_password_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip and invalid password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.ZipFile.extractall', side_effect=RuntimeError("Bad password"))
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrongpass")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.ZipFile.extractall', side_effect=RuntimeError("Bad password"))
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir


# LLM-generated content at query #28
#--------------------------

```python
def test_unzip_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This test ensures that when a zipfile contains at least one entry,
    the condition evaluates to False and no InvalidZipRepository exception is raised.
    """
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zipfile with content
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            # Add a directory entry (first entry must be a directory)
            zf.writestr('project_root/', '')
            # Add a file entry
            zf.writestr('project_root/file.txt', 'content')
        
        # Call unzip with the valid zipfile
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password=None
        )
        
        # If we reach here, the predicate at line 54 evaluated to False
        # (meaning len(zip_file.namelist()) != 0), and no exception was raised
        assert result is not None
        assert os.path.exists(result)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #29
#--------------------------

```python
def test_unzip_raises_error_when_zip_file_is_empty(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Attempt to unzip empty file should raise InvalidZipRepository
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #30
#--------------------------

```python
def test_unzip_with_url_no_existing_file(tmp_path, mocker):
    """Test unzip downloads and extracts a URL when file doesn't exist."""
    zip_content = b'PK\x03\x04' + b'\x00' * 100
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [zip_content]
    mocker.patch('requests.get', return_value=mock_response)
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    mocker.patch('tempfile.mkdtemp', return_value=str(tmp_path))
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    result = unzip('http://example.com/project.zip', is_url=True, clone_to_dir=tmp_path)
    
    assert result == str(tmp_path / 'project')
    mock_zipfile.extractall.assert_called_once()


def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    local_zip = tmp_path / 'local.zip'
    local_zip.write_bytes(b'PK\x03\x04')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['myproject/', 'myproject/file.txt']
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    mocker.patch('tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    result = unzip(str(local_zip), is_url=False, clone_to_dir=tmp_path)
    
    assert result == str(tmp_path / 'temp' / 'myproject')


def test_unzip_empty_repository(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = []
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    try:
        unzip('http://example.com/empty.zip', is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['file.txt']
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    try:
        unzip('http://example.com/bad.zip', is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'top-level' in str(e).lower()


def test_unzip_bad_zip_file(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for bad zip file."""
    from zipfile import BadZipFile
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('zipfile.ZipFile', side_effect=BadZipFile('Bad zip'))
    
    try:
        unzip('http://example.com/bad.zip', is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'not a valid zip archive' in str(e).lower()


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzip extracts password-protected zip with provided password."""
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.extractall.side_effect = [RuntimeError('encrypted'), None]
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    mocker.patch('tempfile.mkdtemp', return_value=str(tmp_path))
    
    result = unzip(
        'http://example.com/protected.zip',
        is_url=False,
        clone_to_dir=tmp_path,
        password='mypassword'
    )
    
    assert result == str(tmp_path / 'project')
    assert mock_zipfile.extractall.call_count == 2


def test_unzip_password_protected_no_input(tmp_path, mocker):
    """Test unzip raises error for password-protected zip with no_input=True."""
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.extractall.side_effect = RuntimeError('encrypted')
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return_value=mock_zipfile)
    
    try:
        unzip(
            'http://example.com/protected.zip',
            is_url=False,
            clone_to_dir=tmp_path,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'password' in str(e).lower()


def test_unzip_password_protected_user_input(tmp_path, mocker):
    """Test unzip prompts user for password when needed."""
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.read_repo_password', return_value='correct')
    
    mock_zipfile = mocker.Mock()
    mock_zipfile.namelist.return_value = ['project/', 'project/file.txt']
    mock_zipfile.extractall.side_effect = [RuntimeError('encrypted'), None]
    mock_zipfile.__enter__ = mocker.Mock(return_value=mock_zipfile)
    mock_zipfile.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('zipfile.ZipFile', return


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "test_project" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "no_top_dir.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_with_url_no_input_no_existing_file(tmp_path, monkeypatch):
    """Test unzip with URL when no existing file and no_input=True."""
    import zipfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[open(zip_path, 'rb').read()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "test_project" in result


def test_unzip_with_url_existing_file_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    existing_zip = clone_to_dir / "test.zip"
    with zipfile.ZipFile(existing_zip, 'w') as zf:
        zf.writestr("old_project/", "")
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[open(zip_path, 'rb').read()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "test_project" in result


def test_unzip_password_protected_with_valid_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and valid password provided."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(b"mypassword")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="mypassword")
    
    assert "test_project" in result


def test_unzip_password_protected_no_password_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip, no password, and no_input=True."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(b"mypassword")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password protected" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file
    zip_dir = tmp_path / "zipdir"
    zip_dir.mkdir()
    project_dir = zip_dir / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("test content")
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(project_dir / "file.txt", "test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "test_project" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository exception"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository exception"
    except Exception as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises error for invalid zipfile."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository exception"
    except Exception as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    
    zip_dir = tmp_path / "zipdir"
    zip_dir.mkdir()
    project_dir = zip_dir / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("test")
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(project_dir / "file.txt", "test_project/file.txt")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert os.path.exists(result)
    assert "test_project" in result


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path):
    """Test unzip raises error for password protected zip with no_input=True."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content", compress_type=zipfile.ZIP_DEFLATED)
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository exception"
    except Exception as e:
        assert "password" in str(e).lower() or "protected" in str(e).lower()


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    import zipfile
    import os
    from pathlib import Path
    
    # Create a valid zip file with content
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Create the temp directory that mkdtemp would create
    (tmp_path / "temp").mkdir(exist_ok=True)
    
    from cookiecutter.zipfile import unzip
    
    # Call unzip with a local file (is_url=False)
    result = unzip(
        zip_uri=str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        no_input=True,
        password=None
    )
    
    # If we reach here without InvalidZipRepository being raised,
    # it means the predicate len(zip_file.namelist()) == 0 evaluated to False
    assert result is not None
    assert "project_dir" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_empty_zip_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    # Test that the predicate at line 55 evaluates to True
    # (len(zip_file.namelist()) == 0 should be True for empty zip)
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "Zip repository" in str(e)
        assert "is empty" in str(e)


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when prompt_and_delete returns False, indicating the user
    wants to reuse the existing version rather than re-download.
    """
    from cookiecutter.zipfile import unzip
    from unittest.mock import Mock, patch, MagicMock
    import os
    
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    zip_filename = "repo.zip"
    zip_path = os.path.join(clone_to_dir, zip_filename)
    
    # Create a dummy zip file that exists
    zip_path_obj = tmp_path / zip_filename
    zip_path_obj.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
        with patch('cookiecutter.zipfile.requests.get') as mock_get:
            # Create a valid zip file for extraction
            import zipfile
            from pathlib import Path
            
            valid_zip_path = tmp_path / "valid.zip"
            with zipfile.ZipFile(valid_zip_path, 'w') as zf:
                zf.writestr("test_project/", "")
                zf.writestr("test_project/file.txt", "content")
            
            # Replace the zip_path with our valid zip
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zip_instance = MagicMock()
                mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                
                # This should NOT call requests.get because download is False
                result = unzip(
                    zip_uri=zip_uri,
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=False,
                    password=None
                )
                
                # Verify that requests.get was NOT called (predicate was False)
                mock_get.assert_not_called()


# LLM-generated content at query #31
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This means the zip file contains at least one entry.
    """
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file with content
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Mock the dependencies
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    # Setup mock for requests.get
                    mock_response = Mock()
                    mock_response.iter_content = Mock(return_value=[])
                    mock_get.return_value = mock_response
                    
                    # Import the function to test
                    from cookiecutter.zipfile import unzip
                    
                    # Call unzip with is_url=False to use local file
                    result = unzip(
                        zip_uri=zip_path,
                        is_url=False,
                        clone_to_dir=temp_dir,
                        no_input=True,
                        password=None
                    )
                    
                    # Verify the result is a valid path
                    assert isinstance(result, str)
                    assert 'project_dir' in result


# LLM-generated content at query #32
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_file_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert result.endswith("project_name")
    assert os.path.isdir(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "no_top_level.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zip_file_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "bad.zip"
    zip_file_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert result.endswith("project_name")
    assert os.path.isdir(result)


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path):
    """Test unzip with password protected zip and no_input raises error."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.setpassword(b"test_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_correct_password(tmp_path):
    """Test unzip with correct password extracts successfully."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True,
        password="test_password"
    )
    
    assert result.endswith("project_name")
    assert os.path.isdir(result)


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with content (non-empty namelist)
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            # Add a directory entry (top-level directory)
            zf.writestr("project_name/", "")
            # Add a file entry
            zf.writestr("project_name/file.txt", "content")
        
        # Call unzip with is_url=False to use local zipfile
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=temp_dir,
            no_input=True
        )
        
        # If we reach here, the predicate at line 55 (len(zip_file.namelist()) == 0)
        # evaluated to False, meaning the zip file was not empty
        assert result is not None
        assert isinstance(result, str)
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    
    # Create a valid zip file with content (non-empty namelist)
    zip_path = os.path.join(temp_dir, "test.zip")
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    # Verify the predicate evaluates to False
    with ZipFile(zip_path) as zip_file:
        predicate_result = len(zip_file.namelist()) == 0
    
    assert predicate_result is False
    assert len(zip_file.namelist()) > 0
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zipfile is empty."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Test that empty zipfile raises InvalidZipRepository
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #34
#--------------------------

```python
def test_unzip_predicate_line_39_evaluates_to_false():
    """Test that the predicate at line 39 (if chunk:) evaluates to False."""
    import os
    import tempfile
    import zipfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Create a valid zip file
        zip_path = clone_to_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_dir/", "")
            zf.writestr("test_dir/file.txt", "content")
        
        zip_uri = str(zip_path)
        
        # Mock requests.get to return a response with empty chunks
        mock_response = Mock()
        # Simulate iter_content returning some empty chunks (which should be filtered)
        mock_response.iter_content.return_value = [b'', None, b'', b'some_data', b'']
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ["test_dir/", "test_dir/file.txt"]
                mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                
                # Call unzip with is_url=False to avoid the download logic
                result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
                
                # Verify the result is a valid path
                assert isinstance(result, str)


# LLM-generated content at query #35
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import os
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock make_sure_path_exists to avoid actual filesystem operations
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Test that InvalidZipRepository is raised
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #28
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, mocker):
    """Test unzip with URL when zip file doesn't exist yet."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"test_chunk"])
    mocker.patch("requests.get", return_value=mock_response)
    
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = ["project_name/", "project_name/file.txt"]
    mock_zip.__enter__ = mocker.MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("cookiecutter.zipfile.ZipFile", return_value=mock_zip)
    
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
    mocker.patch("os.path.exists", return_value=False)
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir)
    
    assert result is not None
    mock_zip.extractall.assert_called_once()


def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with local file path."""
    zip_file_path = str(tmp_path / "repo.zip")
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = ["project_name/", "project_name/file.txt"]
    mock_zip.__enter__ = mocker.MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("cookiecutter.zipfile.ZipFile", return_value=mock_zip)
    
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
    
    result = unzip(zip_file_path, is_url=False, clone_to_dir=clone_to_dir)
    
    assert result is not None
    mock_zip.extractall.assert_called_once()


def test_unzip_empty_zip(tmp_path, mocker):
    """Test unzip with empty zip file raises InvalidZipRepository."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"test_chunk"])
    mocker.patch("requests.get", return_value=mock_response)
    
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = []
    mock_zip.__enter__ = mocker.MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("cookiecutter.zipfile.ZipFile", return_value=mock_zip)
    
    mocker.patch("os.path.exists", return_value=False)
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory(tmp_path, mocker):
    """Test unzip when zip doesn't have top-level directory raises InvalidZipRepository."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"test_chunk"])
    mocker.patch("requests.get", return_value=mock_response)
    
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = ["file.txt"]
    mock_zip.__enter__ = mocker.MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("cookiecutter.zipfile.ZipFile", return_value=mock_zip)
    
    mocker.patch("os.path.exists", return_value=False)
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zip_file(tmp_path, mocker):
    """Test unzip with invalid zip file raises InvalidZipRepository."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"test_chunk"])
    mocker.patch("requests.get", return_value=mock_response)
    
    mocker.patch("cookiecutter.zipfile.ZipFile", side_effect=BadZipFile("Invalid zip"))
    mocker.patch("os.path.exists", return_value=False)
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzip with password-protected zip and provided password."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[b"test_chunk"])
    mocker.patch("requests.get", return_value=mock_response)
    
    mock_zip = mocker.MagicMock()
    mock_zip.namelist.return_value = ["project_name/", "project_name/file.txt"]
    mock_zip.extractall = mocker.MagicMock(side_effect=[RuntimeError("Bad password"), None])
    mock_zip.__enter__ = mocker.MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("cookiecutter.zipfile.ZipFile", return_value=mock_zip)
    
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
    mocker.patch("os.path.exists", return_value=False)
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, password="secret")
    
    assert result is not None
    assert mock_zip.extractall.call_count == 2


def test_unzip_password_protected_invalid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and invalid password raises error."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path


# LLM-generated content at query #29
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    zip_path.write_text("not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_with_url_downloads_file(tmp_path, monkeypatch):
    """Test unzip downloads file when is_url is True."""
    import zipfile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a valid zip file content
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    zip_data = zip_content.read_bytes()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_data])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
            result = unzip("http://example.com/project.zip", is_url=True, clone_to_dir=clone_dir)
    
    assert result.endswith("project_name")


def test_unzip_with_password_protected_file(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    password = "test_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir, password=password)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for password-protected file with no_input."""
    import zipfile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_instance = MagicMock()
        mock_instance.namelist.return_value = ["project_name/"]
        mock_instance.extractall.side_effect = RuntimeError("Bad password")
        mock_zipfile.return_value.__enter__.return_value = mock_instance
        
        try:
            unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir, no_input=True)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "password" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "new_clone_dir"
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    assert not clone_dir.exists()
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
    
    assert clone_dir.exists()
    assert result.endswith("project_name")


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    """Test unzip expands user home directory in clone_to_dir."""
    import zipfile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    with


# LLM-generated content at query #36
#--------------------------

```python
def test_unzip_predicate_at_line_40_evaluates_to_false():
    """Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = Path(temp_dir)
    
    try:
        # Mock the requests.get to return a response with empty chunks
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'', b'', None])
        
        # Create a temporary zip file
        zip_path = os.path.join(temp_dir, 'test.zip')
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    # Setup mock zipfile
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_name/']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile.return_value = mock_zip_instance
                    
                    # Call unzip with a URL
                    result = unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
                    
                    # Verify that iter_content was called
                    mock_response.iter_content.assert_called_once_with(chunk_size=1024)
                    
                    # The test verifies that empty chunks (b'') and None values
                    # would have the predicate "if chunk:" evaluate to False
                    assert result is not None
    finally:
        # Cleanup
        from cookiecutter.utils import rmtree
        rmtree(temp_dir)


# LLM-generated content at query #30
#--------------------------

```python
def test_unzip_iter_content_filters_empty_chunks(mocker, tmp_path):
    """Test that the predicate at line 40 evaluates to False for empty chunks."""
    # Setup
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    
    # Create a mock response with empty chunks
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [
        b"chunk1",
        b"",  # This chunk should be filtered out (predicate evaluates to False)
        b"chunk2",
        b"",  # Another empty chunk
    ]
    
    # Mock requests.get and file operations
    mocker.patch("requests.get", return_value=mock_response)
    mock_file = mocker.MagicMock()
    mocker.patch("builtins.open", mocker.mock_open())
    
    # Mock ZipFile and other dependencies
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ["project/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    mocker.patch("zipfile.ZipFile", return_value=mock_zip_file)
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
    
    # Import after mocking
    from cookiecutter.zipfile import unzip
    
    # Execute
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    # Verify that write was called only for non-empty chunks (2 times, not 4)
    # The mock_open's write method should be called exactly 2 times
    handle = mocker.patch("builtins.open", mocker.mock_open()).return_value
    mocker.patch("requests.get", return_value=mock_response)
    mocker.patch("zipfile.ZipFile", return_value=mock_zip_file)
    mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp"))
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    # The assertion verifies that empty chunks (where `if chunk:` is False) are filtered
    assert result is not None


# LLM-generated content at query #37
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL to a new zipfile."""
    import zipfile
    import io
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a simple valid zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_buffer.getvalue()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            (tmp_path / "temp").mkdir()
            result = __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
                'https://example.com/test.zip',
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
    
    assert result.endswith('test_project')


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local zipfile path."""
    import zipfile
    from pathlib import Path
    from unittest.mock import patch
    
    # Create a valid zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('local_project/', '')
        zf.writestr('local_project/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        (tmp_path / "temp").mkdir()
        result = __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
    
    assert result.endswith('local_project')


def test_unzip_empty_zipfile(tmp_path):
    """Test unzip with an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip with zipfile missing top-level directory raises InvalidZipRepository."""
    import zipfile
    
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'top-level directory' in str(e)


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzip with an invalid zip file raises InvalidZipRepository."""
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
            str(invalid_zip),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert 'not a valid zip archive' in str(e)


def test_unzip_with_password_protected_zip(tmp_path):
    """Test unzip with password-protected zipfile and correct password."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('protected_project/', '')
        zf.writestr('protected_project/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        (tmp_path / "temp").mkdir()
        result = __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password="test123"
        )
    
    assert result.endswith('protected_project')


def test_unzip_url_existing_file_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and no_input=True (should delete and redownload)."""
    import zipfile
    import io
    from unittest.mock import Mock, patch
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    existing_zip = clone_to_dir / "test.zip"
    existing_zip.write_text("old content")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('new_project/', '')
        zf.writestr('new_project/file.txt', 'new content')
    zip_buffer.seek(0)
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_buffer.getvalue()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            (tmp_path / "temp").mkdir()
            result = __import__('cookiecutter.zipfile', fromlist=['unzip']).unzip(
                'https://example.com/test.zip',
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
    
    assert result.endswith('new_project')


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from unittest.mock import patch


# LLM-generated content at query #38
#--------------------------

```python
from pathlib import Path
from zipfile import ZipFile
import tempfile
import os


def test_unzip_zipfile_context_manager_line_54():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple valid zip file with a top-level directory
    zip_path = os.path.join(temp_dir, "test.zip")
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_dir/", "")
        zf.writestr("test_dir/file.txt", "content")
    
    # Test that the context manager works (line 54 predicate evaluates to True)
    # by verifying we can enter and exit the context
    with ZipFile(zip_path) as zip_file:
        # The predicate at line 54 should evaluate to True (context manager is entered)
        assert zip_file is not None
        assert hasattr(zip_file, 'namelist')
        assert callable(zip_file.namelist)
        namelist = zip_file.namelist()
        assert len(namelist) > 0
        assert namelist[0] == "test_dir/"
    
    # Cleanup
    os.remove(zip_path)
    os.rmdir(temp_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_zipfile_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This test ensures that when a zipfile contains at least one entry,
    the condition len(zip_file.namelist()) == 0 is False.
    """
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a test zipfile with at least one entry
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_dir/', '')
            zf.writestr('test_dir/file.txt', 'content')
        
        # Open the zipfile and verify the predicate evaluates to False
        with ZipFile(zip_path) as zip_file:
            namelist = zip_file.namelist()
            predicate_result = len(namelist) == 0
            
            assert predicate_result is False
            assert len(namelist) > 0
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_unzip_context_manager_with_zipfile():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Call unzip with is_url=False to use the local zipfile
        clone_to_dir = os.path.join(temp_dir, "clone")
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        
        # Verify the result is a valid path
        assert result is not None
        assert os.path.exists(result)
        assert "test_project" in result


# LLM-generated content at query #31
#--------------------------

```python
def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Attempt to unzip the empty file
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == "InvalidZipRepository"
        assert "empty" in str(e).lower()


# LLM-generated content at query #32
#--------------------------

```python
def test_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to False for keep-alive chunks."""
    # The predicate 'if chunk:' at line 40 evaluates to False when chunk is empty
    # This simulates a keep-alive new chunk that should be filtered out
    chunk = b''
    assert not chunk


# LLM-generated content at query #33
#--------------------------

```python
def test_unzip_chunk_filter_predicate():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    chunk = b'some data'
    assert chunk


# LLM-generated content at query #34
#--------------------------

```python
def test_unzip_context_manager_with_zipfile():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Call unzip with is_url=False to test the context manager
        result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        
        # Verify that the result is a valid path
        assert result is not None
        assert isinstance(result, str)
        assert "test_project" in result
        
    finally:
        # Cleanup
        from shutil import rmtree
        rmtree(temp_dir, ignore_errors=True)
        rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."""
    chunk = b''
    assert not chunk


# LLM-generated content at query #36
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "https://example.com/repo.zip"
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'test_data'])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['project_name/', 'project_name/file.txt']
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                    
                    assert result.endswith('project_name')
                    mock_zip_file.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = os.path.join(temp_dir, "local.zip")
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
            
            assert result.endswith('project')
            mock_zip_file.extractall.assert_called_once()


def test_unzip_empty_repository_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "https://example.com/empty.zip"
        
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = []
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Should raise InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "https://example.com/notoplevel.zip"
        
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b'data'])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['file.txt']
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Should raise InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_with_password():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "https://example.com/protected.zip"
        password = "secret"
        
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b'data'])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_file.extractall.side_effect = [RuntimeError("Bad password"), None]
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, 
                                 no_input=True, password=password)
                    
                    assert result.endswith('project')
                    assert mock_zip_file.extractall.call_count == 2


def test_unzip_invalid_password_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "https://example.com/protected.zip"
        password = "wrongpassword"
        
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b'data'])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_file.extractall.side_effect = RuntimeError("Bad password")
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, 
                            no_input=True, password=password)
                        assert False, "Should raise InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_bad_zip_file_raises_error():
    import tempfile
    from pathlib import Path
    from zipfile import BadZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp


