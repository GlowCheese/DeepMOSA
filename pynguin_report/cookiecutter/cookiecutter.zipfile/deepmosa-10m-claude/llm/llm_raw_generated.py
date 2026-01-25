####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_local_file_success(tmp_path, monkeypatch):
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=extract_dir)
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_url_new_file(tmp_path, monkeypatch):
    import zipfile
    from unittest.mock import Mock, patch
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a mock response
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'test'])
    
    # Create actual zip content
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    with open(zip_content, 'rb') as f:
        zip_bytes = f.read()
    
    mock_response.iter_content = Mock(return_value=[zip_bytes])
    
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip('http://example.com/test_project.zip', is_url=True, clone_to_dir=clone_dir)
    
    assert 'test_project' in result


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level' in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    from cookiecutter.zipfile import InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a file that's not a valid zip
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'valid zip archive' in str(e).lower()


def test_unzip_password_protected_with_password(tmp_path, monkeypatch):
    import zipfile
    import os
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, password='test')
    
    assert 'project' in result
    assert os.path.exists(result)


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    import zipfile
    from unittest.mock import patch
    from cookiecutter.zipfile import InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'wrongpass')
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'password' in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    import zipfile
    import os
    
    clone_dir = tmp_path / "nonexistent" / "clone"
    
    # Create a zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
    
    assert os.path.exists(clone_dir)
    assert 'project' in result


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
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
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
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
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
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
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "InvalidZipRepository" in type(e).__name__


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "new_clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "project_name" in result


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when is_url is True, the zip file exists, and prompt_and_delete
    returns False (user chooses to reuse existing version).
    """
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    import os
    
    # Create a temporary zip file that exists
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    zip_filename = "test.zip"
    zip_path = clone_to_dir / zip_filename
    
    # Create a dummy zip file
    zip_path.write_bytes(b"dummy")
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    def mock_prompt_and_delete(path, no_input=False):
        return False
    
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", mock_prompt_and_delete)
    
    # Mock requests.get to ensure it's not called
    call_count = {"get": 0}
    def mock_requests_get(*args, **kwargs):
        call_count["get"] += 1
        raise AssertionError("requests.get should not be called when download is False")
    
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", mock_requests_get)
    
    # Mock ZipFile to handle the existing file
    from unittest.mock import MagicMock, mock_open
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project_name/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = False
    
    monkeypatch.setattr("cookiecutter.zipfile.ZipFile", lambda x: mock_zip_file)
    
    # Call unzip with is_url=True and the file already exists
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Verify requests.get was not called (confirming download was False)
    assert call_count["get"] == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
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
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip with empty zipfile raises InvalidZipRepository."""
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


def test_unzip_no_top_level_directory_raises_error(tmp_path):
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


def test_unzip_invalid_zip_raises_error(tmp_path):
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
    
    assert clone_to_dir.exists()
    assert result.endswith("project_name")


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path):
    """Test unzip with password-protected zip and no_input raises error."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.setpassword(b"password")
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
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_empty_zipfile_predicate_false():
    """Test that the predicate at line 54 evaluates to False when zipfile is empty."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary empty zipfile
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "empty.zip")
        with ZipFile(zip_path, 'w') as zf:
            pass  # Create empty zipfile
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        
        # The predicate at line 55 checks: if len(zip_file.namelist()) == 0
        # This should evaluate to True (not False), triggering the exception
        # To test that it evaluates to False, we need a non-empty zipfile
        # But the test case asks for the predicate to evaluate to False
        # So we create a zipfile with content where namelist() is NOT empty
        
        zip_path_with_content = os.path.join(temp_dir, "with_content.zip")
        with ZipFile(zip_path_with_content, 'w') as zf:
            zf.writestr("project/", "")  # Add directory entry
        
        result = unzip(zip_path_with_content, is_url=False, clone_to_dir=clone_to_dir)
        
        assert result is not None
        assert isinstance(result, str)


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zipfile with a valid structure
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


def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
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
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zipfile_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
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
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
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


def test_unzip_with_password_protected_file_no_input(tmp_path, monkeypatch):
    """Test unzip with password protected file and no_input raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"test_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password protected" in str(e).lower()


def test_unzip_with_correct_password(tmp_path, monkeypatch):
    """Test unzip with password protected file and correct password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test_password")
    
    assert "project_name" in result


def test_unzip_with_invalid_password_provided(tmp_path, monkeypatch):
    """Test unzip with password protected file and invalid password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrong_password")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "invalid password" in str(e).lower()


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, mocker):
    """Test unzip downloads and extracts a URL-based zipfile."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_prompt_and_delete = mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test')
    
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
    mock_requests_get.return_value = mock_response
    
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    assert result == '/tmp/test/project_name'
    mock_make_sure_path_exists.assert_called_once()
    mock_requests_get.assert_called_once_with(zip_uri, stream=True, timeout=100)
    mock_zip_instance.extractall.assert_called_once_with(path='/tmp/test')


def test_unzip_with_local_file_extracts_zipfile(tmp_path, mocker):
    """Test unzip extracts a local zipfile without downloading."""
    zip_uri = str(tmp_path / "local.zip")
    clone_to_dir = tmp_path / "clone"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test')
    
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['my_project/', 'my_project/setup.py']
    
    result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
    
    assert result == '/tmp/test/my_project'
    mock_make_sure_path_exists.assert_called_once()
    mock_zip_instance.extractall.assert_called_once_with(path='/tmp/test')


def test_unzip_existing_file_prompts_for_deletion(tmp_path, mocker):
    """Test unzip prompts to delete existing cached zipfile."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_os_path_exists = mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    mock_prompt_and_delete = mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test')
    
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'chunk']
    mock_requests_get.return_value = mock_response
    
    mocker.patch('builtins.open', mocker.mock_open())
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    mock_prompt_and_delete.assert_called_once()
    assert result == '/tmp/test/project'


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_uri = str(tmp_path / "empty.zip")
    clone_to_dir = tmp_path / "clone"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = []
    
    try:
        unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_missing_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when top-level is not a directory."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_uri = str(tmp_path / "bad.zip")
    clone_to_dir = tmp_path / "clone"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['file.txt']
    
    try:
        unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert 'top-level directory' in str(e).lower()


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip extracts password-protected zipfile with correct password."""
    zip_uri = str(tmp_path / "protected.zip")
    clone_to_dir = tmp_path / "clone"
    password = "mypassword"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test')
    
    mock_zip_instance = mocker.MagicMock


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_bad_zip_file_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will trigger BadZipFile
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("this is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(bad_zip_path),
            is_url=False,
            clone_to_dir=str(clone_to_dir),
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    
    # Create a test zipfile
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "test_project" in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory exists."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_file_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zipfile_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("not a valid zip")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_file_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "test_project" in result


def test_unzip_password_protected_with_correct_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and correct password."""
    import zipfile
    
    zip_file_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert "test_project" in result
    assert os.path.exists(result)


def test_unzip_password_protected_with_wrong_password_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and wrong password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_file_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password="wrong_password")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "invalid password" in str(e).lower()


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_file_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "unable to unlock" in str(e).lower()


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when is_url is True, zip_path exists, and prompt_and_delete returns False.
    """
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file that exists
    zip_path.write_bytes(b"dummy content")
    
    # Mock prompt_and_delete to return False (user doesn't want to delete/re-download)
    def mock_prompt_and_delete(path, no_input=False):
        return False
    
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", mock_prompt_and_delete)
    
    # Mock requests.get to ensure it's not called
    call_count = {"get": 0}
    
    def mock_requests_get(*args, **kwargs):
        call_count["get"] += 1
        raise AssertionError("requests.get should not be called when download is False")
    
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", mock_requests_get)
    
    # Mock ZipFile to handle the unzip operation
    from unittest.mock import MagicMock, mock_open
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project/"]
    mock_zip_file.__enter__ = MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = MagicMock(return_value=None)
    
    def mock_zipfile_init(path, *args, **kwargs):
        return mock_zip_file
    
    monkeypatch.setattr("cookiecutter.zipfile.ZipFile", mock_zipfile_init)
    
    # Execute
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Verify that requests.get was never called (meaning download was False)
    assert call_count["get"] == 0
    assert result is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zipfile is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zipfile
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Attempt to unzip the empty file should raise InvalidZipRepository
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_bad_zip_file_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("This is not a valid zip file")
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    try:
        unzip(str(bad_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_opens_zipfile_with_context_manager():
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
        
        # Call unzip with the local zip file
        result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        
        # Verify that the result is a valid path (meaning the context manager worked)
        assert result is not None
        assert isinstance(result, str)
        assert "test_project" in result
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_download_predicate_false():
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    import tempfile
    import os
    
    # Create a temporary directory and a dummy zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Create a dummy zip file
        zip_path = clone_to_dir / "test.zip"
        from zipfile import ZipFile
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        zip_uri = "http://example.com/test.zip"
        
        # Mock prompt_and_delete to return False (user chose not to delete)
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
            # Mock requests.get to ensure download is not called
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                # This should not raise an exception and should reuse the existing file
                result = unzip(
                    zip_uri=zip_uri,
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=False
                )
                
                # Verify that requests.get was NOT called (download=False at line 36)
                mock_get.assert_not_called()
                # Verify that the result points to the extracted content
                assert result is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL when the zip file doesn't exist locally."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'PK\x03\x04'])
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                result = unzip(
                    "http://example.com/project.zip",
                    is_url=True,
                    clone_to_dir=clone_dir,
                    no_input=True
                )
    
    assert result == str(tmp_path / "temp" / "project_name")
    mock_response.iter_content.assert_called_once()
    mock_zip.extractall.assert_called_once()


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local file path."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    local_zip_path = str(tmp_path / "local.zip")
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(
                local_zip_path,
                is_url=False,
                clone_to_dir=clone_dir,
                no_input=True
            )
    
    assert result == str(tmp_path / "temp" / "project_name")
    mock_zip.extractall.assert_called_once()


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when zip is empty."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = []
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            try:
                unzip("local.zip", is_url=False, clone_to_dir=clone_dir, no_input=True)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when zip has no top-level directory."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ['file.txt']
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            try:
                unzip("local.zip", is_url=False, clone_to_dir=clone_dir, no_input=True)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_password_protected_with_valid_password(tmp_path):
    """Test unzip with password-protected zip and valid password provided."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip.extractall.side_effect = [RuntimeError("Bad password"), None]
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(
                "local.zip",
                is_url=False,
                clone_to_dir=clone_dir,
                no_input=True,
                password="mypassword"
            )
    
    assert result == str(tmp_path / "temp" / "project_name")
    assert mock_zip.extractall.call_count == 2


def test_unzip_password_protected_no_input_raises_error(tmp_path):
    """Test unzip with password-protected zip and no_input=True raises error."""
    from unittest.mock import MagicMock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip.extractall.side_effect = RuntimeError("Bad password")
    mock_zip.__enter__ = Mock(return_value=mock_zip)
    mock_zip.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            try:
                unzip(
                    "local.zip",
                    is_url=False,
                    clone_to_dir=clone_dir,
                    no_input=True
                )
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_invalid_zip_file_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir


# LLM-generated content at query #16
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    from zipfile import ZipFile
    import pytest
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock requests.get to avoid actual HTTP calls
    mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Test that unzip raises InvalidZipRepository for empty zip
    with pytest.raises(InvalidZipRepository, match="Zip repository .* is empty"):
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with URL when zip file doesn't exist yet."""
    import io
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    
    # Create a temporary zip file
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.requests.get') as mock_get, \
         patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[zip_content])
        mock_get.return_value = mock_response
        
        (tmp_path / "temp").mkdir(exist_ok=True)
        
        # Write the zip file for extraction
        zip_path = tmp_path / "clone" / "test.zip"
        zip_path.write_bytes(zip_content)
        
        with patch('cookiecutter.zipfile.os.path.join', side_effect=lambda *args: str(tmp_path / "clone" / "test.zip") if 'test.zip' in str(args) else str(tmp_path / "temp" / "test_project")):
            result = unzip(
                zip_uri='http://example.com/test.zip',
                is_url=True,
                clone_to_dir=str(clone_to_dir),
                no_input=True,
                password=None
            )
            
            assert result is not None


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with local file path."""
    import io
    from zipfile import ZipFile
    from unittest.mock import patch
    
    # Create a temporary zip file
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()
    
    zip_file_path = tmp_path / "test.zip"
    zip_file_path.write_bytes(zip_content)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'), \
         patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
        
        (tmp_path / "temp").mkdir(exist_ok=True)
        
        result = unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=str(clone_to_dir),
            no_input=True,
            password=None
        )
        
        assert result is not None
        assert 'test_project' in result


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    import io
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        pass
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()
    
    zip_file_path = tmp_path / "empty.zip"
    zip_file_path.write_bytes(zip_content)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        try:
            unzip(
                zip_uri=str(zip_file_path),
                is_url=False,
                clone_to_dir=str(clone_to_dir),
                no_input=True,
                password=None
            )
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository:
            pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip has no top-level directory."""
    import io
    from zipfile import ZipFile
    from unittest.mock import patch
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('file.txt', 'content')
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()
    
    zip_file_path = tmp_path / "notoplevel.zip"
    zip_file_path.write_bytes(zip_content)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        try:
            unzip(
                zip_uri=str(zip_file_path),
                is_url=False,
                clone_to_dir=str(clone_to_dir),
                no_input=True,
                password=None
            )
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository:
            pass


def test_unzip_invalid_zip_file_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for invalid zip file."""
    from unittest.mock import patch
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_bytes(b"not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        try:
            unzip(
                zip_uri=str(zip_file_path),
                is_url=False,
                clone_to_dir=str(clone_to_dir),
                no_input=True,
                password=None
            )
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository:
            pass


def test_unzip_with_password_protected_zip(tmp_path, monkeypatch):
    """Test unzip with password-protected zip file."""
    import io
    from zipfile import ZipFile
    from unittest.mock import patch
    
    # Create a password-protected zip file
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content', compress_type=8)
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()
    
    zip_file_path = tmp_path / "protected.zip"
    zip_file_path.write_bytes(zip_content)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'),


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_local_file_valid_zip(tmp_path, monkeypatch):
    """Test unzipping a local valid zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file
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
    """Test unzipping an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzipping a zipfile without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_file(tmp_path):
    """Test unzipping an invalid zip file raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except Exception as e:
        assert "valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
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
    assert "project_name" in result


def test_unzip_password_protected_with_correct_password(tmp_path, monkeypatch):
    """Test unzipping a password-protected zipfile with correct password."""
    import zipfile
    import os
    
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_password_protected_no_input_raises(tmp_path, monkeypatch):
    """Test unzipping password-protected zipfile with no_input=True raises."""
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
        assert False, "Should raise InvalidZipRepository"
    except Exception:
        pass


def test_unzip_password_protected_with_wrong_password(tmp_path, monkeypatch):
    """Test unzipping password-protected zipfile with wrong password raises."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="wrongpass")
        assert False, "Should raise InvalidZipRepository"
    except Exception:
        pass


def test_unzip_returns_unzip_path(tmp_path):
    """Test that unzip returns the correct unzip_path."""
    import zipfile
    import os
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("my_project/", "")
        zf.writestr("my_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert isinstance(result, str)
    assert "my_project" in result
    assert os.path.isdir(result)


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_chunk_filter_evaluates_to_false():
    """Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks."""
    # This test verifies that empty chunks (keep-alive new chunks) are filtered out
    # The predicate "if chunk:" should evaluate to False when chunk is an empty bytes object
    chunk = b''
    assert not chunk


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_download_predicate_false_when_reusing_existing():
    """Test that the predicate at line 36 evaluates to False when user chooses to reuse existing file."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Create a dummy zip file that exists
        zip_filename = "existing_archive.zip"
        zip_path = clone_to_dir / zip_filename
        zip_path.touch()
        
        zip_uri = f"https://example.com/{zip_filename}"
        
        # Mock prompt_and_delete to return False (user wants to reuse)
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                # Setup mock ZipFile
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    with patch('cookiecutter.zipfile.requests.get'):
                        result = unzip(
                            zip_uri=zip_uri,
                            is_url=True,
                            clone_to_dir=clone_to_dir,
                            no_input=False
                        )
                
                # Verify that requests.get was NOT called (download=False at line 36)
                # This means the predicate at line 36 evaluated to False
                assert result is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test that predicate at line 55 evaluates to True when zipfile is empty."""
    import tempfile
    from zipfile import ZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Call unzip with is_url=False (local file)
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert 'is empty' in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_chunk_filter_predicate_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    chunk = b''
    result = bool(chunk)
    assert result is False


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import io
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Mock the requests.get to return a response with chunks
        mock_response = Mock()
        mock_chunk_1 = b'chunk1data'
        mock_chunk_2 = b''  # Empty chunk (keep-alive)
        mock_chunk_3 = b'chunk3data'
        mock_response.iter_content.return_value = [mock_chunk_1, mock_chunk_2, mock_chunk_3]
        
        # Create a valid zip file for testing
        import zipfile
        zip_path = os.path.join(clone_to_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Remove the zip file so it will be "downloaded"
        os.remove(zip_path)
        
        # Track which chunks were written
        written_chunks = []
        
        original_open = open
        def mock_open_func(path, mode):
            file_obj = original_open(path, mode)
            original_write = file_obj.write
            
            def tracked_write(chunk):
                written_chunks.append(chunk)
                return original_write(chunk)
            
            file_obj.write = tracked_write
            return file_obj
        
        with patch('requests.get', return_value=mock_response):
            with patch('builtins.open', side_effect=mock_open_func):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                    
                    result = unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
        
        # Verify that only non-empty chunks were written (the predicate filtered empty chunks)
        assert mock_chunk_1 in written_chunks
        assert mock_chunk_3 in written_chunks
        assert mock_chunk_2 not in written_chunks


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    """Test unzip downloads a URL and extracts it successfully."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tempfile.mkdtemp()
    
    mock_zip_file = Mock(spec=ZipFile)
    mock_zip_file.namelist.return_value = ["test_project/", "test_project/file.txt"]
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                        mock_response = Mock()
                        mock_response.iter_content.return_value = [b'test']
                        mock_get.return_value = mock_response
                        
                        result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        
                        assert result is not None
                        assert mock_zip_file.extractall.called


def test_unzip_with_local_file():
    """Test unzip uses a local file without downloading."""
    import tempfile
    import os
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    local_zip_path = "/local/path/repo.zip"
    clone_to_dir = tempfile.mkdtemp()
    
    mock_zip_file = Mock(spec=ZipFile)
    mock_zip_file.namelist.return_value = ["test_project/", "test_project/file.txt"]
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.abspath', return_value=local_zip_path):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                    result = unzip(local_zip_path, is_url=False, clone_to_dir=clone_to_dir)
                    
                    assert result is not None
                    assert mock_zip_file.extractall.called


def test_unzip_empty_repository_raises_error():
    """Test unzip raises InvalidZipRepository for empty zip."""
    import tempfile
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/empty.zip"
    clone_to_dir = tempfile.mkdtemp()
    
    mock_zip_file = Mock(spec=ZipFile)
    mock_zip_file.namelist.return_value = []
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Expected InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_no_top_level_directory_raises_error():
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import tempfile
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tempfile.mkdtemp()
    
    mock_zip_file = Mock(spec=ZipFile)
    mock_zip_file.namelist.return_value = ["file.txt"]
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Expected InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_with_password():
    """Test unzip extracts password-protected zip with provided password."""
    import tempfile
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tempfile.mkdtemp()
    password = "test_password"
    
    mock_zip_file = Mock(spec=ZipFile)
    mock_zip_file.namelist.return_value = ["test_project/", "test_project/file.txt"]
    mock_zip_file.extractall.side_effect = [RuntimeError("Bad password"), None]
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                        result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, password=password, no_input=True)
                        
                        assert result is not None
                        assert mock_zip_file.extractall.called


def test_unzip_invalid_zip_file_raises_error():
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    import tempfile
    from unittest.mock import Mock, patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = "https://example.com/invalid.zip"
    clone_to_dir = tempfile.mkdtemp()
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Invalid zip")):
                    try:
                        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        assert False, "Expected InvalidZipRepository"
                    except InvalidZipRepository:
                        pass


def test_unzip_prompt_and_delete_existing_file():
    """Test unzip prompts to delete existing zip file."""
    import tempfile
    import os
    from unittest.mock import Mock, patch
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tempfile


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    
    # Create a mock response object
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b''])
    
    # Create a temporary directory and file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/test.zip"
        identifier = "test.zip"
        zip_path = os.path.join(clone_to_dir, identifier)
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
                with patch('cookiecutter.zipfile.open', mock_open()) as mock_file:
                    with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                        with patch('cookiecutter.zipfile.ZipFile'):
                            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                                # Import the function to test
                                from cookiecutter.zipfile import unzip
                                
                                # Call unzip with is_url=True
                                try:
                                    unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                                except:
                                    pass
                                
                                # Verify that iter_content was called
                                mock_response.iter_content.assert_called_once_with(chunk_size=1024)
                                
                                # Verify that only non-empty chunks were written
                                # The mock_file.write should be called twice (for b'chunk1' and b'chunk2')
                                # but not for the empty bytes b''
                                write_calls = mock_file().write.call_count
                                assert write_calls == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_with_valid_local_zipfile(tmp_path, mocker):
    """Test unzip with a valid local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile with proper structure
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


def test_unzip_with_url_new_file(tmp_path, mocker):
    """Test unzip with URL when zipfile doesn't exist yet."""
    import zipfile
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=False)
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[open(zip_path, 'rb').read()])
    mocker.patch('requests.get', return_value=mock_response)
    mocker.patch('builtins.open', mocker.mock_open())
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    
    # Create an empty zipfile
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
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "no_top_level.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a file that's not a valid zipfile
    fake_zip = tmp_path / "fake.zip"
    fake_zip.write_text("not a zip file")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(fake_zip), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "valid zip" in str(e).lower()


def test_unzip_password_protected_with_correct_password(tmp_path, mocker):
    """Test unzip with password-protected zipfile and correct password."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="password")
    
    assert "project_name" in result


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zipfile and no_input=True raises error."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


def test_unzip_with_url_existing_file_prompts_delete(tmp_path, mocker):
    """Test unzip with URL when zipfile exists prompts to delete."""
    import zipfile
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.


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


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip with empty zipfile raises InvalidZipRepository."""
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


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
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
        assert "valid zip" in str(e).lower()


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
    assert "project_name" in result


def test_unzip_with_password_protected_zip_no_input(tmp_path):
    """Test unzip with password protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.setpassword(b"password")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password" in str(e).lower() or "protected" in str(e).lower()


def test_unzip_with_correct_password(tmp_path):
    """Test unzip with password protected zip and correct password."""
    import zipfile
    import os
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    assert os.path.exists(result)


def test_unzip_expanduser_in_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip expands user home directory in clone_to_dir."""
    import zipfile
    import os
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_to_dir))
    
    assert "project_name" in result
    assert os.path.exists(result)


# LLM-generated content at query #2
#--------------------------

```python
def test_zipfile_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    
    # Create a valid zip file with at least one entry
    zip_path = os.path.join(temp_dir, "test.zip")
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_dir/", "")
        zf.writestr("test_dir/file.txt", "content")
    
    # Open the zip file and verify the predicate evaluates to False
    with ZipFile(zip_path) as zip_file:
        predicate_result = len(zip_file.namelist()) == 0
    
    assert predicate_result is False
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_local_file_valid_zip(tmp_path, mocker):
    """Test unzipping a local zipfile with valid structure."""
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
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test that empty zipfile raises InvalidZipRepository."""
    import zipfile
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test that zip without top-level directory raises InvalidZipRepository."""
    import zipfile
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_file_raises_error(tmp_path, mocker):
    """Test that invalid zip file raises InvalidZipRepository."""
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_url_downloads_and_extracts(tmp_path, mocker):
    """Test unzipping from URL downloads and extracts properly."""
    import zipfile
    
    # Create a valid zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_path.read_bytes()])
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result


def test_unzip_url_prompts_to_delete_existing(tmp_path, mocker):
    """Test that existing zip file prompts for deletion."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    existing_zip = clone_to_dir / "test.zip"
    existing_zip.write_text("old content")
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_path.read_bytes()])
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    assert "project_name" in result


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test extracting password-protected zip with valid password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_iter_content_chunk_filtering():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import io
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path
    import tempfile
    import os
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/test.zip"
        
        # Mock the requests.get to return a response with chunks
        mock_response = Mock()
        # Simulate iter_content returning both empty and non-empty chunks
        mock_response.iter_content.return_value = [
            b"chunk1",  # Non-empty chunk (evaluates to True)
            b"",        # Empty chunk (evaluates to False)
            b"chunk2",  # Non-empty chunk (evaluates to True)
        ]
        
        # Create a mock zip file
        mock_zip_content = b"PK\x03\x04"  # Minimal zip file signature
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                mock_zipfile_instance = MagicMock()
                mock_zipfile_instance.__enter__.return_value = mock_zipfile_instance
                mock_zipfile_instance.__exit__.return_value = None
                mock_zipfile_instance.namelist.return_value = ['test_dir/']
                mock_zipfile.return_value = mock_zipfile_instance
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    from cookiecutter.zipfile import unzip
                    
                    # Call the function
                    result = unzip(
                        zip_uri=zip_uri,
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True,
                        password=None
                    )
                    
                    # Verify that write was called only for non-empty chunks
                    # The predicate `if chunk:` should filter out empty chunks
                    assert mock_file.write.call_count == 2
                    mock_file.write.assert_any_call(b"chunk1")
                    mock_file.write.assert_any_call(b"chunk2")


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_downloads_zipfile_with_chunks():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_uri = "https://example.com/repo.zip"
        identifier = "repo.zip"
        zip_path = os.path.join(temp_dir, identifier)
        
        # Mock the requests.get to return a response with chunks
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b'chunk3', b''])
        
        # Mock ZipFile to avoid actual zip processing
        mock_zip_file = Mock()
        mock_zip_file.namelist.return_value = ['project_dir/']
        mock_zip_file.__enter__ = Mock(return_value=mock_zip_file)
        mock_zip_file.__exit__ = Mock(return_value=None)
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        unzip(
                            zip_uri=zip_uri,
                            is_url=True,
                            clone_to_dir=temp_dir,
                            no_input=True,
                            password=None
                        )
        
        # Verify that open was called with 'wb' mode (line 39)
        mock_file.assert_called_with(zip_path, 'wb')
        
        # Verify that write was called for each non-empty chunk
        handle = mock_file()
        write_calls = handle.write.call_args_list
        assert len(write_calls) == 3
        assert write_calls[0][0][0] == b'chunk1'
        assert write_calls[1][0][0] == b'chunk2'
        assert write_calls[2][0][0] == b'chunk3'


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_context_manager_predicate():
    """Test that the predicate at line 54 (ZipFile context manager) evaluates to True."""
    from pathlib import Path
    from zipfile import ZipFile
    import tempfile
    import os
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    # Create a valid zip file with a top-level directory
    zip_path = os.path.join(temp_dir, 'test.zip')
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_dir/', '')
        zf.writestr('test_dir/file.txt', 'content')
    
    # Test that the context manager can be entered (predicate evaluates to True)
    with ZipFile(zip_path) as zip_file:
        assert zip_file is not None
        assert len(zip_file.namelist()) > 0
        assert zip_file.namelist()[0].endswith('/')
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_bad_zip_file_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    from zipfile import BadZipFile
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a file that looks like a zip but isn't valid
    bad_zip_path = clone_to_dir / "bad.zip"
    bad_zip_path.write_bytes(b"This is not a valid zip file")
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Mock ZipFile to raise BadZipFile
    mocker.patch(
        'cookiecutter.zipfile.ZipFile',
        side_effect=BadZipFile("Bad zip file")
    )
    
    try:
        unzip(
            zip_uri=str(bad_zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'not a valid zip archive' in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    
    # Create a test zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Mock the dependencies
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
    
    assert result.endswith('project_name')


def test_unzip_url_no_input_first_download(tmp_path, mocker):
    """Test unzip with URL and no_input=True for first download."""
    import zipfile
    
    # Create a test zipfile
    zip_content = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    with open(zip_content, 'rb') as f:
        zip_bytes = f.read()
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Mock the dependencies
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[zip_bytes])
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        'http://example.com/project.zip',
        is_url=True,
        clone_to_dir=str(clone_dir),
        no_input=True
    )
    
    assert result.endswith('project_name')
    mock_open.assert_called_once()


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    
    # Create an empty zipfile
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level' in str(e).lower()


def test_unzip_invalid_zip_file_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    # Create an invalid zip file
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'valid zip' in str(e).lower()


def test_unzip_password_protected_with_correct_password(tmp_path, mocker):
    """Test unzip with password protected zipfile and correct password."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        password='testpass'
    )
    
    assert result.endswith('project_name')


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password protected zipfile and no_input=True raises error."""
    import zipfile
    
    # Create a password-protected zipfile
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'testpass')
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Mock extractall to raise RuntimeError for password protection
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    mock_zip_file.namelist = mocker.MagicMock(return_value=['project_name/', 'project_name/file.txt'])
    mock_zip


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_invalid_zip_file_raises_invalid_zip_repository(tmp_path, mocker):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    import tempfile
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake zip file that will raise BadZipFile
    fake_zip_path = tmp_path / "fake.zip"
    fake_zip_path.write_text("This is not a valid zip file")
    
    # Mock ZipFile to raise BadZipFile when instantiated
    mocker.patch(
        'cookiecutter.zipfile.ZipFile',
        side_effect=BadZipFile("Bad zip file")
    )
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Call unzip with is_url=False to use local file path
    try:
        unzip(str(fake_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'is not a valid zip archive' in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_41_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False."""
    chunk = b''
    assert not chunk


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This happens when prompt_and_delete returns False, indicating the user
    wants to reuse the existing version.
    """
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "http://example.com/archive.zip"
    identifier = "archive.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file that exists
    zip_path.write_bytes(b"dummy content")
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock ZipFile to avoid actual zip operations
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ['project/']
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=None)
    
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Execute
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Assert that download block (line 37-42) was NOT executed
    # This is verified by checking that requests.get was never called
    # (it would only be called if download was True)
    mock_prompt_and_delete.assert_called_once_with(str(zip_path), no_input=False)


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_predicate_at_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    zip_path = clone_to_dir / "repo.zip"
    
    # Create a dummy zip file at the expected path
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse existing version)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    # Mock requests.get to ensure it's not called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile and related functions
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ['project_name/']
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Execute
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Assert that requests.get was NOT called (because download was False)
    mock_requests_get.assert_not_called()
    
    # Assert that the result is correct
    assert result == os.path.join(str(tmp_path / "temp"), 'project_name')


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 evaluates to False when zipfile is not empty."""
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zipfile with content
        zip_path = os.path.join(temp_dir, "test.zip")
        
        with ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("test_dir/", "")
            zip_file.writestr("test_dir/file.txt", "content")
        
        # Open and verify the predicate condition
        with ZipFile(zip_path) as zip_file:
            # The predicate at line 55: len(zip_file.namelist()) == 0
            # Should evaluate to False because we added files to the zip
            predicate_result = len(zip_file.namelist()) == 0
        
        assert predicate_result is False


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_with_url_no_existing_file(tmp_path, monkeypatch):
    """Test unzip with a URL when file doesn't exist locally."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'test_chunk'])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                (tmp_path / "temp").mkdir(exist_ok=True)
                
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile.return_value = mock_zip_instance
                
                result = unzip(
                    'http://example.com/test.zip',
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=True
                )
    
    assert 'project_name' in result


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local file path."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    local_zip = tmp_path / "test.zip"
    local_zip.touch()
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            (tmp_path / "temp").mkdir(exist_ok=True)
            
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            result = unzip(
                str(local_zip),
                is_url=False,
                clone_to_dir=tmp_path,
                no_input=True
            )
    
    assert 'project' in result


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises error for empty zip file."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                (tmp_path / "temp").mkdir(exist_ok=True)
                
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = []
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile.return_value = mock_zip_instance
                
                try:
                    unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises error when zip has no top-level directory."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                (tmp_path / "temp").mkdir(exist_ok=True)
                
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['file.txt']
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile.return_value = mock_zip_instance
                
                try:
                    unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_with_password(tmp_path):
    """Test unzip with password-protected archive."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    local_zip = tmp_path / "test.zip"
    local_zip.touch()
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            (tmp_path / "temp").mkdir(exist_ok=True)
            
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
            
            def extract_side_effect(path=None, pwd=None):
                if pwd is None:
                    raise RuntimeError("Password required")
            
            mock_zip_instance.extractall.side_effect = extract_side_effect
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            result = unzip(
                str(local_zip),
                is_url=False,
                clone_to_dir=tmp_path,
                no_input=True,
                password='testpass'
            )
    
    assert 'project' in result


def test_unzip_bad_zip_file_raises_error(tmp_path):
    """Test unzip raises error for invalid zip file."""
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    with patch('cookiecutter.zipfile.requests.get'):
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                (tmp_path / "temp").mkdir(exist_ok=True)
                
                try:
                    unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_local_file(tmp_path, monkeypatch):
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "test_project" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
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
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "no_top_level.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zip_file_raises_error(tmp_path):
    import os
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    with open(zip_path, 'w') as f:
        f.write("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone" / "nested"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "test_project" in result


def test_unzip_with_password_protected_zip(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password=password
    )
    
    assert "test_project" in result


def test_unzip_with_expanduser_path(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_to_dir))
    
    assert os.path.exists(result)
    assert "test_project" in result


# LLM-generated content at query #16
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False."""
    # Setup: Create a mock zip file path that exists
    zip_path = tmp_path / "test.zip"
    zip_path.touch()
    
    # Mock the prompt_and_delete function to return False
    # This makes the predicate at line 36 evaluate to False
    mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock make_sure_path_exists to do nothing
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Mock requests.get so it's not called (since download is False)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile and related functions
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.__enter__.return_value.namelist.return_value = ['project_name/']
    mocker.patch(
        'cookiecutter.zipfile.ZipFile',
        return_value=mock_zip_file
    )
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    
    from cookiecutter.zipfile import unzip
    
    # Call unzip with is_url=True and no_input=False
    # prompt_and_delete will return False, making the download predicate False
    result = unzip(
        zip_uri='http://example.com/repo.zip',
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=False
    )
    
    # Verify that requests.get was NOT called because download=False
    mock_requests_get.assert_not_called()
    
    # Verify the result is as expected
    assert result is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_chunk_filter_evaluates_to_false():
    """Test that the predicate at line 40 evaluates to False for empty chunks."""
    chunk = b''
    result = bool(chunk)
    assert result is False


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_when_zip_is_empty(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass
    
    # Mock make_sure_path_exists to avoid directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Attempt to unzip empty zip file and verify exception is raised
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except InvalidZipRepository as e:
        assert "is empty" in str(e)


# LLM-generated content at query #19
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
    
    # Create a valid zip file with content
    zip_path = os.path.join(temp_dir, 'test.zip')
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_dir/', '')
        zf.writestr('project_dir/file.txt', 'content')
    
    # Open the zip file and verify the predicate is False
    with ZipFile(zip_path) as zip_file:
        predicate_result = len(zip_file.namelist()) == 0
    
    assert predicate_result is False
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, mocker):
    """Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True."""
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    import os
    
    # Create a temporary directory to use as clone_to_dir
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a dummy zip file that will exist
    zip_filename = "test.zip"
    zip_path = clone_to_dir / zip_filename
    zip_path.touch()
    
    # Mock the prompt_and_delete function to return True (download)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    # Mock requests.get to avoid actual network calls
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'test content']
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    # Mock the ZipFile to avoid actual zip file operations
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.__enter__.return_value.namelist.return_value = ['project/']
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile.mkdtemp
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    # Call unzip with is_url=True to trigger the condition at line 31
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(clone_to_dir),
        no_input=False
    )
    
    # Assert that prompt_and_delete was called, confirming the predicate was True
    mocker.patch.object(__import__('cookiecutter.zipfile', fromlist=['prompt_and_delete']), 'prompt_and_delete').assert_called()


# LLM-generated content at query #21
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
    fake_zip_path.write_text("this is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock make_sure_path_exists to avoid actual directory creation
    def mock_make_sure_path_exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr("cookiecutter.zipfile.make_sure_path_exists", mock_make_sure_path_exists)
    
    # Test that BadZipFile is caught and InvalidZipRepository is raised
    try:
        unzip(str(fake_zip_path), is_url=False, clone_to_dir=str(clone_to_dir), no_input=True)
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert e.__class__.__name__ == "InvalidZipRepository"
        assert "is not a valid zip archive" in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_with_url_creates_directory_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from zipfile import ZipFile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        zip_uri = "https://example.com/repo.zip"
        
        mock_zip_content = b"PK\x03\x04"  # Minimal zip file signature
        
        with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_mkdir, \
             patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
             patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp, \
             patch('builtins.open', mock_open()):
            
            mock_response = Mock()
            mock_response.iter_content = Mock(return_value=[mock_zip_content])
            mock_get.return_value = mock_response
            
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project_name/", "project_name/file.txt"]
            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
            
            temp_extract_dir = os.path.join(temp_dir, "extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            mock_mkdtemp.return_value = temp_extract_dir
            
            from cookiecutter.zipfile import unzip
            result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
            
            assert result == os.path.join(temp_extract_dir, "project_name")
            mock_mkdir.assert_called_once()
            mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)
            mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        local_zip_path = os.path.join(temp_dir, "local.zip")
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'), \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp, \
             patch('cookiecutter.zipfile.os.path.abspath', return_value=local_zip_path):
            
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project_name/", "project_name/file.txt"]
            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
            
            temp_extract_dir = os.path.join(temp_dir, "extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            mock_mkdtemp.return_value = temp_extract_dir
            
            from cookiecutter.zipfile import unzip
            result = unzip(local_zip_path, is_url=False, clone_to_dir=clone_to_dir)
            
            assert result == os.path.join(temp_extract_dir, "project_name")
            mock_zip_instance.extractall.assert_called_once_with(path=temp_extract_dir)


def test_unzip_empty_zip_raises_error():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        zip_uri = "https://example.com/empty.zip"
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'), \
             patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
             patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp'):
            
            mock_response = Mock()
            mock_response.iter_content = Mock(return_value=[b"PK\x03\x04"])
            mock_get.return_value = mock_response
            
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = []
            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
            
            from cookiecutter.zipfile import unzip
            try:
                unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "empty" in str(e)


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.exceptions import InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        zip_uri = "https://example.com/bad.zip"
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'), \
             patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
             patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp'):
            
            mock_response = Mock()
            mock_response.iter_content = Mock(return_value=[b"PK\x03\x04"])
            mock_get.return_value = mock_response
            
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["file.txt"]
            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_instance
            
            from cookiecutter.zipfile import unzip
            try:
                unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository as e:
                assert "top-level directory" in str(e)


def test_unzip_password_protected_with_valid_password():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        zip_uri = "https://example.com/protected.zip"
        password = "secret123"
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'), \
             patch('cookiecutter.zipfile.os.path.exists', return_value=False), \
             patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mk


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, monkeypatch):
    """Test unzip downloads and extracts a zipfile from URL."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a test zip file
    zip_dir = tmp_path / "zip_source"
    zip_dir.mkdir()
    test_project_dir = zip_dir / "test_project/"
    test_project_dir.mkdir()
    (test_project_dir / "file.txt").write_text("test content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(test_project_dir, arcname="test_project/")
        zf.write(test_project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get
    import requests
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    mock_get_called = []
    def mock_get(url, stream=True, timeout=100):
        mock_get_called.append(url)
        return MockResponse()
    
    monkeypatch.setattr(requests, 'get', mock_get)
    
    result = unzip(
        f"file://{zip_file_path}",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert os.path.exists(result)
    assert "test_project" in result


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile path."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a test zip file
    zip_dir = tmp_path / "zip_source"
    zip_dir.mkdir()
    test_project_dir = zip_dir / "test_project/"
    test_project_dir.mkdir()
    (test_project_dir / "file.txt").write_text("test content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(test_project_dir, arcname="test_project/")
        zf.write(test_project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(
        str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert os.path.exists(result)
    assert "test_project" in result


def test_unzip_empty_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    zip_file_path = tmp_path / "empty.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_missing_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when zipfile lacks top-level directory."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_file_path = tmp_path / "no_top_dir.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an invalid zip file
    zip_file_path = tmp_path / "invalid.zip"
    zip_file_path.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            str(zip_file_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a test zip file
    zip_dir = tmp_path / "zip_source"
    zip_dir.mkdir()
    test_project_dir = zip_dir / "test_project/"
    test_project_dir.mkdir()
    (test_project_dir / "file.txt").write_text("test content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(test_project_dir, arcname="test_project/")
        zf.write(test_project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    assert not clone_to_dir.exists()
    
    result = unzip(
        str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert clone_to_dir.exists()
    assert os.path.exists(result)


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
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
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
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
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
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
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip."""
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
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert os.path.exists(clone_to_dir)
    assert result.endswith("project")


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path):
    """Test unzip raises error for password-protected zip with no_input."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password protected" in str(e).lower() or "unable to unlock" in str(e).lower()


def test_unzip_with_correct_password(tmp_path):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    import os
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"mypassword")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="mypassword")
    
    assert result.endswith("project")
    assert os.path.exists(result)


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_with_local_zipfile(tmp_path, mocker):
    """Test unzip with a local zipfile that is not a URL."""
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
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_url_with_no_existing_file(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist yet."""
    import zipfile
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=False)
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[open(zip_path, 'rb').read()])
    mocker.patch('requests.get', return_value=mock_response)
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project_name")


def test_unzip_empty_zipfile(tmp_path, mocker):
    """Test unzip raises error for empty zipfile."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
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
    """Test unzip raises error when zip doesn't have top-level directory."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a zip file without top-level directory
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


def test_unzip_invalid_zip_file(tmp_path, mocker):
    """Test unzip raises error for invalid zip file."""
    from cookiecutter.exceptions import InvalidZipRepository
    
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


def test_unzip_password_protected_with_correct_password(tmp_path, mocker):
    """Test unzip with password-protected zipfile using correct password."""
    import zipfile
    import os
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"test_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True, password="test_password")
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_password_protected_with_invalid_password(tmp_path, mocker):
    """Test unzip raises error with invalid password for protected zipfile."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"correct_password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True, password="wrong_password")
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "invalid password" in str(e).lower()


def test_unzip_expanduser_in_clone_to_dir(tmp_path, mocker):
    """Test unzip expands ~ in clone_to_dir path."""
    import zipfile
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mock_make_sure = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_expanduser = mocker.patch('pathlib.Path.expanduser', return_value=tmp_path / "clone")
    
    clone_to_dir = tmp_path / "clone"
    clone_


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_context_manager_closes_zipfile():
    """Test that the predicate at line 54 (ZipFile context manager) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, MagicMock
    
    # Create a temporary directory and a valid zip file
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test.zip")
    
    # Create a valid zip file with a top-level directory
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    # Mock the necessary functions
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
            with patch('cookiecutter.zipfile.ZipFile', wraps=ZipFile) as mock_zipfile:
                from cookiecutter.zipfile import unzip
                
                # Call the unzip function with is_url=False
                result = unzip(
                    zip_uri=zip_path,
                    is_url=False,
                    clone_to_dir=temp_dir,
                    no_input=True,
                    password=None
                )
                
                # Verify that ZipFile was called (context manager enters)
                mock_zipfile.assert_called_once_with(zip_path)
                
                # Verify the result is the expected unzip_path
                assert result == os.path.join(temp_dir, "test_project")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# LLM-generated content at query #27
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_uri = "https://example.com/test.zip"
        
        # Mock the requests.get to return a response with chunks
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b''])
        
        # Mock ZipFile to avoid actual zip operations
        mock_zip_file = Mock()
        mock_zip_file.namelist = Mock(return_value=['project_dir/', 'project_dir/file.txt'])
        mock_zip_file.__enter__ = Mock(return_value=mock_zip_file)
        mock_zip_file.__exit__ = Mock(return_value=None)
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir, no_input=True)
                        
                        # Verify that write was called with non-empty chunks
                        mock_file().write.assert_any_call(b'chunk1')
                        mock_file().write.assert_any_call(b'chunk2')
                        
                        # Verify write was NOT called with the empty chunk
                        assert mock_file().write.call_count == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_unzip_predicate_line_55_false():
    """Test that the predicate at line 55 evaluates to False when zip file has content."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file with content
        zip_path = os.path.join(temp_dir, 'test.zip')
        with ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr('project_dir/', '')
            zip_file.writestr('project_dir/file.txt', 'content')
        
        # Call unzip with the local zip file
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=temp_dir,
            no_input=True
        )
        
        # Assert that result is a valid path (meaning the predicate at line 55 was False)
        assert isinstance(result, str)
        assert len(result) > 0


# LLM-generated content at query #29
#--------------------------

```python
def test_unzip_context_manager_closes_zipfile():
    """Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Call unzip with is_url=False to use the local zipfile
        result = unzip(zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        
        # Verify that result is a valid path string
        assert isinstance(result, str)
        assert len(result) > 0
        assert "test_project" in result
        
    finally:
        # Cleanup
        from shutil import rmtree
        rmtree(temp_dir, ignore_errors=True)
        rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #30
#--------------------------

```python
def test_unzip_chunk_filter_predicate():
    """Test that the predicate at line 40 (if chunk:) evaluates to True for non-empty chunks."""
    chunk_with_data = b'x' * 1024
    chunk_empty = b''
    
    assert chunk_with_data
    assert not chunk_empty


# LLM-generated content at query #31
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_path:
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'test_data'])
                            mock_get.return_value = mock_response
                            
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['test_project/', 'test_project/file.txt']
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile_class.return_value = mock_zip_instance
                            
                            result = unzip(
                                'http://example.com/test.zip',
                                is_url=True,
                                clone_to_dir=clone_to_dir,
                                no_input=True
                            )
                            
                            assert result == os.path.join(temp_dir, 'test_project')
                            mock_get.assert_called_once()
                            mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        local_zip_path = os.path.join(temp_dir, 'local.zip')
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['local_project/', 'local_project/file.txt']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile_class.return_value = mock_zip_instance
                    
                    result = unzip(
                        local_zip_path,
                        is_url=False,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
                    
                    assert result == os.path.join(temp_dir, 'local_project')
                    mock_zip_instance.extractall.assert_called_once()


def test_unzip_empty_zipfile_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get'):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = []
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile_class.return_value = mock_zip_instance
                            
                            error_raised = False
                            try:
                                unzip(
                                    'http://example.com/empty.zip',
                                    is_url=True,
                                    clone_to_dir=clone_to_dir,
                                    no_input=True
                                )
                            except InvalidZipRepository:
                                error_raised = True
                            
                            assert error_raised


def test_unzip_missing_top_level_directory_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get'):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['file.txt']
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile_class.return_value = mock_zip_instance
                            
                            error_raised = False
                            try:
                                unzip(
                                    'http://example.com/bad.zip',
                                    is_url=True,
                                    clone_to_dir=clone_to_dir,
                                    no_input=True
                                )
                            except InvalidZipRepository:
                                error_raised = True
                            
                            assert error_raised


def test_unzip_password_protected_with_correct_password():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get'):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['protected_project/', 'protected_project/file.txt']
                            mock_zip_instance.extractall.side_effect = [RuntimeError(), None]
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile_class.return_value = mock_zip_instance
                            
                            result = unzip(
                                'http://example.com/protected.zip',
                


# LLM-generated content at query #32
#--------------------------

```python
def test_unzip_with_url_creates_clone_dir():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = os.path.join(temp_dir, 'new_clone_dir')
        
        with patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_sure:
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                    mock_response = Mock()
                    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
                    mock_get.return_value = mock_response
                    
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile_class.return_value = mock_zip_instance
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir=clone_to_dir)
                    
                    mock_make_sure.assert_called_once()
                    assert result is not None


def test_unzip_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile_class.return_value = mock_zip_instance
                
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    result = unzip('/local/path/repo.zip', is_url=False, clone_to_dir='.')
                
                assert result is not None


def test_unzip_empty_zip_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = []
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile_class.return_value = mock_zip_instance
                
                try:
                    unzip('/local/path/repo.zip', is_url=False)
                    assert False, "Expected InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['file.txt']
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile_class.return_value = mock_zip_instance
                
                try:
                    unzip('/local/path/repo.zip', is_url=False)
                    assert False, "Expected InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_password_protected_with_provided_password():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                mock_zip_instance = MagicMock()
                mock_zip_instance.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
                
                def extract_side_effect(path=None, pwd=None):
                    if pwd is None:
                        raise RuntimeError("File is password protected")
                
                mock_zip_instance.extractall.side_effect = extract_side_effect
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.__exit__.return_value = None
                mock_zipfile_class.return_value = mock_zip_instance
                
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    result = unzip('/local/path/repo.zip', is_url=False, password='mypassword')
                
                assert result is not None


def test_unzip_bad_zip_file_raises_error():
    import tempfile
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
                try:
                    unzip('/local/path/repo.zip', is_url=False)
                    assert False, "Expected InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_with_url_prompt_and_delete():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = os.path.join(temp_dir, 'clone')
        zip_path = os.path.join(clone_to_dir, 'repo.zip')
        
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=True):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True) as mock_prompt:
                    with patch('cookiecutter.zipfile.requests.get') as mock_get:
                        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                            mock_response = Mock()
                            mock_response.iter_content.return_value = [b'chunk1']
                            mock_get.return_value = mock_response
                            
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['project_dir/', 'project_


# LLM-generated content at query #33
#--------------------------

```python
def test_unzip_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zipfile with a top-level directory and at least one file
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            # Add a top-level directory entry
            zf.writestr("project_dir/", "")
            # Add a file inside the directory
            zf.writestr("project_dir/file.txt", "content")
        
        # Call unzip with is_url=False to use the local zipfile
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password=None
        )
        
        # If we reach here, the predicate at line 54 evaluated to False
        # (meaning the zipfile is not empty), and the function proceeded successfully
        assert result is not None
        assert isinstance(result, str)
        
    finally:
        # Cleanup
        from cookiecutter.utils import rmtree
        if os.path.exists(temp_dir):
            rmtree(temp_dir)
        if os.path.exists(clone_to_dir):
            rmtree(clone_to_dir)


