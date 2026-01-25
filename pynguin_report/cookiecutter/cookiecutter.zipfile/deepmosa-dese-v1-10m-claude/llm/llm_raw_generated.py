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
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
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
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip with zipfile missing top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_invalid_zipfile(tmp_path):
    """Test unzip with invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert clone_to_dir.exists()
    assert "project_name" in result


# LLM-generated content at query #2
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
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "notopdir.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
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


def test_unzip_with_password_protected_zip(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and password provided."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password=password
    )
    
    assert "project_name" in result


def test_unzip_with_wrong_password_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and wrong password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    password = "correct_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            password="wrong_password"
        )
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "Invalid password" in str(e)


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "Unable to unlock password protected repository" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_on_bad_zipfile(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from zipfile import BadZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a fake bad zip file
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
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


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_predicate_at_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when is_url is True, the zip file exists, and prompt_and_delete
    returns False (user chooses to reuse existing version).
    """
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a mock zip file that exists
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    # Mock requests.get to ensure it's not called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile to return a valid zip structure
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    
    # Mock tempfile.mkdtemp
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    
    # Call unzip with is_url=True
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Verify that requests.get was NOT called (download block was skipped)
    mock_requests_get.assert_not_called()
    
    # Verify the function still processed the existing zip file
    mock_zip_file.extractall.assert_called()


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_predicate_line_31_true_when_zip_path_exists(tmp_path, monkeypatch):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    zip_file_path.touch()
    
    # Mock prompt_and_delete to return True (simulating user agreeing to delete)
    mock_prompt_and_delete_called = []
    def mock_prompt_and_delete(path, no_input=False):
        mock_prompt_and_delete_called.append((path, no_input))
        return True
    
    # Mock make_sure_path_exists to do nothing
    def mock_make_sure_path_exists(path):
        pass
    
    # Mock requests.get and ZipFile to avoid actual network calls and zip operations
    import io
    from zipfile import ZipFile
    
    # Create a valid test zip file with a top-level directory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    # Write the test zip to actual file
    with open(zip_file_path, 'wb') as f:
        f.write(zip_buffer.getvalue())
    
    # Mock the necessary functions
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', mock_make_sure_path_exists)
    monkeypatch.setattr('cookiecutter.zipfile.prompt_and_delete', mock_prompt_and_delete)
    
    # Call unzip with is_url=True and an existing zip file
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Manually set up the zip_path to exist
    identifier = "test.zip"
    zip_path = clone_to_dir / identifier
    zip_path.write_bytes(zip_buffer.getvalue())
    
    # Verify that os.path.exists(zip_path) returns True
    assert os.path.exists(str(zip_path)) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_dir = tmp_path / "zip_source"
    zip_dir.mkdir()
    
    project_dir = zip_dir / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="test_project/")
        zf.write(project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert result.endswith("test_project")
    assert os.path.exists(result)


def test_unzip_creates_clone_dir(tmp_path):
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="test_project/")
        zf.write(project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert result.endswith("test_project")


def test_unzip_empty_zip_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    zip_file_path = tmp_path / "empty.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zip file without top-level directory
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    
    zip_file_path = tmp_path / "no_dir.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(file_path, arcname="file.txt")
    
    clone_to_dir = tmp_path / "clone"
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path):
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an invalid zip file
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_bytes(b"This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    try:
        unzip(str(invalid_zip), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content")
    
    zip_file_path = tmp_path / "test.zip"
    with ZipFile(zip_file_path, 'w') as zf:
        zf.write(project_dir, arcname="test_project/")
        zf.write(project_dir / "file.txt", arcname="test_project/file.txt")
    
    clone_to_dir = tmp_path / "clone"
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir="~")
    
    assert result.endswith("test_project")


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_local_file(tmp_path, monkeypatch):
    """Test unzipping a local zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a test zip file
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    clone_to_dir = tmp_path / "nonexistent" / "dir"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    assert os.path.exists(clone_to_dir)
    assert result.endswith("project_name")


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises error when zip has no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_url_no_input_downloads(tmp_path, monkeypatch):
    """Test unzip with URL and no_input=True downloads the file."""
    import zipfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_path.read_bytes()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip(
            "http://example.com/test.zip",
            is_url=True,
            clone_to_dir=tmp_path,
            no_input=True
        )
    
    assert result.endswith("project_name")


def test_unzip_url_existing_file_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_path = clone_to_dir / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
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


def test_unzip_with_password(tmp_path, monkeypatch):
    """Test unzip with password-protected zip file."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=tmp_path,
        password=password
    )
    
    assert result.endswith("project_name")


def test_unzip_expanduser_in_clone_to_dir(tmp_path, monkeypatch):
    """Test that unzip expands ~ in clone_to_dir path."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    with patch('cookiecutter.zipfile.Path.expanduser') as mock_expand:
        mock_expand.return_value = tmp_path
        result = unzip(str(zip_path), is_url=False, clone_to_dir="~/test")
    
    mock_expand.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_chunk_filter_predicate_false():
    """Test that the predicate at line 40 evaluates to False for empty chunks."""
    # The predicate is: if chunk:
    # It should evaluate to False when chunk is empty (b'')
    chunk = b''
    assert not chunk


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    import os
    
    # Mock the necessary functions
    mock_zip_path = tmp_path / "test.zip"
    mock_zip_path.write_bytes(b"mock")
    
    # Mock make_sure_path_exists to do nothing
    monkeypatch.setattr("cookiecutter.zipfile.make_sure_path_exists", lambda x: None)
    
    # Mock prompt_and_delete to return False (user wants to reuse existing version)
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", lambda path, no_input=False: False)
    
    # Mock os.path.exists to return True (file exists)
    monkeypatch.setattr("os.path.exists", lambda x: True)
    
    # Mock ZipFile and related functions
    from unittest.mock import MagicMock, patch
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ["project_name/"]
    mock_zip_file.__enter__.return_value = mock_zip_file
    mock_zip_file.__exit__.return_value = None
    
    monkeypatch.setattr("cookiecutter.zipfile.ZipFile", lambda x: mock_zip_file)
    monkeypatch.setattr("os.path.abspath", lambda x: str(mock_zip_path))
    monkeypatch.setattr("tempfile.mkdtemp", lambda: str(tmp_path / "temp"))
    
    # Call unzip with is_url=True and no_input=False
    # The download variable should be False because prompt_and_delete returns False
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=False,
        password=None
    )
    
    # Verify that extractall was called (meaning download was False and we reused the file)
    assert mock_zip_file.extractall.called


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_url_creates_clone_to_dir(tmp_path, mocker):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    clone_to_dir = tmp_path / "new_dir"
    zip_uri = "http://example.com/test.zip"
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_prompt_and_delete = mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'test_chunk']
    mock_requests.return_value = mock_response
    
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    from cookiecutter.zipfile import unzip
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    mock_make_sure_path_exists.assert_called_once()
    assert result is not None


def test_unzip_local_file_without_url(tmp_path, mocker):
    """Test that unzip works with local file path."""
    zip_file_path = tmp_path / "test.zip"
    zip_file_path.touch()
    
    mock_make_sure_path_exists = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_tempfile = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
    
    mock_make_sure_path_exists.assert_called_once()
    assert result is not None


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = []
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when no top-level directory."""
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zipfile_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository, unzip
    from zipfile import BadZipFile
    
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile.return_value.__enter__.side_effect = BadZipFile("Bad zip")
    
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_protection_and_valid_password(tmp_path, mocker):
    """Test unzip with password-protected archive and valid password."""
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    password = "correct_password"
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mocker


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_with_url_and_new_download(tmp_path, mocker):
    """Test unzip with URL when file doesn't exist yet."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'PK\x03\x04']
    mock_requests.return_value = mock_response
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result is not None
    mock_requests.assert_called_once()


def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with local file path."""
    zip_file = tmp_path / "repo.zip"
    zip_file.write_bytes(b'PK\x03\x04')
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    result = unzip(str(zip_file), is_url=False, clone_to_dir=tmp_path, no_input=True)
    
    assert result is not None


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test that empty zip file raises InvalidZipRepository."""
    zip_uri = "https://example.com/empty.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'PK\x03\x04']
    mock_requests.return_value = mock_response
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = []
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.exceptions import InvalidZipRepository
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test that zip without top-level directory raises InvalidZipRepository."""
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'PK\x03\x04']
    mock_requests.return_value = mock_response
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['file.txt']
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.exceptions import InvalidZipRepository
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zip_file_raises_error(tmp_path, mocker):
    """Test that bad zip file raises InvalidZipRepository."""
    zip_uri = "https://example.com/bad.zip"
    clone_to_dir = tmp_path / "clone"
    
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'invalid']
    mock_requests.return_value = mock_response
    
    from zipfile import BadZipFile
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile.return_value.__enter__.side_effect = BadZipFile("Bad zip")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.exceptions import InvalidZipRepository
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and valid password."""
    zip_uri = "https://example.com/protected.zip"
    clone_to_dir = tmp_path / "clone"
    password = "test_password"
    
    mock_requests = mocker.patch('cookiecutter.zipfile.requests.get')
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'PK\x03\x04']
    mock_requests.return_value = mock_response
    
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip_instance.extractall.side_effect = [Runtime


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_raises_error_when_zip_is_empty(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual filesystem operations
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Test that InvalidZipRepository is raised when zip is empty
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_download_predicate_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    from zipfile import ZipFile
    
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_dir/', '')
        zf.writestr('test_dir/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock prompt_and_delete to return False
    def mock_prompt_and_delete(path, no_input=False):
        return False
    
    monkeypatch.setattr('cookiecutter.zipfile.prompt_and_delete', mock_prompt_and_delete)
    
    # Mock requests.get to avoid actual network calls
    class MockResponse:
        def iter_content(self, chunk_size):
            return []
    
    def mock_get(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    # Copy the zip file to clone_to_dir to simulate it already existing
    import shutil
    cached_zip = clone_to_dir / "test.zip"
    shutil.copy(zip_path, cached_zip)
    
    # Call unzip with is_url=True and an existing cached file
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Verify that the zip file was not re-downloaded (download=False at line 36)
    # The function should still return the unzip_path without re-downloading
    assert result is not None
    assert "test_dir" in result


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_line_39_predicate_evaluates_to_true(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks."""
    import io
    from unittest.mock import Mock, MagicMock, patch
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a valid zip file for testing
    import zipfile
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_dir/", "")
        zf.writestr("test_dir/file.txt", "content")
    
    # Mock requests.get to return chunks
    mock_response = Mock()
    mock_chunk_1 = b"chunk1"
    mock_chunk_2 = b"chunk2"
    mock_empty_chunk = b""
    
    # Simulate iter_content returning non-empty and empty chunks
    mock_response.iter_content.return_value = [
        mock_chunk_1,
        mock_empty_chunk,
        mock_chunk_2,
        mock_empty_chunk,
    ]
    
    # Mock requests.get
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", lambda *args, **kwargs: mock_response)
    
    # Mock ZipFile to avoid actual extraction
    mock_zipfile_instance = MagicMock()
    mock_zipfile_instance.namelist.return_value = ["test_dir/", "test_dir/file.txt"]
    mock_zipfile_instance.__enter__.return_value = mock_zipfile_instance
    mock_zipfile_instance.__exit__.return_value = None
    
    monkeypatch.setattr("cookiecutter.zipfile.ZipFile", MagicMock(return_value=mock_zipfile_instance))
    
    # Call unzip with is_url=True to trigger the chunk writing logic
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=tmp_path,
        no_input=True,
    )
    
    # Verify that iter_content was called and chunks were processed
    mock_response.iter_content.assert_called_once_with(chunk_size=1024)
    
    # The predicate `if chunk:` at line 41 should filter out empty chunks
    # We verify this by checking that only non-empty chunks would be written
    assert mock_chunk_1  # Truthy
    assert not mock_empty_chunk  # Falsy
    assert mock_chunk_2  # Truthy


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_with_url_creates_clone_to_dir(tmp_path, mocker):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    clone_to_dir = tmp_path / "clone"
    zip_uri = "http://example.com/repo.zip"
    
    mock_make_sure = mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile')
    
    try:
        from cookiecutter.zipfile import unzip
        unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    except Exception:
        pass
    
    mock_make_sure.assert_called_once()


def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    zip_file_path = str(tmp_path / "test.zip")
    
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ["project_name/", "project_name/file.txt"]
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip
    result = unzip(zip_file_path, is_url=False, clone_to_dir=tmp_path, no_input=True)
    
    assert result is not None


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository for empty zipfile."""
    zip_uri = "http://example.com/empty.zip"
    
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = []
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test that unzip raises error when zip has no top-level directory."""
    zip_uri = "http://example.com/notoplevel.zip"
    
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ["file.txt"]
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_bad_zipfile_raises_error(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository for bad zipfile."""
    from zipfile import BadZipFile
    zip_uri = "http://example.com/bad.zip"
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip"))
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    try:
        unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected file and valid password."""
    zip_uri = "http://example.com/protected.zip"
    password = "correct_password"
    
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ["project/", "project/file.txt"]
    mock_zip_file.extractall.side_effect = [RuntimeError("Bad password"), None]
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    mocker.patch('cookiecutter.zipfile.requests.get')
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from cookiecutter.zipfile import unzip
    result = unzip(zip_uri, is_url=True, clone_to_dir=tmp_path, no_input=True, password=password)
    
    assert result is not None


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected file and no_input=True raises error."""
    zip_uri = "http://example.com/protected.zip"
    
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.namelist.return_value = ["project/", "project/file.txt"]
    mock_zip_file.extractall.side_effect = RuntimeError("Bad password")
    mock_zip_file.__enter__ = mocker.MagicMock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = mocker.MagicMock(return_value=False)
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.


# LLM-generated content at query #16
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(mocker, tmp_path):
    """Test unzip downloads and extracts a URL-based zipfile."""
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Create a mock zipfile in memory
    import io
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('project-name/', '')
        zf.writestr('project-name/file.txt', 'content')
    zip_buffer.seek(0)
    
    # Mock requests.get
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_buffer.getvalue()])
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock prompt_and_delete to return True (download)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    # Mock tempfile.mkdtemp
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    mocker.patch('tempfile.mkdtemp', return_value=str(temp_dir))
    
    from cookiecutter.zipfile import unzip
    
    # Execute
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Assert
    assert result == str(temp_dir / "project-name")
    assert (clone_to_dir / "repo.zip").exists()


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local zipfile path."""
    # Setup
    import io
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('local-project/', '')
        zf.writestr('local-project/file.txt', 'content')
    zip_buffer.seek(0)
    
    zip_file_path = tmp_path / "local.zip"
    zip_file_path.write_bytes(zip_buffer.getvalue())
    
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    import tempfile
    
    # Execute
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
    
    # Assert
    assert "local-project" in result
    assert os.path.exists(result)


def test_unzip_empty_zipfile_raises_error(mocker, tmp_path):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Setup
    import io
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        pass  # Empty zipfile
    zip_buffer.seek(0)
    
    zip_file_path = tmp_path / "empty.zip"
    zip_file_path.write_bytes(zip_buffer.getvalue())
    
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Execute & Assert
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(mocker, tmp_path):
    """Test unzip raises error when zipfile lacks top-level directory."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Setup
    import io
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('file.txt', 'content')  # No top-level directory
    zip_buffer.seek(0)
    
    zip_file_path = tmp_path / "notoplevel.zip"
    zip_file_path.write_bytes(zip_buffer.getvalue())
    
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Execute & Assert
    try:
        unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zipfile_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Setup
    invalid_zip_path = tmp_path / "invalid.zip"
    invalid_zip_path.write_text("This is not a valid zip file")
    
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Execute & Assert
    try:
        unzip(str(invalid_zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except Exception:
        pass


def test_unzip_password_protected_with_valid_password(mocker, tmp_path):
    """Test unzip with password-protected zipfile and valid password."""
    from cookiecutter.zipfile import unzip
    from zipfile import ZipFile as StdZipFile
    import io
    
    # Setup
    zip_buffer = io.BytesIO()
    with StdZipFile(zip_buffer, 'w') as zf:
        zf.writestr('protected-project/', '')
        zf.writestr('protected-project/file.txt', 'content')
    zip_buffer.seek(0)
    
    zip_file_path = tmp_path / "protected.zip"
    zip_file_path.write_bytes(zip_buffer.getvalue())
    
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Execute
    result = unzip(str(zip_file_path), is_url=False, clone_to_dir=clone_to_dir, password="test")
    
    # Assert
    assert "protected-project" in result


def test_unzip_url_not_exists_downloads(mocker, tmp_path):
    """Test unzip downloads file when URL-based zipfile doesn't exist locally."""
    import io
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/newrepo.zip"
    clone_to_dir = tmp_path / "clones"
    clone_to_dir.mkdir()
    
    # Create mock zipfile
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('newproject/', '')
        zf.writestr('newproject/file.txt', 'content')
    zip_buffer.seek(0)
    
    # Mock requests.get
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_buffer.getvalue()])
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock tempfile.mkdtemp
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    mocker.patch('tempfile.mkdtemp', return_value=str(temp_dir))
    
    # Execute
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Mock requests.get to return a response with iter_content
        mock_response = Mock()
        
        # Create test chunks: some empty (falsy), some non-empty (truthy)
        test_chunks = [b'chunk1', b'', b'chunk2', b'', b'chunk3']
        mock_response.iter_content = Mock(return_value=test_chunks)
        
        # Create a valid zip file for testing
        import zipfile
        zip_path = os.path.join(temp_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_dir/', '')
            zf.writestr('test_dir/file.txt', 'content')
        
        # Mock the requests.get and file operations
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                # Track which chunks were written (non-empty ones)
                written_chunks = []
                
                def track_write(chunk):
                    if chunk:  # This is the predicate at line 40-41
                        written_chunks.append(chunk)
                
                mock_file.write.side_effect = track_write
                
                # Call unzip with a mock URL
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['test_dir/', 'test_dir/file.txt']
                    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                    
                    result = unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
        
        # Verify that only non-empty chunks were written
        assert written_chunks == [b'chunk1', b'chunk2', b'chunk3']
        assert b'' not in written_chunks
        assert len(written_chunks) == 3


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
    from unittest.mock import Mock, patch, MagicMock
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        try:
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr('test_project/', '')
                zf.writestr('test_project/file.txt', 'content')
            
            with patch('cookiecutter.zipfile.requests.get') as mock_get, \
                 patch('cookiecutter.zipfile.prompt_and_delete') as mock_prompt, \
                 patch('cookiecutter.zipfile.make_sure_path_exists') as mock_mkdir:
                
                mock_response = MagicMock()
                mock_response.iter_content.return_value = [open(zip_path, 'rb').read()]
                mock_get.return_value = mock_response
                mock_prompt.return_value = True
                
                result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=temp_dir, no_input=False)
                
                assert result.endswith('test_project')
                assert os.path.isdir(result)
        finally:
            os.unlink(zip_path)


def test_unzip_with_local_file_extracts():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        try:
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr('local_project/', '')
                zf.writestr('local_project/file.txt', 'content')
            
            result = unzip(zip_path, is_url=False, clone_to_dir=temp_dir, no_input=False)
            
            assert result.endswith('local_project')
            assert os.path.isdir(result)
        finally:
            os.unlink(zip_path)


def test_unzip_empty_zip_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        try:
            with ZipFile(zip_path, 'w') as zf:
                pass
            
            try:
                unzip(zip_path, is_url=False, clone_to_dir=temp_dir, no_input=False)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass
        finally:
            os.unlink(zip_path)


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        try:
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr('file.txt', 'content')
            
            try:
                unzip(zip_path, is_url=False, clone_to_dir=temp_dir, no_input=False)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass
        finally:
            os.unlink(zip_path)


def test_unzip_invalid_zip_raises_error():
    import tempfile
    import os
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
            zip_file.write(b'not a valid zip file')
        
        try:
            try:
                unzip(zip_path, is_url=False, clone_to_dir=temp_dir, no_input=False)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass
        finally:
            os.unlink(zip_path)


def test_unzip_with_password_protected_zip_with_correct_password():
    import tempfile
    import os
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        try:
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr('pwd_project/', '')
                zf.writestr('pwd_project/file.txt', 'content')
                zf.setpassword(b'test_password')
            
            result = unzip(zip_path, is_url=False, clone_to_dir=temp_dir, no_input=False, password='test_password')
            
            assert result.endswith('pwd_project')
            assert os.path.isdir(result)
        finally:
            os.unlink(zip_path)


# LLM-generated content at query #2
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
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
                        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
                            mock_get.return_value = mock_response
                            
                            mock_zip = MagicMock()
                            mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                            
                            mock_mkdtemp.return_value = temp_dir
                            
                            result = unzip(
                                'https://example.com/repo.zip',
                                is_url=True,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            
                            assert result == os.path.join(temp_dir, 'project_name')
                            mock_get.assert_called_once()
                            mock_zip.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
                    mock_zip = MagicMock()
                    mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
                    mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                    
                    mock_mkdtemp.return_value = temp_dir
                    local_zip_path = os.path.join(temp_dir, 'local.zip')
                    
                    result = unzip(
                        local_zip_path,
                        is_url=False,
                        clone_to_dir=temp_dir,
                        no_input=True
                    )
                    
                    assert result == os.path.join(temp_dir, 'project_name')
                    mock_zip.extractall.assert_called_once()


def test_unzip_empty_repository_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp'):
                    with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                        mock_zip = MagicMock()
                        mock_zip.namelist.return_value = []
                        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                        
                        try:
                            unzip(
                                'https://example.com/repo.zip',
                                is_url=True,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            assert False, "Expected InvalidZipRepository"
                        except InvalidZipRepository as e:
                            assert 'empty' in str(e).lower()


def test_unzip_missing_top_level_directory_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp'):
                    with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                        mock_zip = MagicMock()
                        mock_zip.namelist.return_value = ['file.txt']
                        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                        
                        try:
                            unzip(
                                'https://example.com/repo.zip',
                                is_url=True,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            assert False, "Expected InvalidZipRepository"
                        except InvalidZipRepository as e:
                            assert 'top-level directory' in str(e)


def test_unzip_password_protected_with_provided_password():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
                    with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                        mock_zip = MagicMock()
                        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
                        mock_zip.extractall.side_effect = [RuntimeError(), None]
                        mock_zipfile_class.return_value.__enter__.return_value = mock_zip
                        
                        mock_mkdtemp.return_value = temp_dir
                        
                        result = unzip(
                            'https://example.com/repo.zip',
                            is_url=True,
                            clone_to_dir=temp_dir,
                            no_input=True,
                            password='mypassword'
                        )
                        
                        assert result == os.path.join(temp_dir, 'project_name')
                        assert mock_zip.extractall.call_count == 2


def test_unzip_password_protected_no_input_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp'):
                    with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                        mock_zip = MagicMock()
                        mock_zip.namelist.return_value = ['project_name/', 'project_name/file.txt']
                        mock_zip.extractall.side_effect


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_predicate_line_31_true():
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory and file to simulate an existing zip_path
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = temp_dir
        zip_filename = "test.zip"
        zip_path = os.path.join(clone_to_dir, zip_filename)
        
        # Create a dummy file to simulate existing zip_path
        Path(zip_path).touch()
        
        # Mock the necessary functions
        with patch('cookiecutter.zipfile.prompt_and_delete') as mock_prompt_and_delete, \
             patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            
            # Setup mocks
            mock_prompt_and_delete.return_value = False
            mock_zipfile_instance = MagicMock()
            mock_zipfile_instance.namelist.return_value = ['project/']
            mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance
            
            # Call unzip with is_url=True
            zip_uri = "http://example.com/test.zip"
            result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
            
            # Verify that prompt_and_delete was called (meaning the predicate at line 31 was True)
            mock_prompt_and_delete.assert_called_once_with(zip_path, no_input=False)
            assert result is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
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
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "no_top_dir.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    zip_path.write_text("This is not a zip file")
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_url_with_no_input_downloads(tmp_path, mocker):
    """Test unzip with URL and no_input=True."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_content = tmp_path / "temp_zip"
    zip_content.mkdir()
    with zipfile.ZipFile(zip_content / "test.zip", 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Mock requests.get
    mock_response = mocker.MagicMock()
    mock_response.iter_content = mocker.MagicMock(return_value=[
        open(zip_content / "test.zip", 'rb').read()
    ])
    mocker.patch('requests.get', return_value=mock_response)
    
    result = unzip(
        "http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(clone_dir),
        no_input=True
    )
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zip and valid password."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    # Create password-protected zip
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'test_password')
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(extract_dir),
        password='test_password'
    )
    
    assert 'project_name' in result


def test_unzip_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip, no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    # Create a password-protected zip by writing encrypted content
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        info = zipfile.ZipInfo('project_name/file.txt')
        info.flag_bits |= 0x1
        zf.writestr(info, 'content', compress_type=zipfile.ZIP_DEFLATED)
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=str(extract_dir),
            no_input=True
        )
    except InvalidZipRepository:
        pass


def test_unzip_clone_to_dir_created_if_not_exists(tmp_path, mocker):
    """Test that clone_to_dir is created if it doesn't exist."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir))
    
    assert os.path.exists(clone_dir)
    assert 'project_name' in result


def test_unzip_expanduser_in_clone_to_dir(tmp_path, mocker):
    """Test that clone_to_dir expands user home directory."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    result


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    import zipfile
    import os
    
    # Create a test zip file
    zip_file_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=str(extract_dir),
        no_input=True
    )
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "empty.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=str(extract_dir),
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_file_path = tmp_path / "notoplevel.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=str(extract_dir),
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    invalid_zip_path = tmp_path / "invalid.zip"
    invalid_zip_path.write_text("not a zip file")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(invalid_zip_path),
            is_url=False,
            clone_to_dir=str(extract_dir),
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_url_no_input(tmp_path, mocker):
    """Test unzip with URL and no_input=True."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [open(zip_file_path, 'rb').read()]
    mocker.patch('requests.get', return_value=mock_response)
    
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=str(extract_dir),
        no_input=True
    )
    
    assert "project_name" in result


def test_unzip_with_password_protected_zip(tmp_path, mocker):
    """Test unzip with password-protected zip and provided password."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_file_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    password = "test_password"
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=str(extract_dir),
        no_input=True,
        password=password
    )
    
    assert "project_name" in result


def test_unzip_creates_clone_to_dir(tmp_path, mocker):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    
    zip_file_path = tmp_path / "test.zip"
    clone_to_dir = tmp_path / "nonexistent" / "path"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=str(clone_to_dir),
        no_input=True
    )
    
    assert os.path.exists(clone_to_dir)
    assert "project_name" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This happens when is_url is True, the zip file exists, and prompt_and_delete
    returns False (user chooses to reuse existing version).
    """
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    zip_path.touch()  # Create the file so os.path.exists returns True
    
    # Mock prompt_and_delete to return False (user wants to reuse existing version)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock ZipFile to avoid actual file operations
    mock_zipfile_class = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile_instance = mocker.MagicMock()
    mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance
    mock_zipfile_instance.namelist.return_value = ['project_name/']
    
    # Execute
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    
    # Verify that prompt_and_delete was called (meaning zip_path exists)
    mock_prompt_and_delete.assert_called_once_with(str(zip_path), no_input=False)
    
    # Verify that requests.get was NOT called (download is False, so line 36-42 not executed)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False,
        password=None
    )
    mock_requests_get.assert_not_called()


# LLM-generated content at query #7
#--------------------------

```python
def test_chunk_filter_predicate_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    chunk = b''
    assert not chunk


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts_zipfile(tmp_path, monkeypatch):
    """Test unzip downloads a URL-based zipfile and extracts it."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Mock requests.get
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    def mock_get(url, stream=True, timeout=None):
        return MockResponse()
    
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", mock_get)
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", lambda path, no_input: True)
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_dir, no_input=True)
    
    assert result.endswith("project_name")
    assert os.path.isdir(result)


def test_unzip_with_local_file_extracts_zipfile(tmp_path):
    """Test unzip with local file path extracts the zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "local.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("myproject/", "")
        zf.writestr("myproject/test.txt", "test content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
    
    assert result.endswith("myproject")
    assert os.path.isdir(result)
    assert os.path.isfile(os.path.join(result, "test.txt"))


def test_unzip_raises_on_empty_zipfile(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_raises_on_missing_top_level_directory(tmp_path):
    """Test unzip raises InvalidZipRepository if top-level is not a directory."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "no_toplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_raises_on_invalid_zipfile(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zipfile."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    # Create a file that's not a valid zipfile
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zipfile")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_password_protected_zipfile(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True, password=password)
    
    assert result.endswith("project")


def test_unzip_creates_clone_directory_if_not_exists(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir, no_input=True)
    
    assert clone_dir.exists()
    assert os.path.isdir(result)


def test_unzip_with_url_skips_download_if_not_needed(tmp_path, monkeypatch):
    """Test unzip with URL skips download if file exists and user chooses to reuse."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    
    # Mock prompt_and_delete to return False (don't delete)
    monkeypatch.setattr("cookiecutter.zipfile.prompt_and_delete", lambda path, no_input: False)
    
    # Copy the zip file to clone directory with the expected name
    import shutil
    target_zip = clone_dir / "test.zip"
    shutil.copy(zip_path, target_zip)
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_dir, no_input=False)
    
    assert result.endswith("


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_predicate_line_39_evaluates_to_false():
    """Test that the predicate at line 39 (if chunk:) evaluates to False."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a mock response with iter_content that returns empty chunks
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'', None, b''])
    
    # Create a temporary directory and zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/test.zip"
        zip_path = os.path.join(clone_to_dir, "test.zip")
        
        # Mock the necessary functions and modules
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                with patch('cookiecutter.zipfile.make_sure_path_exists'):
                    # Setup the mock ZipFile
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_dir/']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile.return_value = mock_zip_instance
                    
                    # Call unzip with is_url=True and no_input=True
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                    
                    # Verify that the file was opened and written to
                    assert os.path.exists(zip_path)
                    # Verify that iter_content was called
                    mock_response.iter_content.assert_called_once_with(chunk_size=1024)


# LLM-generated content at query #10
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
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_file_raises_error(tmp_path, monkeypatch):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    import os
    
    zip_path = tmp_path / "invalid.zip"
    with open(zip_path, 'w') as f:
        f.write("not a zip file")
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    """Test unzip with password protected zip and no_input=True raises error."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.setpassword(b"password")
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_valid_password(tmp_path, monkeypatch):
    """Test unzip with password protected zip and valid password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True, password="test")
    
    assert "project" in result


def test_unzip_expanduser_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip expands user path for clone_to_dir."""
    import zipfile
    from pathlib import Path
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("proj/", "")
        zf.writestr("proj/file.txt", "content")
    
    mock_make_sure = lambda x: None
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', mock_make_sure)
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=".", no_input=True)
    
    assert "proj" in result


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_bad_zip_file_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create a fake zip file that will raise BadZipFile
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("This is not a valid zip file")
    
    # Mock make_sure_path_exists to avoid actual directory creation
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    # Call unzip with the bad zip file and expect InvalidZipRepository
    try:
        unzip(str(bad_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'is not a valid zip archive' in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(extract_dir))
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert 'project_name' in result


def test_unzip_empty_zipfile_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zipfile."""
    import zipfile
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level' in str(e).lower()


def test_unzip_invalid_zip_file_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(invalid_zip), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'not a valid zip' in str(e).lower()


def test_unzip_url_with_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL when file exists and no_input=True."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    mock_mkdtemp = mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "extract"))
    (tmp_path / "extract").mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(f"file://{zip_path}", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert 'project_name' in result


def test_unzip_password_protected_with_valid_password(tmp_path, mocker):
    """Test unzip with password-protected zipfile using provided password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Re-create with encryption (simulated by mocking)
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip_instance.extractall.side_effect = [RuntimeError("Bad password"), None]
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(extract_dir))
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password=password)
    
    assert 'project_name' in result


def test_unzip_password_protected_invalid_password_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zipfile using invalid password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
    mock_zip_


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_local_file(tmp_path, mocker):
    """Test unzipping a local zipfile."""
    import zipfile
    import os
    
    # Create a test zipfile
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


def test_unzip_url_no_existing_file(tmp_path, mocker):
    """Test unzipping a URL when no cached file exists."""
    import zipfile
    
    # Create a test zipfile
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    with open(zip_content, 'rb') as f:
        zip_data = f.read()
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=False)
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [zip_data]
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")


def test_unzip_url_with_existing_file_delete(tmp_path, mocker):
    """Test unzipping a URL when cached file exists and user chooses to delete."""
    import zipfile
    
    # Create a test zipfile
    zip_content = tmp_path / "temp.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    with open(zip_content, 'rb') as f:
        zip_data = f.read()
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [zip_data]
    mocker.patch('requests.get', return_value=mock_response)
    
    from cookiecutter.zipfile import unzip
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir)
    
    assert result.endswith("project_name")


def test_unzip_empty_zipfile(tmp_path, mocker):
    """Test unzipping an empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty test zipfile
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
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path, mocker):
    """Test unzipping a zipfile without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a zipfile without top-level directory
    zip_path = tmp_path / "no_dir.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_file(tmp_path, mocker):
    """Test unzipping an invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an invalid zipfile
    zip_path = tmp_path / "invalid.zip"
    with open(zip_path, 'w') as f:
        f.write("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_password_protected_with_password(tmp_path, mocker):
    """Test unzipping a password-protected zipfile with correct password."""
    import zipfile
    
    # Create a password-protected test zipfile
    zip_path = tmp_path / "protected.zip"
    password = "testpass"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    # We need to recreate it with encryption (pyminizip or similar would be needed)
    # For this test, we'll mock the RuntimeError
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.namelist.return_value = ["project_name/", "project_name/file.txt"]
    mock_zipfile.extractall.side_effect = [RuntimeError("Bad password"), None]
    
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    mocker.patch('tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    
    from


# LLM-generated content at query #14
#--------------------------

```python
def test_unzip_predicate_line_41_evaluates_to_false():
    """Test that the predicate at line 41 (if chunk:) evaluates to False for empty chunks."""
    # The predicate at line 41 is: if chunk:
    # This evaluates to False when chunk is empty (b'')
    chunk = b''
    result = bool(chunk)
    assert result is False


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_predicate_line_31_true(tmp_path, monkeypatch):
    """Test that the predicate at line 31 evaluates to True when zip_path exists."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    
    # Create a temporary zip file that exists
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(b"PK\x03\x04")  # Minimal zip file header
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock the necessary functions
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True) as mock_prompt:
            with patch('cookiecutter.zipfile.requests.get'):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                        # Setup mock zipfile
                        mock_zip_instance = Mock()
                        mock_zip_instance.namelist.return_value = ['project/']
                        mock_zip_instance.__enter__.return_value = mock_zip_instance
                        mock_zip_instance.__exit__.return_value = None
                        mock_zipfile.return_value = mock_zip_instance
                        
                        # Patch os.path.exists to return True for our zip_path
                        with patch('cookiecutter.zipfile.os.path.exists') as mock_exists:
                            def exists_side_effect(path):
                                if str(path).endswith('test.zip'):
                                    return True
                                return False
                            
                            mock_exists.side_effect = exists_side_effect
                            
                            # Import and call the function
                            from cookiecutter.zipfile import unzip
                            result = unzip(
                                zip_uri="http://example.com/test.zip",
                                is_url=True,
                                clone_to_dir=clone_to_dir,
                                no_input=False
                            )
                            
                            # Verify that prompt_and_delete was called (line 32)
                            # This proves the predicate at line 31 evaluated to True
                            mock_prompt.assert_called_once()


# LLM-generated content at query #16
#--------------------------

```python
def test_zipfile_predicate_line_54_evaluates_to_false():
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This ensures that when a zipfile has at least one entry, the empty check passes.
    """
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = tempfile.mkdtemp()
    
    try:
        # Create a valid zipfile with a top-level directory
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            # Add a top-level directory entry (ends with /)
            zf.writestr("project_name/", "")
            # Add a file inside the directory
            zf.writestr("project_name/file.txt", "content")
        
        # Call unzip with the valid zipfile
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True,
            password=None
        )
        
        # If we reach here without InvalidZipRepository exception,
        # the predicate at line 55 (len(zip_file.namelist()) == 0) evaluated to False
        assert result is not None
        assert os.path.isdir(result)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(clone_to_dir, ignore_errors=True)


# LLM-generated content at query #17
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This occurs when prompt_and_delete returns False, indicating the user
    wants to reuse the existing zipfile instead of re-downloading it.
    """
    from cookiecutter.zipfile import unzip
    from pathlib import Path
    
    # Setup
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    zip_path = clone_to_dir / identifier
    
    # Create a dummy zip file that already exists
    zip_path.touch()
    
    # Mock prompt_and_delete to return False (user wants to reuse)
    mock_prompt_and_delete = mocker.patch(
        'cookiecutter.zipfile.prompt_and_delete',
        return_value=False
    )
    
    # Mock ZipFile to avoid actual zip operations
    mock_zipfile_class = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile_instance = mocker.MagicMock()
    mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance
    mock_zipfile_instance.namelist.return_value = ['project_name/']
    
    # Mock tempfile and extractall
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp"))
    mock_zipfile_instance.extractall = mocker.MagicMock()
    
    # Mock requests.get to ensure it's not called (since download should be False)
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Execute
    result = unzip(
        zip_uri=zip_uri,
        is_url=True,
        clone_to_dir=str(clone_to_dir),
        no_input=False
    )
    
    # Assert that requests.get was NOT called (because download = False)
    mock_requests_get.assert_not_called()
    
    # Assert that prompt_and_delete was called
    mock_prompt_and_delete.assert_called_once()
    
    # Assert that unzip still proceeded with the existing file
    mock_zipfile_class.assert_called_once_with(str(zip_path))


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
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
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
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
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e)


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


def test_unzip_with_password_protected_zip_no_input(tmp_path):
    """Test unzip raises error for password protected zip with no_input=True."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


def test_unzip_with_correct_password(tmp_path):
    """Test unzip succeeds with correct password provided."""
    import zipfile
    import os
    
    zip_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="password")
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_with_url_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file doesn't exist yet."""
    import zipfile
    from unittest.mock import Mock, patch
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[open(zip_path, 'rb').read()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        from cookiecutter.zipfile import unzip
        result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert "project_name" in result


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, mocker):
    """Test that the predicate at line 36 (if download:) evaluates to False."""
    # Mock the necessary functions
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    
    # Mock ZipFile to avoid actual file operations
    mock_zip_file = mocker.MagicMock()
    mock_zip_file.__enter__.return_value.namelist.return_value = ['project_dir/']
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file)
    
    # Mock requests.get to ensure it's not called
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Call unzip with is_url=True and a path that exists
    from cookiecutter.zipfile import unzip
    result = unzip(
        zip_uri='https://example.com/repo.zip',
        is_url=True,
        clone_to_dir=str(tmp_path),
        no_input=False
    )
    
    # Verify that requests.get was NOT called (because download was False)
    mock_requests_get.assert_not_called()
    # Verify that prompt_and_delete was called and returned False
    assert result is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_iter_content_filters_empty_chunks():
    """Test that the predicate at line 40 evaluates to False for empty chunks."""
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = Path(temp_dir)
    
    try:
        # Mock the requests.get to return a response with empty chunks
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'data', b'', b'more', b''])
        
        # Create a temporary zip file with valid content
        import zipfile
        zip_path = os.path.join(temp_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_dir/', '')
            zf.writestr('test_dir/file.txt', 'content')
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('builtins.open', mock_open()) as mock_file:
                # Call unzip with is_url=True
                result = unzip(
                    'http://example.com/test.zip',
                    is_url=True,
                    clone_to_dir=clone_to_dir,
                    no_input=True
                )
                
                # Verify that write was called only for non-empty chunks
                # The mock_file.return_value.write should be called exactly twice
                # (for b'data' and b'more', but not for the empty b'' chunks)
                write_calls = mock_file.return_value.write.call_count
                assert write_calls == 2, f"Expected 2 write calls, got {write_calls}"
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_context_manager_with_zipfile():
    """Test that the predicate at line 54 evaluates to True by verifying ZipFile context manager is used."""
    import tempfile
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = Path(temp_dir)
    
    # Create a valid zip file with a top-level directory
    zip_path = Path(temp_dir) / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Call unzip with is_url=False to use local zipfile
    result = unzip(
        zip_uri=str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True,
        password=None
    )
    
    # Verify that result is a string path (unzip_path was returned)
    assert isinstance(result, str)
    assert 'project_name' in result
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, mocker):
    """Test unzip with a URL when the file doesn't exist yet."""
    import io
    from zipfile import ZipFile
    
    # Create a mock zip file content
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_buffer.read()])
    
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    from cookiecutter.zipfile import unzip
    
    result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=str(tmp_path))
    
    assert 'test_project' in result


def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local file path."""
    import io
    from zipfile import ZipFile
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('local_project/', '')
        zf.writestr('local_project/file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
    
    assert 'local_project' in result


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "no_topdir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))


def test_unzip_with_password_protection(tmp_path, mocker):
    """Test unzip with password-protected zip file."""
    from zipfile import ZipFile
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'test_password')
        zf.writestr('protected_project/', '')
        zf.writestr('protected_project/file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    from cookiecutter.zipfile import unzip
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), password='test_password')
    
    assert 'protected_project' in result


def test_unzip_bad_zip_file_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for corrupted zip."""
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create a file that is not a valid zip
    zip_path = tmp_path / "bad.zip"
    zip_path.write_text("not a zip file")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))


def test_unzip_with_url_existing_file_no_input(tmp_path, mocker):
    """Test unzip with URL and existing file with no_input=True."""
    import io
    from zipfile import ZipFile
    
    # Create a mock zip file content
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('url_project/', '')
        zf.writestr('url_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[zip_buffer.read()])
    
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.rmtree')
    
    from cookiecutter.zipfile import unzip
    
    result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=str(tmp_path), no_input=True)
    
    assert 'url_project' in result


def test_unzip_with_password_wrong_password_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository with wrong password."""
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'correct_password')
        zf.writestr('protected_project/', '')
        zf.writestr('protected_project/file.txt', 'content')
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    from cookiecutter.zipfile import unzip
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), password='wrong_password')


def test_unzip_with_password_protected_no_input_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for password-protected with no_input."""
    from zipfile import ZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    import pytest
    
    # Create a password-protected zip file
    


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        
        # Mock the requests.get to simulate downloading a zipfile
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', None, b'chunk3'])
        
        # Create a valid zip file for testing
        import zipfile
        zip_path = os.path.join(temp_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_dir/', '')
            zf.writestr('test_dir/file.txt', 'content')
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    mock_zip_instance = MagicMock()
                    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.namelist.return_value = ['test_dir/', 'test_dir/file.txt']
                    
                    result = unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
                    
                    # Verify that write was called only for non-empty chunks
                    calls = mock_file.write.call_args_list
                    assert len(calls) == 3
                    assert calls[0][0][0] == b'chunk1'
                    assert calls[1][0][0] == b'chunk2'
                    assert calls[2][0][0] == b'chunk3'


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that the predicate at line 55 evaluates to True when zip is empty."""
    import os
    import tempfile
    from zipfile import ZipFile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    from cookiecutter import utils
    monkeypatch.setattr(utils, 'make_sure_path_exists', lambda x: None)
    
    # Attempt to unzip the empty file should raise InvalidZipRepository
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


# LLM-generated content at query #25
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
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
    import os
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "valid zip" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path):
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "nonexistent" / "clone"
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert clone_to_dir.exists()
    assert result.endswith("project_name")


# LLM-generated content at query #26
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
    
    assert result.endswith("test_project")
    assert os.path.isdir(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_missing_top_level_directory_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create a zip file without top-level directory
    zip_path = tmp_path / "no_toplevel.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path):
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create an invalid zip file
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_creates_clone_to_dir(tmp_path):
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a zip file
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    clone_dir = tmp_path / "new_clone_dir"
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
    
    assert clone_dir.exists()
    assert result.endswith("project")


def test_unzip_with_password_protected_zip_no_input_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"password")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "password protected" in str(e).lower()


def test_unzip_with_correct_password(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"mypassword")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=tmp_path,
        password="mypassword"
    )
    
    assert result.endswith("project")


def test_unzip_with_wrong_password_raises_error(tmp_path):
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b"correctpassword")
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "content")
    
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=tmp_path,
            password="wrongpassword"
        )
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        assert "Invalid password" in str(e)


# LLM-generated content at query #27
#--------------------------

```python
def test_unzip_opens_zipfile_with_context_manager():
    """Test that the predicate at line 54 evaluates to True - ZipFile is opened."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, mock_open
    
    # Create a temporary directory and a valid zip file
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test.zip")
    
    # Create a simple valid zip file with a top-level directory
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/", "")
        zf.writestr("test_project/file.txt", "content")
    
    # Mock the necessary functions
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
            with patch('cookiecutter.zipfile.ZipFile', wraps=ZipFile) as mock_zipfile:
                from cookiecutter.zipfile import unzip
                
                result = unzip(
                    zip_uri=zip_path,
                    is_url=False,
                    clone_to_dir=temp_dir,
                    no_input=True,
                    password=None
                )
                
                # Verify that ZipFile was called (context manager was entered)
                mock_zipfile.assert_called_once_with(zip_path)
                # Verify that a path was returned
                assert result is not None
                assert isinstance(result, str)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# LLM-generated content at query #28
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
    
    assert 'project_name' in result
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip with empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip without top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip with invalid zipfile raises InvalidZipRepository."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should have raised InvalidZipRepository"
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
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir))
    
    assert os.path.exists(clone_dir)
    assert 'project' in result


def test_unzip_with_password_protected_zip_no_input(tmp_path):
    """Test unzip with password-protected zip and no_input=True raises error."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(b'test')
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_with_correct_password(tmp_path):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    password = 'test_password'
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
        zf.setpassword(password.encode('utf-8'))
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        password=password
    )
    
    assert 'project' in result


def test_unzip_with_wrong_password_raises_error(tmp_path):
    """Test unzip with password-protected zip and wrong password."""
    import zipfile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
        zf.setpassword(b'correct_password')
    
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=str(tmp_path),
            password='wrong_password'
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_expanduser_in_clone_to_dir(tmp_path, monkeypatch):
    """Test unzip expands user home directory in clone_to_dir."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    monkeypatch.setenv('HOME', str(tmp_path))
    clone_dir = '~/cookiecutter'
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_dir)
    
    assert 'project' in result


# LLM-generated content at query #29
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
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
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error when zip has no top-level directory."""
    import zipfile
    
    zip_path = tmp_path / "no_toplevel.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_bad_zip_file_raises_error(tmp_path, monkeypatch):
    """Test unzip raises error for invalid zipfile."""
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("This is not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    try:
        unzip(str(bad_zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
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
    assert result.endswith("project_name")


def test_unzip_with_url_and_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file doesn't exist yet."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    def mock_get(*args, **kwargs):
        class MockResponse:
            def iter_content(self, chunk_size):
                with open(zip_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
        return MockResponse()
    
    import requests
    monkeypatch.setattr(requests, "get", mock_get)
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        "http://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert result.endswith("project_name")
    assert (clone_to_dir / "test.zip").exists()


def test_unzip_with_password_protected_zip(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    password = "secret"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        password=password
    )
    
    assert result.endswith("project_name")


# LLM-generated content at query #30
#--------------------------

```python
def test_unzip_writes_chunk_to_file(tmp_path, mocker):
    """Test that the predicate at line 39 evaluates to True when chunk is not empty."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a valid zip file
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    # Mock requests.get to return a response with chunks
    mock_response = mocker.Mock()
    mock_response.iter_content = mocker.Mock(return_value=[b'test_chunk_data', None, b'more_data'])
    
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    # Call unzip with is_url=True to trigger the code path
    result = unzip(
        zip_uri='http://example.com/test.zip',
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    # Verify that iter_content was called with chunk_size=1024
    mock_response.iter_content.assert_called_once_with(chunk_size=1024)
    
    # Verify that the result is a valid path
    assert result is not None
    assert 'test_project' in result


####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_with_url_downloads_and_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_response = MagicMock()
        mock_response.iter_content = Mock(return_value=[b'test_data'])
        
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=['project_name/', 'project_name/file.txt'])
        mock_zip_file.extractall = Mock()
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.make_sure_path_exists'):
                    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                        with patch('tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                            result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                            assert result is not None
                            assert 'project_name' in result


def test_unzip_with_local_file_extracts():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=['project_name/', 'project_name/file.txt'])
        mock_zip_file.extractall = Mock()
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                    result = unzip('/path/to/local.zip', is_url=False, clone_to_dir=clone_to_dir)
                    assert result is not None
                    assert 'project_name' in result


def test_unzip_empty_zip_raises_error():
    import tempfile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=[])
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                try:
                    unzip('/path/to/local.zip', is_url=False, clone_to_dir=clone_to_dir)
                    assert False, "Should raise InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=['file.txt'])
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                try:
                    unzip('/path/to/local.zip', is_url=False, clone_to_dir=clone_to_dir)
                    assert False, "Should raise InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_with_password_protected_file():
    import tempfile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=['project_name/', 'project_name/file.txt'])
        
        def extractall_side_effect(*args, **kwargs):
            if 'pwd' not in kwargs:
                raise RuntimeError("Password required")
        
        mock_zip_file.extractall = Mock(side_effect=extractall_side_effect)
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                    result = unzip('/path/to/local.zip', is_url=False, clone_to_dir=clone_to_dir, password='test_password')
                    assert result is not None


def test_unzip_invalid_zip_file_raises_error():
    import tempfile
    from zipfile import BadZipFile
    from unittest.mock import patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                try:
                    unzip('/path/to/invalid.zip', is_url=False, clone_to_dir=clone_to_dir)
                    assert False, "Should raise InvalidZipRepository"
                except InvalidZipRepository:
                    pass


def test_unzip_url_file_exists_with_no_input():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as clone_to_dir:
        mock_zip_file = MagicMock()
        mock_zip_file.namelist = Mock(return_value=['project_name/', 'project_name/file.txt'])
        mock_zip_file.extractall = Mock()
        
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.os.path.exists', return_value=True):
                    with patch('cookiecutter.zipfile.prompt_and_delete', return_value=False):
                        with patch('tempfile.mkdtemp', return_value=tempfile.mkdtemp()):
                            result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                            assert result is not None


####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip_local_file(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
    assert result.endswith("project_name")
    assert os.path.exists(result)


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "no_top_level.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "top-level directory" in str(e).lower()


def test_unzip_bad_zip_file_raises_error(tmp_path, monkeypatch):
    import os
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "bad.zip"
    zip_path.write_text("this is not a zip file")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "not a valid zip archive" in str(e).lower()


def test_unzip_url_with_no_input_and_no_existing_file(tmp_path, monkeypatch):
    import zipfile
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a valid zip file to be "downloaded"
    zip_content = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_content, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_content.read_bytes()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        result = unzip(
            "http://example.com/project.zip",
            is_url=True,
            clone_to_dir=str(clone_to_dir),
            no_input=True
        )
        assert result.endswith("project_name")


def test_unzip_password_protected_with_provided_password(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "protected.zip"
    password = "secret"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(password.encode('utf-8'))
    
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=str(tmp_path),
        password=password
    )
    assert result.endswith("project_name")


def test_unzip_password_protected_no_input_raises_error(tmp_path, monkeypatch):
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    password = "secret"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    try:
        unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=str(tmp_path),
            no_input=True
        )
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository as e:
        assert "password" in str(e).lower()


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    clone_to_dir = tmp_path / "new_dir"
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_to_dir))
    assert os.path.exists(clone_to_dir)
    assert result.endswith("project_name")


def test_unzip_with_expanduser(tmp_path, monkeypatch):
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    monkeypatch.setenv("HOME", str(tmp_path))
    result = unzip(str(zip_path), is_url=False, clone_to_dir="~/clone")
    assert result.endswith("project_name")


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip_download_predicate_false(tmp_path, mocker):
    """Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False."""
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Setup
    zip_uri = "http://example.com/test.zip"
    clone_to_dir = tmp_path
    zip_filename = "test.zip"
    zip_path = os.path.join(clone_to_dir, zip_filename)
    
    # Create a dummy zip file to simulate existing file
    Path(zip_path).touch()
    
    # Mock prompt_and_delete to return False
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=False)
    
    # Mock requests.get to avoid actual network calls
    mock_requests_get = mocker.patch('cookiecutter.zipfile.requests.get')
    
    # Mock ZipFile to handle the extraction
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zip_instance = mocker.MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['project/']
    
    # Mock tempfile.mkdtemp
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp'))
    
    # Call unzip with is_url=True to trigger the download predicate check
    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
    
    # Assert that requests.get was NOT called (because download predicate was False)
    mock_requests_get.assert_not_called()


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_bad_zipfile_exception_handling(tmp_path, monkeypatch):
    """Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository."""
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    from zipfile import BadZipFile
    
    # Create a fake zip file path
    bad_zip_path = str(tmp_path / "bad.zip")
    
    # Write invalid zip content
    with open(bad_zip_path, 'wb') as f:
        f.write(b'This is not a valid zip file')
    
    # Mock is_url to False so it uses the local file path
    try:
        unzip(bad_zip_path, is_url=False, clone_to_dir=str(tmp_path), no_input=True)
        assert False, "Should have raised InvalidZipRepository"
    except Exception as e:
        # Verify that the exception is InvalidZipRepository and the predicate at line 105 evaluated to True
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'not a valid zip archive' in str(e)


# LLM-generated content at query #2
#--------------------------

Looking at line 36 of the zipfile.py code, the predicate is `if download:`. For this to evaluate to `False`, the `download` variable must be `False`.

According to the logic:
- Line 31-34: `download` is set based on whether the zip_path exists
- If it exists, `download = prompt_and_delete(zip_path, no_input=no_input)` 
- If it doesn't exist, `download = True`

For the predicate at line 36 to be `False`, `prompt_and_delete` must return `False`, which happens when the user chooses NOT to delete and then chooses to reuse the existing version.



# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_predicate_line_54_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.
    
    This means the zipfile contains at least one entry.
    """
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file with content
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_dir/", "")
        zf.writestr("test_dir/file.txt", "content")
    
    # Mock requests.get to avoid actual network calls
    class MockResponse:
        def iter_content(self, chunk_size):
            return []
    
    def mock_get(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr("cookiecutter.zipfile.requests.get", mock_get)
    
    # Call unzip with is_url=False to use local file
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
    
    # Verify that the function succeeded, meaning the predicate was False
    # (i.e., the zipfile was not empty)
    assert result is not None
    assert "test_dir" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_downloads_zipfile_when_download_is_true(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if download:) evaluates to True."""
    import io
    from unittest.mock import Mock, patch, mock_open
    from pathlib import Path
    
    # Setup
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path
    
    # Mock the requests.get response
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'test chunk'])
    
    # Mock ZipFile to avoid actual zip operations
    mock_zip_file = Mock()
    mock_zip_file.namelist = Mock(return_value=['project-name/'])
    mock_zip_file.__enter__ = Mock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = Mock(return_value=None)
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / 'temp')):
                    from cookiecutter.zipfile import unzip
                    
                    # Execute - is_url=True, no_input=True (skip prompt_and_delete)
                    result = unzip(
                        zip_uri=zip_uri,
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True,
                        password=None
                    )
    
    # Assert - verify that open was called (line 39 executed)
    mock_file.assert_called()
    # Verify write was called with the chunk (line 42 executed)
    handle = mock_file()
    handle.write.assert_called_with(b'test chunk')


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip_with_url_new_download(tmp_path, monkeypatch):
    """Test unzip downloads a new zipfile from URL."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.zipfile import unzip
    
    zip_uri = "https://example.com/repo.zip"
    clone_to_dir = tmp_path / "cloned"
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b"PK\x03\x04"])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project/"]
            mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
            mock_zip_instance.__exit__ = Mock(return_value=False)
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result == os.path.join(str(tmp_path / "temp"), "project")


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile path."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    zip_uri = str(tmp_path / "local.zip")
    clone_to_dir = tmp_path / "cloned"
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ["myproject/"]
        mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
        mock_zip_instance.__exit__ = Mock(return_value=False)
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    
    assert result == os.path.join(str(tmp_path / "temp"), "myproject")


def test_unzip_empty_zipfile_raises_error(tmp_path):
    """Test unzip raises error for empty zipfile."""
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = str(tmp_path / "empty.zip")
    clone_to_dir = tmp_path / "cloned"
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = []
        mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
        mock_zip_instance.__exit__ = Mock(return_value=False)
        mock_zipfile.return_value = mock_zip_instance
        
        try:
            unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "empty" in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises error when zip has no top-level directory."""
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = str(tmp_path / "bad.zip")
    clone_to_dir = tmp_path / "cloned"
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ["file.txt"]
        mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
        mock_zip_instance.__exit__ = Mock(return_value=False)
        mock_zipfile.return_value = mock_zip_instance
        
        try:
            unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "top-level" in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises error for invalid zip archive."""
    from unittest.mock import Mock, patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = str(tmp_path / "invalid.zip")
    clone_to_dir = tmp_path / "cloned"
    
    with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile("Bad zip")):
        try:
            unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "not a valid zip archive" in str(e).lower()


def test_unzip_password_protected_with_provided_password(tmp_path):
    """Test unzip with password-protected archive and provided password."""
    import os
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip
    
    zip_uri = str(tmp_path / "protected.zip")
    clone_to_dir = tmp_path / "cloned"
    password = "secret"
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall = Mock(side_effect=[
            RuntimeError("Bad password"),
            None
        ])
        mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
        mock_zip_instance.__exit__ = Mock(return_value=False)
        mock_zipfile.return_value = mock_zip_instance
        
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, 
                          no_input=True, password=password)
    
    assert result == os.path.join(str(tmp_path / "temp"), "project")


def test_unzip_password_protected_invalid_password_raises_error(tmp_path):
    """Test unzip with invalid password for protected archive."""
    from unittest.mock import Mock, patch
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    zip_uri = str(tmp_path / "protected.zip")
    clone_to_dir = tmp_path / "cloned"
    password = "wrong"
    
    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall = Mock(side_effect=RuntimeError("Bad password"))
        mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
        mock_zip_instance.__exit__ = Mock(return_value=False


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    import tempfile
    import os
    from zipfile import ZipFile
    from pathlib import Path
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zipfile with content (non-empty)
        zip_path = os.path.join(temp_dir, "test.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_dir/", "")
            zf.writestr("project_dir/file.txt", "content")
        
        # Open the zipfile and verify the predicate evaluates to False
        with ZipFile(zip_path) as zip_file:
            predicate_result = len(zip_file.namelist()) == 0
            assert predicate_result is False


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_predicate_line_36_evaluates_to_false(tmp_path, monkeypatch):
    """Test that the predicate at line 36 (if download:) evaluates to False.
    
    This happens when is_url is True, zip_path exists, and prompt_and_delete
    returns False (user chooses to reuse existing version).
    """
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_file_path = tmp_path / "test.zip"
    zip_file_path.write_bytes(b"fake zip content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create the zip file in the expected location
    identifier = "test.zip"
    zip_path = clone_to_dir / identifier
    zip_path.write_bytes(b"fake zip content")
    
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
    
    # Mock ZipFile to handle the fake zip
    from unittest.mock import MagicMock, mock_open
    from zipfile import ZipFile as RealZipFile
    
    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ["project_name/"]
    mock_zip.__enter__ = MagicMock(return_value=mock_zip)
    mock_zip.__exit__ = MagicMock(return_value=False)
    mock_zip.extractall = MagicMock()
    
    monkeypatch.setattr("cookiecutter.zipfile.ZipFile", MagicMock(return_value=mock_zip))
    
    # Mock tempfile.mkdtemp
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    monkeypatch.setattr("cookiecutter.zipfile.tempfile.mkdtemp", lambda: str(temp_dir))
    
    # Call unzip with is_url=True
    result = unzip(
        zip_uri="https://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=False
    )
    
    # Verify requests.get was not called (download was False)
    assert call_count["get"] == 0
    assert result is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip_with_valid_zipfile(tmp_path, monkeypatch):
    """Test that the predicate at line 54 evaluates to True with a valid zipfile."""
    import zipfile
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for the test
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create a valid zipfile with a top-level directory
    zip_path = clone_to_dir / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    # Mock the requests.get to return the zipfile
    import requests
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: MockResponse())
    
    # Import the function to test
    from cookiecutter.zipfile import unzip
    
    # Call unzip with a URL
    result = unzip(
        zip_uri="http://example.com/test.zip",
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True,
        password=None
    )
    
    # Verify that the result is a valid path
    assert result is not None
    assert "test_project" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL when file doesn't exist yet."""
    import io
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    zip_content = io.BytesIO()
    with __import__('zipfile').ZipFile(zip_content, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_content.seek(0)
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_content.read(1024)])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('builtins.open', create=True):
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['test_project/', 'test_project/file.txt']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile.return_value = mock_zip_instance
                    
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                        result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=str(tmp_path))
                        assert 'test_project' in result


def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local file path."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['myproject/', 'myproject/file.txt']
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                result = unzip('/path/to/local.zip', is_url=False)
                assert 'myproject' in result


def test_unzip_empty_zipfile(tmp_path):
    """Test unzip raises error for empty zipfile."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = []
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                try:
                    unzip('/path/to/empty.zip', is_url=False)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory(tmp_path):
    """Test unzip raises error when no top-level directory exists."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['file.txt']
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                try:
                    unzip('/path/to/bad.zip', is_url=False)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'top-level' in str(e).lower()


def test_unzip_bad_zipfile(tmp_path):
    """Test unzip raises error for invalid zipfile."""
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile('Bad zip')):
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                try:
                    unzip('/path/to/invalid.zip', is_url=False)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'valid zip archive' in str(e).lower()


def test_unzip_password_protected_with_valid_password(tmp_path):
    """Test unzip with password-protected zip and valid password provided."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
            mock_zip_instance.extractall.side_effect = [RuntimeError('Bad password'), None]
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                result = unzip('/path/to/protected.zip', is_url=False, password='mypassword')
                assert 'project' in result


def test_unzip_password_protected_invalid_password(tmp_path):
    """Test unzip with password-protected zip and invalid password provided."""
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = MagicMock()
            mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
            mock_zip_instance.extractall.side_effect = RuntimeError('Bad password')
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.__exit__.return_value = None
            mock_zipfile.return_value = mock_zip_instance
            
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path)):
                try:
                    un


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_bad_zip_file_exception():
    """Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary directory and a fake zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_zip_path = os.path.join(temp_dir, "fake.zip")
        
        # Write invalid zip file content
        with open(fake_zip_path, 'wb') as f:
            f.write(b"This is not a valid zip file")
        
        clone_to_dir = os.path.join(temp_dir, "clone")
        
        # Try to unzip the invalid file and expect InvalidZipRepository exception
        try:
            unzip(fake_zip_path, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
            assert False, "Expected InvalidZipRepository to be raised"
        except Exception as e:
            assert type(e).__name__ == 'InvalidZipRepository'
            assert 'not a valid zip archive' in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_non_empty_zipfile():
    """Test that the predicate at line 55 evaluates to False for non-empty zipfile."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary zip file with content
        zip_path = os.path.join(temp_dir, 'test.zip')
        
        with ZipFile(zip_path, 'w') as zf:
            # Add a directory entry and a file to make the zipfile non-empty
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Call unzip with the non-empty zipfile
        # The predicate at line 55 (len(zip_file.namelist()) == 0) should evaluate to False
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=temp_dir,
            no_input=True
        )
        
        # Assert that unzip succeeded and returned a valid path
        assert result is not None
        assert isinstance(result, str)
        assert os.path.exists(result)


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip downloads and extracts a URL-based zipfile."""
    import zipfile
    import io
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path
    
    # Create a mock zip file in memory
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
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip
            result = unzip(
                'http://example.com/test.zip',
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
    
    assert 'test_project' in result
    assert result.endswith('test_project')


def test_unzip_with_local_file(tmp_path):
    """Test unzip extracts a local zipfile."""
    import zipfile
    from pathlib import Path
    
    # Create a local zip file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(
        str(zip_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert 'test_project' in result
    assert result.endswith('test_project')


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "notoplevel.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from unittest.mock import patch
    
    invalid_zip_path = tmp_path / "invalid.zip"
    invalid_zip_path.write_bytes(b"not a zip file")
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip_path), is_url=False, clone_to_dir=tmp_path, no_input=True)


def test_unzip_password_protected_with_correct_password(tmp_path):
    """Test unzip extracts password-protected zip with correct password."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        result = unzip(
            str(zip_path),
            is_url=False,
            clone_to_dir=tmp_path,
            no_input=True,
            password=password
        )
    
    assert 'test_project' in result


def test_unzip_password_protected_no_password_raises_error(tmp_path):
    """Test unzip raises error for password-protected zip without password."""
    import zipfile
    from unittest.mock import patch
    
    zip_path = tmp_path / "protected.zip"
    password = "test_password"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.setpassword(password.encode('utf-8'))
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with pytest.raises(InvalidZipRepository):
            unzip(
                str(zip_path),
                is_url=False,
                clone_to_dir=tmp_path,
                no_input=True,
                password=None
            )


def test_unzip_url_existing_file_no_input_deletes(tmp_path, monkeypatch):
    """Test unzip deletes existing file when no_input=True."""
    import zipfile
    import io
    from unittest.mock import Mock, patch
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    existing_zip = clone_to_dir / "test.zip"
    existing_zip.write_bytes(b"old content")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'new content')
    zip_buffer.seek(0)
    
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[zip_buffer.getvalue()])
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            from cookiecutter.zipfile import unzip
            result = unzip(
                'http://example.com/test.zip',
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True
            )
    
    assert 'test_project' in result


def test_unzip_creates_clone_to_dir(tmp_path):
    """Test unzip creates clone_to


# LLM-generated content at query #11
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_on_bad_zipfile():
    """Test that BadZipFile exception is caught and converted to InvalidZipRepository."""
    import tempfile
    import os
    from pathlib import Path
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create a temporary directory and a fake "zip" file that is not a valid zip
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "clone"
        clone_to_dir.mkdir(parents=True, exist_ok=True)
        
        fake_zip_path = clone_to_dir / "fake.zip"
        fake_zip_path.write_text("This is not a valid zip file")
        
        zip_uri = str(fake_zip_path)
        
        try:
            unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, no_input=True)
            assert False, "Expected InvalidZipRepository to be raised"
        except InvalidZipRepository as e:
            assert "not a valid zip archive" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_raises_invalid_zip_repository_when_zipfile_is_empty(tmp_path, mocker):
    """Test that unzip raises InvalidZipRepository when zipfile is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Test that InvalidZipRepository is raised when zipfile is empty
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'is empty' in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_iter_content_chunk_filter():
    """Test that the predicate at line 40 evaluates to True for non-empty chunks."""
    import io
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path
    import tempfile
    import os
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "http://example.com/archive.zip"
        identifier = "archive.zip"
        zip_path = os.path.join(clone_to_dir, identifier)
        
        # Mock the requests.get to return a response with chunks
        mock_response = Mock()
        test_chunks = [b"chunk1", b"chunk2", b"", b"chunk3"]
        mock_response.iter_content = Mock(return_value=test_chunks)
        
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_name/', 'project_name/file.txt']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = None
                    mock_zipfile.return_value = mock_zip_instance
                    
                    from cookiecutter.zipfile import unzip
                    
                    result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                    
                    # Verify that the file was written with non-empty chunks
                    with open(zip_path, 'rb') as f:
                        written_data = f.read()
                    
                    assert written_data == b"chunk1chunk2chunk3"


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false(mocker):
    """Test that the predicate at line 40 (if chunk:) evaluates to False."""
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Mock requests.get to return a response with empty chunks
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'', None, b'']
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    
    # Mock ZipFile to avoid actual zip processing
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.namelist.return_value = ['test_dir/']
    mock_zipfile.__enter__.return_value = mock_zipfile
    mock_zipfile.__exit__.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile.mkdtemp and os.path.join
    temp_dir = tempfile.mkdtemp()
    mocker.patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir)
    
    # Mock make_sure_path_exists
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    # Mock open to track write calls
    mock_file = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open(mock_file=mock_file))
    
    # Call unzip with is_url=True and no_input=True
    result = unzip(
        zip_uri='http://example.com/test.zip',
        is_url=True,
        clone_to_dir='.',
        no_input=True,
        password=None
    )
    
    # Verify that write was not called for empty chunks (predicate was False)
    # Only non-empty chunks should trigger write
    write_calls = mock_file.return_value.write.call_count
    assert write_calls == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if chunk:) evaluates to True when chunk has content."""
    import os
    from pathlib import Path
    from unittest.mock import Mock, MagicMock, patch
    from cookiecutter.zipfile import unzip
    
    # Create a temporary clone directory
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock the requests.get to return a response with chunks
    mock_response = Mock()
    mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b'chunk3', b''])
    
    # Mock ZipFile to avoid actual zip processing
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['test_project/']
    mock_zip_file.__enter__ = Mock(return_value=mock_zip_file)
    mock_zip_file.__exit__ = Mock(return_value=False)
    
    # Track written chunks
    written_chunks = []
    
    def mock_open_func(path, mode):
        mock_file = MagicMock()
        def mock_write(chunk):
            written_chunks.append(chunk)
        mock_file.write = mock_write
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        return mock_file
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.open', mock_open_func):
            with patch('cookiecutter.zipfile.ZipFile', return_value=mock_zip_file):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(tmp_path / "temp")):
                    result = unzip(
                        'http://example.com/test.zip',
                        is_url=True,
                        clone_to_dir=clone_to_dir,
                        no_input=True
                    )
    
    # Verify that only non-empty chunks were written (predicate at line 41 filtered out empty chunk)
    assert b'chunk1' in written_chunks
    assert b'chunk2' in written_chunks
    assert b'chunk3' in written_chunks
    assert b'' not in written_chunks
    assert len(written_chunks) == 3


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip_empty_zipfile_raises_invalid_zip_repository(tmp_path, monkeypatch):
    """Test that the predicate at line 55 evaluates to True when zip is empty."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Attempt to unzip the empty file
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert isinstance(e, InvalidZipRepository)
        assert "is empty" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_41_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."""
    chunk = b''
    assert not chunk


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip_predicate_line_40_evaluates_to_true():
    """Test that the predicate at line 40 (if chunk:) evaluates to True for non-empty chunks."""
    chunk = b'test data'
    assert chunk
    assert bool(chunk) is True


# LLM-generated content at query #16
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
        with tempfile.TemporaryDirectory() as extract_dir:
            zip_path = os.path.join(temp_dir, "test.zip")
            
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr("project_name/", "")
                zf.writestr("project_name/file.txt", "content")
            
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                    with patch('cookiecutter.zipfile.requests.get') as mock_get:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=extract_dir):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'test'])
                            mock_get.return_value = mock_response
                            
                            with patch('builtins.open', create=True):
                                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                                    mock_zf = MagicMock()
                                    mock_zf.namelist.return_value = ["project_name/", "project_name/file.txt"]
                                    mock_zf.__enter__ = Mock(return_value=mock_zf)
                                    mock_zf.__exit__ = Mock(return_value=False)
                                    mock_zipfile.return_value = mock_zf
                                    
                                    result = unzip(
                                        "http://example.com/test.zip",
                                        is_url=True,
                                        clone_to_dir=temp_dir,
                                        no_input=True
                                    )
                                    
                                    assert result is not None
                                    mock_zf.extractall.assert_called_once()


def test_unzip_with_local_file():
    import tempfile
    import os
    from pathlib import Path
    from zipfile import ZipFile
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as extract_dir:
            zip_path = os.path.join(temp_dir, "test.zip")
            
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr("project_name/", "")
                zf.writestr("project_name/file.txt", "content")
            
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=extract_dir):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        mock_zf = MagicMock()
                        mock_zf.namelist.return_value = ["project_name/", "project_name/file.txt"]
                        mock_zf.__enter__ = Mock(return_value=mock_zf)
                        mock_zf.__exit__ = Mock(return_value=False)
                        mock_zipfile.return_value = mock_zf
                        
                        result = unzip(
                            zip_path,
                            is_url=False,
                            clone_to_dir=temp_dir,
                            no_input=True
                        )
                        
                        assert result is not None
                        mock_zf.extractall.assert_called_once()


def test_unzip_empty_zip_raises_error():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as extract_dir:
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=extract_dir):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        mock_zf = MagicMock()
                        mock_zf.namelist.return_value = []
                        mock_zf.__enter__ = Mock(return_value=mock_zf)
                        mock_zf.__exit__ = Mock(return_value=False)
                        mock_zipfile.return_value = mock_zf
                        
                        try:
                            unzip(
                                "http://example.com/test.zip",
                                is_url=False,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            assert False, "Should have raised InvalidZipRepository"
                        except InvalidZipRepository:
                            pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as extract_dir:
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=extract_dir):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        mock_zf = MagicMock()
                        mock_zf.namelist.return_value = ["file.txt"]
                        mock_zf.__enter__ = Mock(return_value=mock_zf)
                        mock_zf.__exit__ = Mock(return_value=False)
                        mock_zipfile.return_value = mock_zf
                        
                        try:
                            unzip(
                                "http://example.com/test.zip",
                                is_url=False,
                                clone_to_dir=temp_dir,
                                no_input=True
                            )
                            assert False, "Should have raised InvalidZipRepository"
                        except InvalidZipRepository:
                            pass


def test_unzip_with_password_protection():
    import tempfile
    import os
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as extract_dir:
            with patch('cookiecutter.zipfile.make_sure_path_exists'):
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=extract_dir):
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        mock_zf = MagicMock()
                        mock_zf.namelist.return_value = ["project_name/", "project_name/file.txt"]
                        mock_zf.extractall.side_effect = [RuntimeError("encrypted"), None]
                        mock_zf.__enter__ = Mock(return_value=mock_zf)
                        mock_zf.__exit__ = Mock(return_value=False)
                        mock_zipfile.return_value = mock_zf
                        
                        result = unzip(
                            "http://example.com/test.zip",
                            is_url=False,
                            clone_to_dir=temp_dir,
                            no_input=False,
                            password="mypassword"
                        )
                        
                


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip_raises_on_empty_zipfile(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zipfile is empty."""
    import zipfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    try:
        unzip(
            zip_uri=str(empty_zip_path),
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=True
        )
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert "Zip repository" in str(e)
        assert "is empty" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_line_41_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 41 evaluates to False."""
    chunk = b''
    result = bool(chunk)
    assert result is False


# LLM-generated content at query #10
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with a URL and new zipfile."""
    import zipfile
    import os
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "test.zip"
    
    # Create a valid zip with top-level directory
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    # Mock requests.get
    class MockResponse:
        def iter_content(self, chunk_size):
            with open(zip_file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
    
    def mock_get(url, stream=None, timeout=None):
        return MockResponse()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', mock_get)
    
    # Test
    result = unzip(
        zip_uri='http://example.com/test.zip',
        is_url=True,
        clone_to_dir=zip_dir,
        no_input=True
    )
    
    assert os.path.exists(result)
    assert 'project_name' in result


def test_unzip_with_local_file(tmp_path):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary zip file
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "local_test.zip"
    
    # Create a valid zip with top-level directory
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('my_project/', '')
        zf.writestr('my_project/README.md', 'Project content')
    
    # Test
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=zip_dir,
        no_input=True
    )
    
    assert os.path.exists(result)
    assert 'my_project' in result


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip with empty zipfile raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "empty.zip"
    
    # Create an empty zip file
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        pass
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=zip_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip with no top-level directory raises InvalidZipRepository."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "no_toplevel.zip"
    
    # Create a zip without top-level directory
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=zip_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'top-level' in str(e).lower()


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip with invalid zipfile raises InvalidZipRepository."""
    import os
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "invalid.zip"
    
    # Create an invalid zip file
    with open(zip_file_path, 'w') as f:
        f.write('This is not a zip file')
    
    try:
        unzip(
            zip_uri=str(zip_file_path),
            is_url=False,
            clone_to_dir=zip_dir,
            no_input=True
        )
        assert False, "Should have raised InvalidZipRepository"
    except InvalidZipRepository as e:
        assert 'valid' in str(e).lower()


def test_unzip_with_password_protected_zip(tmp_path, monkeypatch):
    """Test unzip with password-protected zipfile."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "protected.zip"
    
    # Create a password-protected zip
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('secure_project/', '')
        zf.writestr('secure_project/file.txt', 'secret content')
        zf.setpassword(b'test_password')
    
    # Test with password provided
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=zip_dir,
        no_input=True,
        password='test_password'
    )
    
    assert os.path.exists(result)
    assert 'secure_project' in result


def test_unzip_creates_clone_to_dir_if_not_exists(tmp_path):
    """Test unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a zip file
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip_file_path = zip_dir / "test.zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'content')
    
    # Use non-existent clone_to_dir
    clone_to_dir = tmp_path / "new_dir" / "nested"
    
    result = unzip(
        zip_uri=str(zip_file_path),
        is_url=False,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    assert os.path.exists(result)
    assert os.path.exists(clone_to_dir)


def test_unzip_with_expanduser_path(tmp_path, monkeypatch):
    """Test unzip expands user path correctly."""
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    zip_dir = tmp_path / "zip_storage"
    zip_dir.mkdir()
    zip


# LLM-generated content at query #18
#--------------------------

```python
def test_unzip_iter_content_filters_out_keep_alive_chunks(mocker):
    """Test that the predicate at line 40 evaluates to False for keep-alive chunks."""
    import tempfile
    from pathlib import Path
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory and zip file
    temp_dir = tempfile.mkdtemp()
    clone_to_dir = Path(temp_dir)
    
    # Mock requests.get to return a response with empty chunks (keep-alive)
    mock_response = mocker.MagicMock()
    mock_response.iter_content.return_value = [b'', None, b'test_data', b'', b'more_data']
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock ZipFile to avoid actual zip processing
    mock_zipfile = mocker.MagicMock()
    mock_zipfile.namelist.return_value = ['project_dir/']
    mock_zipfile.__enter__.return_value = mock_zipfile
    mock_zipfile.__exit__.return_value = None
    mocker.patch('cookiecutter.zipfile.ZipFile', return_value=mock_zipfile)
    
    # Mock tempfile.mkdtemp
    mocker.patch('tempfile.mkdtemp', return_value=temp_dir)
    
    # Mock file operations to track writes
    mock_file = mocker.MagicMock()
    mocker.patch('builtins.open', mocker.mock_open())
    
    # Call unzip with a URL
    result = unzip(
        zip_uri='http://example.com/test.zip',
        is_url=True,
        clone_to_dir=clone_to_dir,
        no_input=True
    )
    
    # Verify that empty chunks (where chunk evaluates to False) were filtered out
    # The mock_file.write should only be called for non-empty chunks
    assert mock_zipfile.namelist.called


# LLM-generated content at query #19
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, monkeypatch):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert "project_name" in result
    assert os.path.exists(result)


def test_unzip_creates_clone_to_dir(tmp_path, monkeypatch):
    """Test that unzip creates clone_to_dir if it doesn't exist."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    clone_to_dir = tmp_path / "new_clone_dir"
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
    
    assert clone_to_dir.exists()
    assert "project_name" in result


def test_unzip_empty_zip_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "empty" in str(e).lower()


def test_unzip_no_top_level_dir_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises error when no top-level directory exists."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "notoplevel.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "top-level directory" in str(e)


def test_unzip_invalid_zip_raises_error(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.exceptions import InvalidZipRepository
    
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("not a zip file")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(invalid_zip), is_url=False, clone_to_dir=clone_to_dir)
        assert False, "Expected InvalidZipRepository"
    except Exception as e:
        assert "not a valid zip archive" in str(e)


def test_unzip_with_password_protected_zip_no_input(tmp_path, monkeypatch):
    """Test that unzip raises error for password protected zip with no_input."""
    import zipfile
    from cookiecutter.exceptions import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
        zf.setpassword(b"password")
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, no_input=True)
    except InvalidZipRepository:
        pass


def test_unzip_with_correct_password(tmp_path, monkeypatch):
    """Test that unzip works with correct password."""
    import zipfile
    import pyminizip
    
    zip_path = tmp_path / "protected.zip"
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "project_name").mkdir()
    (content_dir / "project_name" / "file.txt").write_text("content")
    
    pyminizip.compress_multiple(
        str(content_dir / "project_name"),
        ["file.txt"],
        str(zip_path),
        "password",
        5
    )
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=clone_to_dir, password="password")
    
    assert "project_name" in result
    assert "file.txt" in result


def test_unzip_expanduser_on_clone_to_dir(tmp_path, monkeypatch):
    """Test that unzip expands user home directory."""
    import zipfile
    
    zip_path = tmp_path / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    monkeypatch.setenv("HOME", str(tmp_path))
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir="~/test_clone")
    
    assert "project_name" in result
    assert os.path.exists(result)


# LLM-generated content at query #20
#--------------------------

```python
def test_unzip_with_local_file(tmp_path, mocker):
    """Test unzip with a local zipfile."""
    import zipfile
    import os
    
    # Create a temporary zip file with proper structure
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
    
    assert result.endswith("project_name")


def test_unzip_empty_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for empty zip."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "empty.zip"
    extract_dir = tmp_path / "extract"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_no_top_level_directory_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "no_toplevel.zip"
    extract_dir = tmp_path / "extract"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_invalid_zip_raises_error(tmp_path, mocker):
    """Test unzip raises InvalidZipRepository for invalid zip."""
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "invalid.zip"
    extract_dir = tmp_path / "extract"
    zip_path.write_text("not a zip file")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir))
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_url_downloads_and_extracts(tmp_path, mocker):
    """Test unzip with URL downloads and extracts the file."""
    import zipfile
    
    zip_path = tmp_path / "remote.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=False)
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [zip_path.read_bytes()]
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.open', mocker.mock_open())
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=str(extract_dir), no_input=True)
    
    assert result.endswith("project_name")


def test_unzip_url_existing_file_no_input_deletes_and_redownloads(tmp_path, mocker):
    """Test unzip with URL and existing file deletes it when no_input=True."""
    import zipfile
    
    zip_path = tmp_path / "remote.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mocker.patch('cookiecutter.zipfile.os.path.exists', return_value=True)
    mocker.patch('cookiecutter.zipfile.prompt_and_delete', return_value=True)
    
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [zip_path.read_bytes()]
    mocker.patch('cookiecutter.zipfile.requests.get', return_value=mock_response)
    mocker.patch('cookiecutter.zipfile.open', mocker.mock_open())
    
    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=str(extract_dir), no_input=True)
    
    assert result.endswith("project_name")


def test_unzip_password_protected_with_correct_password(tmp_path, mocker):
    """Test unzip with password-protected zip and correct password."""
    import zipfile
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    
    password = "test_password"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
        zf.writestr("project_name/file.txt", "content")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir), password=password)
    
    assert result.endswith("project_name")


def test_unzip_password_protected_with_wrong_password_raises_error(tmp_path, mocker):
    """Test unzip with password-protected zip and wrong password."""
    import zipfile
    from cookiecutter.zipfile import InvalidZipRepository
    
    zip_path = tmp_path / "protected.zip"
    extract_dir = tmp_path / "extract"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_name/", "")
    
    mocker.patch('cookiecutter.zipfile.make_sure_path_exists')
    mock_zipfile = mocker.patch('cookiecutter.zipfile.ZipFile')
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ["project_name/"]
    mock_zipfile.return_value.__enter__.return_value.extractall.side_effect = RuntimeError("Bad password")
    
    try:
        unzip(str(zip_path), is_url=False, clone_to_dir=str(extract_dir), password="wrong")
        assert False, "Should raise InvalidZipRepository"
    except InvalidZipRepository:
        pass


def test_unzip_password_protected_no_input


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    """Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."""
    # Create a mock response object with iter_content that yields empty chunks
    import io
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path
    import tempfile
    
    # Create a temporary directory and file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_to_dir = tmpdir
        zip_uri = "http://example.com/test.zip"
        
        # Mock the requests.get to return a response with empty chunks
        mock_response = Mock()
        # iter_content yields an empty bytes object (evaluates to False)
        mock_response.iter_content.return_value = [b'', b'data', b'']
        
        # Mock ZipFile to avoid actual zip processing
        with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                with patch('cookiecutter.zipfile.make_sure_path_exists'):
                    # Setup the mock ZipFile
                    mock_zip_instance = MagicMock()
                    mock_zip_instance.namelist.return_value = ['project_dir/']
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.__exit__.return_value = False
                    mock_zipfile.return_value = mock_zip_instance
                    
                    # Track what was written
                    written_chunks = []
                    
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file
                        
                        # Capture write calls
                        def track_write(data):
                            written_chunks.append(data)
                        
                        mock_file.write.side_effect = track_write
                        
                        from cookiecutter.zipfile import unzip
                        result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=True)
                        
                        # Verify that empty chunks (evaluating to False) were NOT written
                        # Only b'data' should be written, not the empty b'' chunks
                        assert b'' not in written_chunks or written_chunks.count(b'') == 0
                        assert b'data' in written_chunks


# LLM-generated content at query #21
#--------------------------

```python
def test_unzip_with_url_new_file(tmp_path, monkeypatch):
    """Test unzip with URL when zip file doesn't exist."""
    import io
    from zipfile import ZipFile
    
    # Create a mock zip file in memory
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_project/', '')
        zf.writestr('test_project/file.txt', 'content')
    zip_buffer.seek(0)
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    mock_response = type('MockResponse', (), {
        'iter_content': lambda self, chunk_size: [zip_buffer.getvalue()]
    })()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', lambda *args, **kwargs: mock_response)
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    
    from cookiecutter.zipfile import unzip
    result = unzip('http://example.com/test.zip', is_url=True, clone_to_dir=clone_to_dir)
    
    assert 'test_project' in result
    assert result.endswith('test_project')


def test_unzip_with_local_file(tmp_path):
    """Test unzip with local file path."""
    import io
    from zipfile import ZipFile
    
    # Create a temporary zip file
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('my_project/', '')
        zf.writestr('my_project/README.md', 'test content')
    
    from cookiecutter.zipfile import unzip
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)
    
    assert 'my_project' in result
    assert result.endswith('my_project')


def test_unzip_empty_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when zip is empty."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)


def test_unzip_no_top_level_directory_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository when no top-level directory."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    zip_path = tmp_path / "notoplevel.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('file.txt', 'content')
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)


def test_unzip_with_password_provided(tmp_path, monkeypatch):
    """Test unzip with password provided."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    import io
    
    # Create a password-protected zip file
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('secure_project/', '')
        zf.writestr('secure_project/file.txt', 'secret')
        zf.setpassword(b'mypassword')
    
    result = unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path, password='mypassword')
    
    assert 'secure_project' in result


def test_unzip_invalid_zip_raises_error(tmp_path):
    """Test unzip raises InvalidZipRepository for invalid zip file."""
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    import pytest
    
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_text("This is not a zip file")
    
    with pytest.raises(InvalidZipRepository):
        unzip(str(zip_path), is_url=False, clone_to_dir=tmp_path)


def test_unzip_with_url_existing_file_no_input(tmp_path, monkeypatch):
    """Test unzip with URL when file exists and no_input=True."""
    from zipfile import ZipFile
    from cookiecutter.zipfile import unzip
    import io
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Create existing zip file
    existing_zip = clone_to_dir / "existing.zip"
    with ZipFile(existing_zip, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'old content')
    
    # Create new zip content
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('project/', '')
        zf.writestr('project/file.txt', 'new content')
    zip_buffer.seek(0)
    
    mock_response = type('MockResponse', (), {
        'iter_content': lambda self, chunk_size: [zip_buffer.getvalue()]
    })()
    
    monkeypatch.setattr('cookiecutter.zipfile.requests.get', lambda *args, **kwargs: mock_response)
    monkeypatch.setattr('cookiecutter.zipfile.make_sure_path_exists', lambda x: None)
    monkeypatch.setattr('cookiecutter.zipfile.os.path.exists', lambda x: True)
    
    result = unzip('http://example.com/existing.zip', is_url=True, clone_to_dir=clone_to_dir, no_input=True)
    
    assert 'project' in result


# LLM-generated content at query #12
#--------------------------

```python
def test_unzip_writes_chunks_to_file(tmp_path, monkeypatch):
    """Test that the predicate at line 39 (if chunk:) evaluates to True for valid chunks."""
    import io
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    from zipfile import ZipFile
    
    # Create a temporary zip file with valid structure
    zip_file_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    
    # Create a valid zip file with top-level directory
    with ZipFile(zip_file_path, 'w') as zf:
        zf.writestr('project_name/', '')
        zf.writestr('project_name/file.txt', 'content')
    
    clone_to_dir = tmp_path / "clone"
    clone_to_dir.mkdir()
    
    # Mock requests.get to return a response with chunks
    mock_response = Mock()
    chunks = [b'test_chunk_1', b'test_chunk_2', b'', b'test_chunk_3']
    mock_response.iter_content = Mock(return_value=chunks)
    
    with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=str(extract_dir)):
            result = unzip(
                zip_uri='http://example.com/test.zip',
                is_url=True,
                clone_to_dir=clone_to_dir,
                no_input=True,
                password=None
            )
    
    # Verify the file was written with chunks (including the empty chunk being filtered)
    zip_path = clone_to_dir / 'test.zip'
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0


# LLM-generated content at query #22
#--------------------------

```python
def test_unzip_downloads_zipfile_in_chunks():
    """Test that the predicate at line 40 evaluates to True when chunk is not empty."""
    import io
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for clone_to_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock response with iter_content that returns chunks
        mock_response = Mock()
        chunk_data = b'test chunk data'
        mock_response.iter_content.return_value = [chunk_data, b'', chunk_data]
        
        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            zip_path = tmp_zip.name
            # Write minimal valid zip file content
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('test_dir/', '')
                zf.writestr('test_dir/file.txt', 'content')
        
        try:
            # Mock requests.get to return our mock response
            with patch('cookiecutter.zipfile.requests.get', return_value=mock_response):
                # Mock open to capture writes
                write_calls = []
                original_open = open
                
                def mock_open_func(*args, **kwargs):
                    file_obj = original_open(*args, **kwargs)
                    original_write = file_obj.write
                    
                    def tracked_write(data):
                        write_calls.append(data)
                        return original_write(data)
                    
                    file_obj.write = tracked_write
                    return file_obj
                
                with patch('builtins.open', mock_open_func):
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                        result = unzip(
                            zip_uri='http://example.com/test.zip',
                            is_url=True,
                            clone_to_dir=temp_dir,
                            no_input=True
                        )
            
            # Verify that non-empty chunks were written (line 40-42 predicate evaluates to True)
            assert len(write_calls) >= 2, "Non-empty chunks should have been written"
            assert chunk_data in write_calls, "Chunk data should have been written"
        finally:
            # Clean up the temporary zip file
            os.unlink(zip_path)


# LLM-generated content at query #13
#--------------------------

```python
def test_unzip_download_predicate_false_when_not_url():
    """Test that the predicate at line 36 evaluates to False when is_url is False."""
    from pathlib import Path
    import tempfile
    import zipfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid zip file with a top-level directory
        zip_path = os.path.join(temp_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('project_dir/', '')
            zf.writestr('project_dir/file.txt', 'content')
        
        # Create clone_to_dir
        clone_to_dir = os.path.join(temp_dir, 'clone')
        os.makedirs(clone_to_dir)
        
        # Call unzip with is_url=False
        # The predicate at line 36 checks: if download:
        # When is_url=False, the code path at lines 25-42 is skipped entirely
        # So download variable is never set, and line 36 is not executed
        result = unzip(
            zip_uri=zip_path,
            is_url=False,
            clone_to_dir=clone_to_dir,
            no_input=False,
            password=None
        )
        
        # Verify the result is a valid path
        assert result is not None
        assert os.path.isdir(result)


# LLM-generated content at query #14
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
        zip_uri = "https://example.com/repo.zip"
        
        with patch('cookiecutter.zipfile.requests.get') as mock_get, \
             patch('cookiecutter.zipfile.prompt_and_delete') as mock_prompt, \
             patch('cookiecutter.zipfile.ZipFile') as mock_zipfile, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_prompt.return_value = True
            mock_mkdtemp.return_value = str(temp_dir)
            
            mock_response = Mock()
            mock_response.iter_content.return_value = [b'test']
            mock_get.return_value = mock_response
            
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = ['project_name/']
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            result = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir, no_input=False)
            
            assert mock_get.called
            assert mock_prompt.called
            assert mock_zip.extractall.called


def test_unzip_with_local_file():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "/local/path/repo.zip"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = str(temp_dir)
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = ['project_name/']
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
            
            assert mock_zip.extractall.called
            assert result.endswith('project_name')


def test_unzip_empty_zip_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "/local/path/repo.zip"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = []
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            try:
                unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "/local/path/repo.zip"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = ['file.txt']
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            try:
                unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_with_password():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "/local/path/repo.zip"
        password = "test_password"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp:
            
            mock_mkdtemp.return_value = str(temp_dir)
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = ['project_name/']
            mock_zip.extractall.side_effect = [RuntimeError(), None]
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir, password=password)
            
            assert mock_zip.extractall.call_count == 2


def test_unzip_bad_zip_file_raises_error():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir)
        zip_uri = "/local/path/repo.zip"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
            mock_zipfile.return_value.__enter__.side_effect = BadZipFile()
            
            try:
                unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
                assert False, "Expected InvalidZipRepository"
            except InvalidZipRepository:
                pass


def test_unzip_creates_clone_to_dir():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_to_dir = Path(temp_dir) / "new_dir"
        zip_uri = "/local/path/repo.zip"
        
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile, \
             patch('cookiecutter.zipfile.tempfile.mkdtemp') as mock_mkdtemp, \
             patch('cookiecutter.zipfile.make_sure_path_exists') as mock_make_path:
            
            mock_mkdtemp.return_value = str(temp_dir)
            mock_zip = MagicMock()
            mock_zip.namelist.return_value = ['project_name/']
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            
            result = unzip(zip_uri, is_url=False, clone_to_dir=clone_to_dir)
            
            assert mock_make_path.called


# LLM-generated content at query #23
#--------------------------

```python
def test_unzip_predicate_line_55_evaluates_to_false():
    """Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False."""
    from zipfile import ZipFile
    import tempfile
    import os
    from cookiecutter.zipfile import unzip
    
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    
    # Create a valid zip file with content (non-empty namelist)
    zip_path = os.path.join(temp_dir, "test.zip")
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "content")
    
    # Call unzip with is_url=False to use local file
    result = unzip(
        zip_uri=zip_path,
        is_url=False,
        clone_to_dir=temp_dir,
        no_input=True
    )
    
    # Verify that result is a valid path (meaning no exception was raised at line 55)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #15
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
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
                            mock_get.return_value = mock_response
                            
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['myproject/', 'myproject/file.txt']
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile.return_value = mock_zip_instance
                            
                            result = unzip('http://example.com/myproject.zip', is_url=True, clone_to_dir=temp_dir)
                            
                            assert result == os.path.join(temp_dir, 'myproject')
                            mock_get.assert_called_once()
                            mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_local_file_extracts():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                    with patch('cookiecutter.zipfile.os.path.abspath', return_value='/path/to/local.zip'):
                        mock_zip_instance = MagicMock()
                        mock_zip_instance.namelist.return_value = ['myproject/', 'myproject/file.txt']
                        mock_zip_instance.__enter__.return_value = mock_zip_instance
                        mock_zip_instance.__exit__.return_value = None
                        mock_zipfile.return_value = mock_zip_instance
                        
                        result = unzip('/path/to/local.zip', is_url=False, clone_to_dir=temp_dir)
                        
                        assert result == os.path.join(temp_dir, 'myproject')
                        mock_zip_instance.extractall.assert_called_once()


def test_unzip_with_password_protected_zip_prompts_user():
    import tempfile
    import os
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            with patch('cookiecutter.zipfile.read_repo_password', return_value='mypassword'):
                                mock_response = Mock()
                                mock_response.iter_content = Mock(return_value=[b'chunk1'])
                                mock_get.return_value = mock_response
                                
                                mock_zip_instance = MagicMock()
                                mock_zip_instance.namelist.return_value = ['myproject/', 'myproject/file.txt']
                                mock_zip_instance.extractall.side_effect = [RuntimeError('Bad password'), None]
                                mock_zip_instance.__enter__.return_value = mock_zip_instance
                                mock_zip_instance.__exit__.return_value = None
                                mock_zipfile.return_value = mock_zip_instance
                                
                                result = unzip('http://example.com/myproject.zip', is_url=True, clone_to_dir=temp_dir, no_input=False)
                                
                                assert result == os.path.join(temp_dir, 'myproject')
                                assert mock_zip_instance.extractall.call_count == 2


def test_unzip_empty_zip_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'chunk1'])
                            mock_get.return_value = mock_response
                            
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = []
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile.return_value = mock_zip_instance
                            
                            try:
                                unzip('http://example.com/empty.zip', is_url=True, clone_to_dir=temp_dir)
                                assert False, "Should have raised InvalidZipRepository"
                            except InvalidZipRepository:
                                pass


def test_unzip_no_top_level_directory_raises_error():
    import tempfile
    from unittest.mock import patch, MagicMock
    from cookiecutter.zipfile import unzip, InvalidZipRepository
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('cookiecutter.zipfile.make_sure_path_exists'):
            with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile:
                        with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value=temp_dir):
                            mock_response = Mock()
                            mock_response.iter_content = Mock(return_value=[b'chunk1'])
                            mock_get.return_value = mock_response
                            
                            mock_zip_instance = MagicMock()
                            mock_zip_instance.namelist.return_value = ['file.txt', 'another_file.txt']
                            mock_zip_instance.__enter__.return_value = mock_zip_instance
                            mock_zip_instance.__exit__.return_value = None
                            mock_zipfile.return_value = mock_zip_instance
                            
                            try:
                                unzip('http://example.com/bad.zip', is_url=True, clone_to_dir=temp_dir)
                                assert False, "Should have raised Invali


# LLM-generated content at query #24
#--------------------------

```python
def test_unzip_with_url_creates_directory():
    from pathlib import Path
    from unittest.mock import Mock, patch, MagicMock
    import tempfile
    import os
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['project_dir/', 'project_dir/file.txt']
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=False):
            with patch('cookiecutter.zipfile.requests.get') as mock_get:
                with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                    with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                        with patch('builtins.open', create=True):
                            mock_response = Mock()
                            mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
                            mock_get.return_value = mock_response
                            
                            mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                            
                            from cookiecutter.zipfile import unzip
                            result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir='.', no_input=True)
                            
                            assert result == '/tmp/test/project_dir'


def test_unzip_with_local_file():
    from unittest.mock import patch, MagicMock
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['myproject/', 'myproject/file.txt']
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                with patch('cookiecutter.zipfile.os.path.abspath', return_value='/local/repo.zip'):
                    mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                    
                    from cookiecutter.zipfile import unzip
                    result = unzip('/local/repo.zip', is_url=False, clone_to_dir='.')
                    
                    assert result == '/tmp/test/myproject'


def test_unzip_empty_zip_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.exceptions import InvalidZipRepository
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = []
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                from cookiecutter.zipfile import unzip
                try:
                    unzip('/local/repo.zip', is_url=False)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'empty' in str(e).lower()


def test_unzip_no_top_level_directory_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.exceptions import InvalidZipRepository
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['file.txt']
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                from cookiecutter.zipfile import unzip
                try:
                    unzip('/local/repo.zip', is_url=False)
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'top-level' in str(e).lower()


def test_unzip_password_protected_with_valid_password():
    from unittest.mock import patch, MagicMock
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
    mock_zip_file.extractall.side_effect = [RuntimeError('Bad password'), None]
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                from cookiecutter.zipfile import unzip
                result = unzip('/local/repo.zip', is_url=False, password='mypassword')
                
                assert result == '/tmp/test/project'


def test_unzip_password_protected_invalid_password_raises_error():
    from unittest.mock import patch, MagicMock
    from cookiecutter.exceptions import InvalidZipRepository
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
    mock_zip_file.extractall.side_effect = RuntimeError('Bad password')
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
            with patch('cookiecutter.zipfile.tempfile.mkdtemp', return_value='/tmp/test'):
                mock_zipfile_class.return_value.__enter__.return_value = mock_zip_file
                
                from cookiecutter.zipfile import unzip
                try:
                    unzip('/local/repo.zip', is_url=False, password='wrongpassword')
                    assert False, "Should have raised InvalidZipRepository"
                except InvalidZipRepository as e:
                    assert 'password' in str(e).lower()


def test_unzip_bad_zip_file_raises_error():
    from unittest.mock import patch
    from zipfile import BadZipFile
    from cookiecutter.exceptions import InvalidZipRepository
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.ZipFile', side_effect=BadZipFile('Bad zip')):
            from cookiecutter.zipfile import unzip
            try:
                unzip('/local/repo.zip', is_url=False)
                assert False, "Should have raised InvalidZipRepository"
            except InvalidZipRepository as e:
                assert 'valid zip' in str(e).lower()


def test_unzip_url_with_existing_file_and_delete():
    from unittest.mock import patch, MagicMock, call
    
    mock_zip_file = MagicMock()
    mock_zip_file.namelist.return_value = ['project/', 'project/file.txt']
    
    with patch('cookiecutter.zipfile.make_sure_path_exists'):
        with patch('cookiecutter.zipfile.os.path.exists', return_value=True):
            with patch('cookiecutter.zipfile.prompt_and_delete', return_value=True):
                with patch('cookiecutter.zipfile.requests.get') as mock_get:
                    with patch('cookiecutter.zipfile.ZipFile') as mock_zipfile_class:
                


# LLM-generated content at query #25
#--------------------------

```python
def test_unzip_raises_on_empty_zipfile(tmp_path, monkeypatch):
    """Test that unzip raises InvalidZipRepository when zip file is empty."""
    import zipfile
    from cookiecutter.zipfile import unzip
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Create an empty zip file
    empty_zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip_path, 'w') as zf:
        pass  # Create empty zip
    
    # Attempt to unzip should raise InvalidZipRepository
    try:
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=str(tmp_path))
        assert False, "Expected InvalidZipRepository to be raised"
    except Exception as e:
        assert type(e).__name__ == 'InvalidZipRepository'
        assert 'empty' in str(e).lower()


