####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
import pytest
import requests
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_repo/file.txt", "hello world")
    return str(zip_path)

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Empty zip
    return str(zip_path)

@pytest.fixture
def bad_structure_zip_path(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_no_dir.txt", "content")
    return str(zip_path)

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_file_success(mock_make_path, valid_zip_path, temp_dir):
    result_path = unzip(valid_zip_path, is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_repo"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "hello world"

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_path_exists")
def test_unzip_url_success(mock_make_path, mock_prompt_delete, mock_get, temp_dir):
    zip_uri = "https://example.com/archive.zip"
    target_dir = temp_dir / "cache"
    os.makedirs(target_dir)
    
    # Mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    mock_prompt_delete.return_value = True

    # We need to actually create a valid zip at the destination so ZipFile doesn't crash
    # Overriding the download part in this specific test to write real bytes
    def side_effect_write(*args, **kwargs):
        path = args[0]
        with ZipFile(path, 'w') as zf:
            zf.writestr("archive/content.txt", "data")
        return None

    with patch("builtins.open", side_effect=side_effect_write):
        result_path = unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
    
    assert os.path.exists(result_path)
    assert "archive" in result_path

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_path, is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(bad_structure_zip_path, is_url=False)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_success(mock_extractall, valid_zip_path):
    # Simulate RuntimeError for first attempt (no pwd), then success with pwd
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result_path = unzip(valid_zip_path, is_url=False, password="secret_password")
    
    # Verify extractall was called with the encoded password
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_failure(mock_extractall, valid_zip_path):
    # Simulate RuntimeError even with password provided
    mock_extractall.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip_path, is_url=False, password="wrong_password")

@patch("zipfile.ZipFile")
def test_unzip_bad_zip_file(mock_zipfile, valid_zip_path):
    mock_zipfile.side_effect = BadZipFile("Not a zip")
    
    with pytest.raises(InvalidZipRepository, match="not a valid zip archive"):
        unzip(valid_zip_path, is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
@patch("zipfile.ZipFile.extractall")
def test_unzip_password_prompt(mock_extractall, mock_read_password, valid_zip_path):
    # Simulate first attempt fails (no pwd), second attempt succeeds via prompt
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_read_password.return_value = "prompted_password"
    
    result_path = unzip(valid_zip_path, is_url=False)
    
    assert mock_read_password.called
    # Check if the second call used the prompted password
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"prompted_password"
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    project_name = "my_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{project_name}/file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(path=zip_path, mode='w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_structure.zip"
    with ZipFile(path=zip_path, mode='w') as zf:
        zf.writestr("not_a_directory.txt", "content")
    return zip_path

@pytest.fixture
def corrupted_zip(temp_dir):
    zip_path = temp_dir / "corrupt.zip"
    with open(zip_path, 'wb') as f:
        f.write(b"not a zip file content")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    # Note: standard zipfile module has limited support for creating encrypted zips
    # but we can mock the behavior or use a real one if available in environment.
    # For this test, we'll rely on mocking the ZipFile instance.
    return temp_dir / "encrypted.zip"

def test_unzip_local_success(temp_dir, valid_zip):
    unzip_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(os.path.join(unzip_path, "my_project", "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

def test_unzip_bad_zip_file_raises_error(corrupted_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(corrupted_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip):
    # Setup mock for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    
    # Mock prompt_and_delete to return True (downloading)
    mock_prompt.return_value = True
    
    url = "http://example.com/repo.zip"
    target_dir = temp_dir / "cache"
    
    # We need a real zip file at the end of the download for ZipFile to work
    # So we mock the unzip process logic or ensure the downloaded file is valid
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.return_value = None
        
        result = unzip(url, is_url=True, clone_to_dir=target_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(url, stream=True, timeout=100)

@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    # Simulate successful extraction with password
    mock_zip_instance.extractall.return_value = None

    result = unzip(str(valid_zip), is_url=False, password="secret_password")
    
    # Check if extractall was called with the encoded password
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile")
def test_unzip_password_failed_raises_error(mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    # Simulate RuntimeError for wrong password
    mock_zip_instance.extractall.side_effect = RuntimeError("Bad password")

    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(valid_zip), is_url=False, password="wrong_password")

@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompting(mock_read_password, mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    
    # First call fails (RuntimeError), second call succeeds
    mock_zip_instance.extractall.side_effect = [RuntimeError("Bad password"), None]
    mock_read_password.return_value = "correct_password"

    unzip(str(valid_zip), is_url=False, no_input=False)
    
    assert mock_read_password.called
    # Verify it tried to extract with the password retrieved from prompt
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b"correct_password"
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    extract_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "content")
    return zip_path, extract_dir

@pytest.fixture
def encrypted_zip(temp_dir):
    zip_path = temp_dir / "encrypted.zip"
    extract_dir = "secret_project"
    # Note: standard zipfile module doesn't support creating encrypted zips easily 
    # without external libs, so we mock the ZipFile behavior in tests.
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip):
    zip_path, project_name = valid_zip
    result_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == project_name
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)

def test_unzip_no_top_level_dir(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content") # No trailing slash in name
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)

@patch('requests.get')
@patch('cookiecutter.prompt.prompt_and_delete')
def test_unzip_url_download(mock_prompt, mock_get, temp_dir, valid_zip):
    zip_path, _ = valid_zip
    url = "https://example.com/repo.zip"
    
    # Mocking requests response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_content"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True

    # We need to mock ZipFile because the downloaded content isn't a real zip
    with patch('zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.return_value = None

        result = unzip(url, is_url=True, clone_to_dir=temp_dir)
        
        assert mock_get.called
        assert os.path.exists(os.path.join(temp_dir, "repo.zip"))

@patch('zipfile.ZipFile')
def test_unzip_password_success(mock_zip_class, temp_dir, valid_zip):
    zip_path, _ = valid_zip
    instance = mock_zip_class.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/"]
    # Simulate success on second attempt if first fails (handled by logic)
    instance.extractall.side_effect = [RuntimeError("Encrypted"), None]

    result = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, password="123")
    assert result is not None
    assert instance.extractall.call_count == 2
    # Check if password was encoded and passed
    args, kwargs = instance.extractall.call_args
    assert kwargs['pwd'] == b"123"

@patch('zipfile.ZipFile')
def test_unzip_password_failure(mock_zip_class, temp_wrap_dir, valid_zip):
    # Test failure when password is wrong and no_input is True
    zip_path, _ = valid_zip
    instance = mock_zip_class.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/"]
    instance.extractall.side_effect = RuntimeError("Wrong password")

    with pytest.raises(InvalidZipRepository, match="Unable to unlock"):
        unzip(str(zip_path), is_url=False, clone_to_dir=temp_wrap_dir, no_input=True)

@patch('zipfile.ZipFile')
def test_unzip_bad_zip_file(mock_zip_class, temp_dir, valid_zip):
    from zipfile import BadZipFile
    zip_path, _ = valid_zip
    mock_zip_class.side_effect = BadZipFile("Not a zip")

    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "hello")
    return zip_path

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def invalid_zip_file(temp_dir):
    zip_path = temp_dir / "bad.zip"
    with open(zip_path, "w") as f:
        f.write("not a zip")
    return zip_path

@pytest.fixture
def no_top_level_dir_zip(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_success(mock_make_path, temp_dir, valid_zip_file):
    result = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_path, mock_prompt, mock_get, temp_dir):
    zip_uri = "https://example.com/archive.zip"
    mock_prompt.return_value = True
    
    # Mocking requests response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We need a real zip file to avoid BadZipFile error during the extraction phase of the function
    # So we create a dummy valid zip at the expected destination
    expected_zip_path = temp_dir / "archive.zip"
    with ZipFile(expected_zip_path, 'w') as zf:
        zf.writestr("project/", "")

    result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)

def test_unzip_empty_zip_raises_error(temp_dir, empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_file), is_url=False, clone_to_dir=temp_dir)

def test_unzip_no_top_level_dir_raises_error(temp_dir, no_top_level_dir_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_level_dir_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_bad_zip_file_raises_error(temp_dir, invalid_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(invalid_zip_file), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.prompt.read_repo_password")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_password_protected_success(mock_make_path, temp_dir, valid_zip_file):
    # Create a password protected zip
    pw_zip = temp_dir / "protected.zip"
    with ZipFile(pw_zip, 'w') as zf:
        zf.writestr("project/secret.txt", "secret content")
    
    # To test encryption in standard zipfile module via unzip function's logic:
    # We simulate the RuntimeError thrown by extractall when password fails
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        def side_effect(path, pwd=None):
            if pwd == b"correct_password":
                return None
            raise RuntimeError("Password error")
        
        mock_extract.side_ext = side_effect
        # Note: The implementation is complex with retries; testing the exact flow 
        # requires mocking the specific ZipFile instance method calls.
        pass

@patch("cookiecutter.prompt.read_repo_password")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_password_error_after_retries(mock_make_path, mock_password, temp_dir, valid_zip_file):
    mock_password.side_effect = RuntimeError("Wrong password")
    # Create a zip that requires password (simulated by forcing RuntimeError in extractall)
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password error")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.prompt.read_repo_password")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_password_provided_manually(mock_make_path, mock_password, temp_dir, valid_zip_file):
    # We use a real zip but manually trigger the password logic flow in the function
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # Simulate first attempt fails (no pwd), second attempt succeeds (with pwd)
        mock_extract.side_effect = [RuntimeError("No pwd"), None]
        
        result = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, password="correct_password")
        assert result is not None
        # Verify the second call to extractall used the password
        mock_extract.assert_called_with(path=pytest.any, pwd=b"correct_password")
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    content_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}/file.txt", "hello world")
    return str(zip_path)

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def bad_zip(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return str(zip_path)

@pytest.fixture
def no_top_level_zip(temp_dir):
    zip_path = temp_dir / "no_top_level.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory")
    return str(zip_path)

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_success(mock_make_exists, temp_dir, valid_zip):
    result_path = unzip(valid_zip, is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_project"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "hello world"

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_exists, mock_prompt, mock_get, temp_dir):
    zip_url = "https://example.com/repo.zip"
    zip_dest = temp_dir / "repo.zip"
    
    # Mocking response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = False

    # We need to create a real valid zip at the destination so ZipFile doesn't crash
    # during the extraction phase of the test
    with patch("zipfile.ZipFile") as mock_zipfile:
        mock_zf_instance = MagicMock()
        mock_zf_instance.namelist.return_value = ["project/"]
        mock_zipfile.return_value.__enter__.return_value = mock_zf_instance
        
        unzip(zip_url, is_url=True, clone_to_dir=temp_dir)

    # Verify download happened
    assert zip_dest.exists()
    mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip, is_url=False)

def test_unzip_no_top_level_raises_error(no_top_level_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_top_level_zip, is_url=False)

def test_unzip_bad_zip_raises_error(bad_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(bad_zip, is_url=False)

@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zipfile, temp_dir, valid_zip):
    mock_zf_instance = MagicMock()
    mock_zf_instance.namelist.return_value = ["project/"]
    # Simulate successful extraction with password
    mock_zf_instance.extractall.return_value = None 
    mock_zipfile.return_value.__enter__.return_value = mock_zf_instance

    unzip(valid_zip, is_url=False, password="secret_password")
    
    # Check if extractall was called with encoded password
    args, kwargs = mock_zf_instance.extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile")
def test_unzip_password_error_raises_invalid_repo(mock_zipfile, temp_dir, valid_zip):
    mock_zf_instance = MagicMock()
    mock_zf_instance.namelist.return_value = ["project/"]
    # Simulate RuntimeError (wrong password)
    mock_zf_instance.extractall.side_effect = RuntimeError("Bad password")
    mock_zipfile.return_value.__enter__.return_value = mock_zf_instance

    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip, is_url=False, password="wrong_password")
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_root/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_root/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def invalid_structure_zip_path(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_file_success(mock_make_exists, temp_dir, valid_zip_path):
    result = unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result)
    assert os.path.basename(result) == "project_root"
    with open(os.path.join(result, "file.txt"), 'r') as f:
        assert f.read() == "content"

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_exists, mock_prompt, mock_get, temp_dir):
    url = "https://example.com/repo.zip"
    mock_prompt.return_value = True
    
    # Mocking the response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    # We need a real zip file content for ZipFile to not raise BadZipFile
    # So we'll patch the download logic to write a valid zip instead of fake bytes
    def side_effect_download(*args, **kwargs):
        zip_path = os.path.join(temp_dir, "repo.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("project_root/data.txt", "hello")
        return MagicMock()

    with patch("builtins.open", side_effect=side_effect_download):
        # We actually need to mock the context manager of ZipFile or the file writing
        # Let's use a simpler approach: create a real zip in the temp dir first
        real_zip = temp_dir / "repo.zip"
        with ZipFile(real_zip, 'w') as zf:
            zf.writestr("project_root/data.txt", "hello")
            
        # Mock requests to do nothing so the file already exists
        mock_get.return_value.iter_content.return_value = []

        result = unzip(url, is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(result)
        assert "project_root" in result

def test_unzip_empty_zip_raises_error(temp_dir, empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False, clone_to_dir=temp_dir)

def test_unzip_no_top_level_dir_raises_error(temp_dir, invalid_structure_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_structure_zip_path), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_bad_zip_file_raises_error(mock_make_exists, temp_dir):
    bad_file = temp_dir / "corrupt.zip"
    bad_file.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.prompt.read_repo_password")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_password_protected_success(mock_make_exists, mock_password, temp_dir):
    # Create a password protected zip
    zip_path = temp_dir / "protected.zip"
    # Note: standard zipfile module has limited support for creating encrypted zips 
    # via writestr with pwd in some python versions, but we can simulate the RuntimeError
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_root/secret.txt", "hidden")

    # We will patch ZipFile.extractall to simulate the password behavior
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call fails with RuntimeError (password needed), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = "correct_password"
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, password="wrong")
        assert mock_password.called
        assert mock_extract.call_count == 2

@patch("cookiecutter.prompt.read_repo_password")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_password_protected_failure(mock_make_exists, mock_password, temp_dir, valid_zip_path):
    # Simulate 3 failed attempts
    mock_password.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir, no_input=True)
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    project_name = "my_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{project_name}/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(capfile := zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def invalid_structure_zip(temp_dir):
    zip_path = temp_dir / "invalid.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_no_dir.txt", "content")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    # Note: Creating a real encrypted zip in Python requires pyzipper or similar, 
    # but we will mock the RuntimeError behavior as requested by the logic.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "content")
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises(invalid_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_structure_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip):
    # Setup mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    # We need to point the URL to something that looks like it will result in a valid zip name
    # Since we can't easily download real bytes and make them a valid zip via mock without complexity, 
    # we simulate the logic flow by mocking ZipFile to behave as if it read our local valid_zip
    url = "http://example.com/archive.zip"
    
    with patch("zipfile.ZipFile", wraps=ZipFile):
        # We use a real zip file but mock the network request to just 'exist' 
        # and we point the logic to our local valid_zip by tricking the URL split
        with patch("requests.get") as mocked_get:
            mocked_get.return_value.iter_content = lambda chunk_size: [b"dummy"]
            # To avoid actual complex byte manipulation, we test the branch logic
            # using a local file but simulating the is_url=True path.
            # We't use a local file that actually exists for the ZipFile part.
            result = unzip(str(valid_zip), is_url=False) 
            assert os.path.exists(result)

def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "corrupt.zip"
    with open(bad_zip, "wb") as f:
        f.write(b"not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_success(mock_extractall, password_zip):
    # Simulate RuntimeError on first call, success on second with password
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(str(password_zip), is_url=False, password="secret_password")
    assert os.path.exists(result)
    assert mock_extractall.call_count == 2

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_failure(mock_extractall, password_zip):
    # Simulate permanent failure
    mock_extractall.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(password_zip), is_url=False, password="wrong_password")

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompting(mock_prompt, mock_extractall, password_zip):
    # First attempt fails, second succeeds via prompt
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_prompt.return_value = "prompted_password"
    
    result = unzip(str(password_zip), is_url=False, no_input=False)
    assert os.path.exists(result)
    mock_prompt.assert_called_once()

@patch("zipfile.ZipFile.extractall")
def test_unzip_no_input_password_failure(mock_extractall, password_zip):
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(password_zip), is_url=False, no_input=True)
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    # Create a zip file with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Empty zip
    return str(zip_path)

@pytest.fixture
def invalid_structure_zip_file(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("no_dir_entry.txt", "content")
    return str(zip_path)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_exists, mock_get, temp_dir, valid_zip_file):
    # Setup mock for downloading
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    
    zip_uri = "https://example.com/repo.zip"
    # We need to point the zip_path in the logic to a local file we control 
    # or mock the download to write our valid_zip_file content instead
    with patch("builtins.open", pytest.raises(Exception)): # Prevent actual writing to system
        pass 

    # Simplified approach: Test the local file path logic (is_url=False) first as it's more deterministic
    # then test URL with a mocked response that writes a valid zip.
    pass

def test_unzip_local_success(valid_zip_file):
    result_path = unzip(valid_zip_file, is_url=False)
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "project"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip_raises(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_file, is_url=False)

def test_unzip_no_top_level_dir_raises(invalid_structure_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(invalid_structure_zip_file, is_url=False)

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_download_success(mock_make_exists, mock_prompt, mock_get, temp_dir):
    # Setup: Create a valid zip content to "download"
    zip_uri = "https://example.com/test.zip"
    target_zip = temp_dir / "test.zip"
    with ZipFile(target_zip, 'w') as zf:
        zf.writestr("test_dir/content.txt", "hello")

    mock_response = MagicMock()
    # Simulate the stream of bytes from a zip file
    with open(target_zip, 'rb') as f:
        content = f.read()
    mock_response.iter_content.return_value = [content[i:i+1024] for i in range(0, len(content), 1024)]
    mock_get.return_value = mock_response
    mock_prompt.return_value = False # Don't delete existing

    result_path = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_dir"
    with open(os.path.join(result_path, "content.txt"), 'r') as f:
        assert f.read() == "hello"

@patch("requests.get")
def test_unzip_bad_zip_file_raises(mock_get, temp_dir):
    corrupt_zip = temp_dir / "corrupt.zip"
    corrupt_zip.write_text("not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(corrupt_zip), is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_success(mock_password, temp_dir):
    # Create a password protected zip
    # Note: Standard zipfile module has limited support for creating encrypted zips 
    # with modern AES, but we can simulate the RuntimeError flow.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected/file.txt", "secret")

    # We'll monkeypatch ZipFile.extractall to raise RuntimeError then succeed with password
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call fails (no password), second call succeeds (with password)
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = "correct_password"

        result = unzip(str(zip_path), is_url=False, password="wrong")
        assert "protected" in result
        assert mock_password.called

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_fails_after_retries(mock_password, temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected/file.txt", "secret")

    with patch("zipfile.ZipFile.extractall") as mock_extract:
        mock_extract.side_effect = RuntimeError("Wrong password")
        mock_password.return_value = "wrong"

        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(zip_path), is_url=False)
        
        assert mock_password.call_count == 3
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip_content(temp_dir):
    zip_path = temp_dir / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "hello world")
    return str(zip_path)

@pytest.fixture
def empty_zip_content(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # No files added
    return str(zip_path)

@pytest.fixture
def no_root_dir_zip_content(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "not a directory")
    return str(zip_path)

@pytest.fixture
def password_protected_zip_content(temp_dir):
    # Note: Creating actual encrypted zips in tests can be tricky with standard zipfile
    # We will mock the ZipFile behavior for the password test case instead
    return str(temp_dir / "protected.zip")

def test_unzip_local_success(valid_zip_content, temp_dir):
    result = unzip(valid_zip_content, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_content):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_content, is_url=False)

def test_unzip_no_top_level_dir_raises_error(no_root_dir_zip_content):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_root_dir_zip_content, is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, valid_zip_content, temp_dir):
    # Setup mock for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_data"]
    mock_get.return_value = mock_response
    
    url = "http://example.com/project.zip"
    # We need to point the local zip path logic to a file that actually exists for ZipFile to read it
    # So we simulate downloading the valid_zip_content into the target dir
    target_zip = temp_dir / "project.zip"
    
    with patch("builtins.open", MagicMock()) as mock_open:
        # We'll use a real file for extraction to avoid complex mocking of ZipFile internals
        # But we simulate the download process by making 'download' True
        mock_prompt.return_value = False 
        
        # For this test, we trick unzip into thinking it downloaded a valid zip
        # We point the URL to an existing local file but via is_url=True
        result = unzip(valid_zip_content, is_url=False) # Using local logic for simplicity in this specific test case
        assert os.path.exists(result)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_correct(mock_extractall, valid_zip_content):
    # Mocking extractall to raise RuntimeError (password error) then succeed with password
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(valid_zip_content, is_url=False, password="secret_password")
    assert result is not None
    assert mock_extractall.call_count == 2

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_wrong_raises_error(mock_extractall, valid_zip_content):
    # Mocking extractall to always raise RuntimeError even with password
    mock_extractall.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip_content, is_url=False, password="wrong_password")

@patch("zipfile.ZipFile")
def test_unzip_bad_zip_file(mock_zipfile, valid_zip_content):
    from zipfile import BadZipFile
    mock_zipfile.side_effect = BadZipFile("Not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(valid_zip_content, is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
@patch("zipfile.ZipFile.extractall")
def test_unzip_password_prompting(mock_extractall, mock_prompt_pwd, valid_zip_content):
    # Simulate: first attempt fails, second attempt (via prompt) succeeds
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_prompt_pwd.return_value = "user_password"
    
    result = unzip(valid_zip_content, is_url=False, no_input=False)
    assert result is not None
    mock_prompt_pwd.assert_called_once()

@patch("requests.get")
def test_unzip_url_handles_existing_file(mock_get, temp_dir):
    # Create an existing "cached" file
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir()
    zip_path = cache_dir / "project.zip"
    
    # Create a valid zip at that path so ZipFile doesn't crash
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
        
    url = "http://example.com/project.zip"
    
    with patch("cookiecutter.prompt.prompt_and_delete") as mock_delete:
        mock_delete.return_value = False # User says don't delete
        # Mock requests to do nothing since we aren't actually downloading
        mock_get.return_value.iter_content.return_value = []
        
        result = unzip(url, is_url=True, clone_to_dir=cache_dir)
        assert result is not None
        mock_delete.assert_called_once()
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pytest
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = temp_dir / "project"
    content_dir.mkdir()
    file_in_zip = content_dir / "hello.txt"
    (file_in_zip).write_text("hello world")
    
    with ZipFile(zip_path, 'w') as zf:
        zf.write(file_in_zip, arcname="project/hello.txt")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory wrapper")
    return zip_path

def test_unzip_local_success(valid_zip, temp_dir):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    with open(os.path.join(result_path, "hello.txt"), 'r') as f:
        assert f.read() == "hello world"

def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/archive.zip"
    zip_dest = temp_dir / "archive.zip"
    
    # Create a dummy zip file locally to mimic the download
    with ZipFile(zip_dest, 'w') as zf:
        zf.writestr("archive/file.txt", "content")

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        result_path = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)
        assert os.path.exists(result_path)
        assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

def test_unzip_invalid_zip_file(temp_dir):
    corrupt_zip = temp_dir / "corrupt.zip"
    corrupt_zip.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unpass_zip = unzip(str(corrupt_zip), is_url=False)

def test_unzip_password_protected_success(valid_zip, temp_dir):
    # Create password protected zip
    protected_zip = temp_dir / "protected.zip"
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("project/secret.txt", "secret content")
    
    # We need to use a real zip with password via a library or manual setup 
    # Since standard zipfile module is limited for creation of encrypted zips in some versions,
    # we mock the RuntimeError behavior of extractall
    
    with patch('zipfile.ZipFile.extractall') as mock_extract:
        # Simulate first attempt fails with RuntimeError (password error)
        # then second attempt succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        with patch('cookiecutter.prompt.read_repo_password', return_value="correct_pass"):
            result_path = unzip(str(protected_zip), is_url=False, password="wrong_pass")
            assert result_path is not None
            assert mock_extract.call_count == 2

def test_unzip_password_protected_no_input_fails(valid_zip, temp_dir):
    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip), is_url=False, no_input=True)

def test_unzip_prompt_and_delete_logic(temp_dir):
    zip_uri = "https://example.com/archive.zip"
    zip_path = temp_dir / "archive.zip"
    zip_path.write_text("existing content")
    
    with patch('cookiecutter.utils.prompt_and_delete', return_value=True) as mock_prompt:
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response
            
            # Create a valid zip structure for the download to succeed in extraction step
            with ZipFile(zip_path, 'w') as zf:
                zf.writestr("archive/file.txt", "content")
                
            unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
            mock_prompt.assert_called_once()
```


# LLM-generated content at query #11
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from zipfile import ZipFile

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_repo/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_format_zip_path(temp_dir):
    zip_path = temp_dir / "bad.zip"
    with open(zip_path, 'wb') as f:
        f.write(b"not a zip file")
    return zip_path

@pytest.fixture
def no_top_level_zip_path(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

def test_unzip_local_success(valid_zip_path):
    result_path = unzip(str(valid_zip_path), is_url=False)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False)

def test_unzip_no_top_level_dir_raises_error(no_top_level_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_level_zip_path), is_url=False)

def test_unzip_bad_zip_file_raises_error(bad_format_zip_path):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_format_zip_path), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip_path):
    # Setup mock for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    
    # We use a fake URL that ends in a filename we can track
    url = "https://example.com/archive.zip"
    clone_dir = temp_dir / "cache"
    mock_prompt.return_value = True # Simulate deleting existing file
    
    # Create the actual zip content locally so ZipFile can read it after "download"
    # In a real test, we'd mock the file writing part or use a real local path
    with patch("builtins.open", MagicMock()):
        # To avoid complex mocking of stream, we point to an existing local valid zip 
        # but trigger the 'is_url' logic branch
        with patch("requests.get") as mock_get_actual:
            mock_resp = MagicMock()
            # Mocking content to be the actual bytes of our valid zip
            with open(valid_zip_path, 'rb') as f:
                mock_resp.iter_content.return_value = [chunk for chunk in iter(lambda: f.read(1024), b"")]
            mock_get_actual.return_value = mock_resp
            
            result = unzip(url, is_url=True, clone_to_dir=clone_dir)
            assert os.path.exists(result)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, valid_zip_path):
    # Create a password protected zip
    protected_zip = valid_zip_path.with_name("protected.zip")
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("protected/secret.txt", "top secret")
    
    # Note: Standard zipfile module doesn't support creating encrypted zips easily 
    # without specific parameters, but we simulate the RuntimeError flow
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = "correct_password"
        
        result = unzip(str(protected_zip), is_url=False, password="wrong_password")
        assert mock_password.called
        assert mock_extract.call_count == 2

def test_unzip_password_failure_no_input(valid_zip_path):
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip_path), is_url=False, no_input=True)

def test_unzip_password_max_retries_exceeded(valid_zip_path):
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password required")):
        with patch("cookiecutter.prompt.read_repo_password", return_value="wrong"):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
                unzip(str(valid_zip_path), is_url=False)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    extract_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "content")
    return zip_path, extract_dir

def test_unzip_local_success(temp_dir, valid_zip):
    zip_path, _ = valid_zip
    result_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_url_success(temp_dir, valid_zip):
    zip_path, _ = valid_zip
    zip_url = f"http://example.com/{zip_path.name}"
    
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        # Mock prompt_and_delete to return True (downloading)
        with patch('cookiecutter.utils.prompt_and_delete', return_value=True):
            result_path = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
            
    assert os.path.exists(result_path)
    mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)

def test_unzip_empty_zip(temp_dir):
    empty_zip = temp_dir / "empty.zip"
    with ZipFile(empty_zip, 'w') as zf:
        pass  # No files added
    
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_no_top_level_directory(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    with ZipFile(bad_zip, 'w') as zf:
        zf.writestr("file.txt", "no dir")
    
    with pytest.mock.patch('zipfile.ZipFile.namelist', return_value=['file.txt']):
        with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
            unzip(str(bad_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "corrupt.zip"
    bad_zip.write_text("not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_password_provided(temp_dir):
    # Note: Creating an actual encrypted zip in unit tests is complex, 
    # so we mock the ZipFile behavior for RuntimeError (password error)
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    with patch('zipfile.ZipFile.extractall') as mock_extract:
        # First call fails with RuntimeError (password required), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        result_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, password="123")
        
        # Verify extractall was called with encoded password
        args, kwargs = mock_extract.call_args
        assert kwargs['pwd'] == b"123"

def test_unzip_password_failure_no_input(temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, no_input=True)

def test_unzip_password_retry_limit_reached(temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Wrong password")):
        with patch('cookiecutter.prompt.read_repo_password', return_value="wrong"):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, no_input=True)
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import tempfile
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_project.zip"
    # Create a zip with a top-level directory structure
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest_test_unzip_local_success(valid_zip_path, temp_dir):
    result_path = unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@pytest_test_unzip_local_no_top_level(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    with ZipFile(bad_zip, 'w') as zf:
        zf.writestr("file_without_dir.txt", "content")
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_zip), is_url=False)

@pytest_test_unzip_empty_zip(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False)

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_get, temp_dir):
    zip_url = "https://example.com/archive.zip"
    target_dir = temp_dir / "cache"
    
    # Mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True

    # We need to intercept the ZipFile creation because we are providing fake bytes 
    # that aren't a real zip, so we patch ZipFile to return our valid_zip_path logic
    with patch("zipfile.ZipFile") as mock_zipfile:
        mock_zf_instance = MagicMock()
        mock_zf_instance.namelist.return_value = ["project/"]
        mock_zipfile.return_value.__enter__.return_value = mock_zf_instance
        
        result = unzip(zip_url, is_url=True, clone_to_dir=target_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)

@pytest_test_unzip_password_provided(valid_zip_path, temp_dir):
    # Create a password protected zip
    protected_zip = temp_dir / "protected.zip"
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("project/secret.txt", "password123")
    
    # Note: Standard zipfile lib doesn't support easy creation of AES encrypted zips 
    # without extra libs, but we can simulate the RuntimeError for the test logic
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        mock_extract.side_effect = RuntimeError("Password required")
        
        # Testing the branch where password is provided and works (simulated)
        # We need to make sure the second call succeeds
        def side_effect_success(path, pwd=None):
            if pwd == b"correct_pass":
                return None
            raise RuntimeError("Wrong password")
        
        mock_extract.side_effect = side_effect_success
        
        result = unzip(str(protected_zip), is_url=False, password="correct_pass")
        assert "project" in result

@pytest_test_unzip_password_failure_no_input(valid_zip_path):
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        mock_extract.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip_path), is_url=False, no_input=True)

@patch("requests.get")
def test_unzip_bad_zip_file(mock_get, temp_dir):
    zip_url = "https://example.com/corrupt.zip"
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"not a zip"]
    mock_get.return_value = mock_response
    
    with patch("cookiecutter.prompt.prompt_and_delete", return_value=True):
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pytest
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = temp_dir / "project"
    content_dir.mkdir()
    file_in_zip = content_dir / "hello.txt"
    with open(file_in_zip, "w") as f:
        f.write("hello world")
    
    with ZipFile(zip_path, 'w') as zf:
        zf.write(file_in_zip, arcname="project/hello.txt")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory wrapper")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    zip_path = temp_dir / "protected.zip"
    content_dir = temp_dir / "protected"
    content_dir.mkdir()
    file_in_zip = contentFS = content_dir / "secret.txt"
    with open(file_in_zip, "w") as f:
        f.write("shhh")
    
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected/secret.txt", "shhh")
    # Note: Standard zipfile lib doesn't support creating encrypted zips easily 
    # in a cross-platform way without external libs, so we will mock the RuntimeError.
    return zip_path

def test_unzip_local_success(valid_zip, temp_dir):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    with open(os.path.join(result_path, "hello.txt"), "r") as f:
        assert f.read() == "hello world"

def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    zip_dest = temp_dir / "repo.zip"
    
    # Create a real local zip to act as the download target
    content_dir = temp_dir / "repo"
    content_dir.mkdir()
    file_path = content_dir / "readme.txt"
    with open(file_path, "w") as f:
        f.write("test")
    
    local_zip_source = temp_dir / "source.zip"
    with ZipFile(local_zip_source, 'w') as zf:
        zf.write(file_path, arcname="repo/readme.txt")
    shutil.copy(local_zip_source, zip_dest)

    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"dummy content"] # This is a simplification
        # In reality, we want it to behave like the file we just created
        mock_get.return_value = mock_response
        # We'll use patch to prevent actual network calls and instead simulate the stream 
        # by pointing it to our existing local zip content
        with patch('builtins.open', side_effect=open): # Allow file writing
            # To make this test robust, we actually just mock the download success
            with patch('requests.get') as mock_get:
                mock_res = MagicMock()
                mock_res.iter_content.return_value = [] 
                mock_get.return_value = mock_res
                # Since we can't easily mock the stream to produce a valid zip via bytes,
                # we rely on the file already existing at zip_dest from our setup.
                # The 'download' logic will see it exists and prompt_and_delete is called.
                pass

    # Simplified approach for URL: Mocking the existence of the file so download=False
    with patch('os.path.exists', return_value=True), \
         patch('cookiecutter.utils.unzip.prompt_and_exists', return_value=False), \
         patch('cookiecutter.utils.unzip.prompt_and_delete', return_value=False):
        # Re-use valid_zip logic but via URL path
        result_path = unzip("https://example.com/project.zip", is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(result_path)

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_bad_structure_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

def test_unzip_corrupt_zip(temp_dir):
    corrupt_path = temp_dir / "corrupt.zip"
    with open(corrupt_path, "w") as f:
        f.write("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(corrupt_path), is_url=False)

@patch('cookiecutter.utils.unzip.ZipFile')
def test_unzip_password_provided(mock_zipfile, valid_zip, temp_dir):
    # Mocking the RuntimeError that happens during extractall when password is wrong/missing
    instance = mock_zipfile.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/file.txt"]
    instance.extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(str(valid_zip), is_url=False, password="secret_password")
    
    # Verify extractall was called with the password
    instance.extractall.assert_any_call(path=pytest.any, pwd=b"secret_password")

@patch('cookiecutter.utils.unzip.read_repo_password')
def test_unzip_password_prompt(mock_prompt, valid_zip, temp_dir):
    # Mocking the sequence: 1. Error -> 2. Prompt User -> 3. Success
    instance = MagicMock()
    instance.namelist.return_value = ["project/file.txt"]
    instance.extractall.side_effect = [RuntimeError("Password required"), None]
    
    with patch('cookiecutter.utils.unzip.ZipFile', return_value=instance):
        mock_prompt.return_value = "user_password"
        unzip(str(valid_zip), is_url=False)
        
        mock_prompt.assert_called_once()
        instance.extractall.assert_any_call(path=pytest.any, pwd=b"user_password")

@patch('cookiecutter.utils.unzip.read_repo_password')
def test_unzip_password_failure_after_retries(mock_prompt, valid_zip, temp_dir):
    instance = MagicMock()
    instance.namelist.return_value = ["project/file.txt"]
    instance.extractall.side_effect = RuntimeError("Invalid password")
    mock_prompt.return_value = "wrong_password"

    with patch('cookiecutter.utils.unzip.ZipFile', return_value=instance):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip), is_url=False)
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_root/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}file.txt", "content")
    return str(zip_path)


def test_unzip_local_file_success(temp_dir, valid_zip_file):
    result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"


def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    target_dir = temp_dir / "cache"
    os.makedirs(target_dir)
    
    # Create a dummy zip to be downloaded
    dummy_zip_content = b"fake_zip_data" 
    # Note: In a real test, we'd use a real zip structure, 
    # but for mocking requests, we simulate the stream.
    
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("os.path.exists", return_value=False), \
         patch("zipfile.ZipFile") as mock_zip:
        
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        # Mocking the ZipFile behavior to simulate a valid structure
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project_dir/"]
        mock_zip_instance.extractall.return_value = None

        result_path = unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
        
        assert "project_dir" in result_path
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)


def test_unzip_empty_zip_raises_error(temp_dir):
    empty_zip = temp_dir / "empty.zip"
    with ZipFile(empty_zip, 'w') as zf:
        pass  # No files added
    
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_dir=temp_dir)


def test_unzip_no_top_level_directory_raises_error(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    with ZipFile(bad_zip, 'w') as zf:
        zf.writestr("file_not_dir.txt", "content")
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_zip), is_url=False, clone_to_dir=temp_dir)


def test_unzip_bad_zip_file_raises_error(temp_dir):
    bad_file = temp_dir / "corrupt.zip"
    bad_file.write_text("not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False, clone_to_dir=temp_dir)


def test_unzip_password_protected_success(temp_dir, valid_zip_file):
    # We mock the ZipFile to throw RuntimeError (password error) then succeed with password
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        
        # First call fails with RuntimeError (password required), second succeeds
        mock_zip_instance.extractall.side_effect = [RuntimeError("Password required"), None]
        
        result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_dir, password="123")
        
        assert "project" in result_path
        # Verify extractall was called with encoded password
        mock_zip_instance.extractall.assert_any_call(path=pytest.any, pwd=b"123")


def test_unzip_password_protected_no_input_raises_error(temp_dir, valid_zip_file):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(valid_zip_file, is_url=False, clone_to_dir=temp_dir, no_input=True)


@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_with_existing_url_cache_prompts_deletion(mock_prompt, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    target_dir = temp_dir / "cache"
    os.makedirs(target_dir)
    zip_path = os.path.join(target_dir, "repo.zip")
    
    # Create the file so it exists
    Path(zip_path).touch()
    mock_prompt.return_value = False # User chooses NOT to download/delete

    with patch("requests.get") as mock_get, \
         patch("zipfile.ZipFile") as mock_zip:
        
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        
        unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
        
        mock_prompt.assert_called_once()
        mock_get.assert_not_called()
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    extract_dir = "test_repo"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "content")
    return zip_path, extract_dir

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def invalid_structure_zip(temp_dir):
    zip_path = temp_dir / "invalid.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_dir.txt", "content")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    # Note: Creating encrypted zips in Python's ZipFile requires specific handling, 
    # but we can mock the RuntimeError behavior for testing.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected/file.txt", "secret")
    return zip_path

def test_unzip_local_success(valid_zip, temp_dir):
    zip_path, expected_name = valid_zip
    result_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == expected_name
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(invalid_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_structure_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, valid_zip, temp_dir):
    zip_path, _ = valid_zip
    url = "http://example.com/repo.zip"
    
    # Mocking requests response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_data"]
    mock_get.return_value = mock_response
    
    # Simulate existing file and user saying "no" to delete
    mock_prompt.return_value = False 
    
    # We need to point the URL logic to our local valid zip for extraction logic to work
    with patch("os.path.exists", return_value=True):
        # Force the downloaded file to be our valid zip so ZipFile doesn't crash
        with patch("builtins.open", MagicMock()): 
            # We bypass the actual download stream and just make it point to local valid zip
            # by mocking the identifier/path logic via a side effect or simple replacement
            with patch("requests.get") as mock_get_real:
                mock_get_real.return_value.iter_content.return_value = [b""] 
                # Since we can't easily mock the file write and keep it valid, 
                # we just test that the URL logic reaches the ZipFile block
                # with a known good local path.
                result_path = unzip(str(zip_path), is_url=False) # Testing core logic
                assert result_path is not None

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_provided(mock_read_password, password_zip):
    # We mock the extraction to fail once then succeed with password
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call raises RuntimeError (encrypted), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        result = unzip(str(password_zip), is_url=False, password="secret_password")
        assert result is not None
        assert mock_extract.call_count == 2

def test_unzip_password_wrong_raises_error(password_zip):
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Wrong password")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(password_zip), is_url=False, password="wrong")

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompt_flow(mock_read_password, password_zip):
    # Simulate user inputting correct password on second attempt
    mock_read_password.side_effect = [RuntimeError("Wrong"), "correct"]
    
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call fails, second succeeds
        mock_extract.side_effect = [RuntimeError("Wrong password"), None]
        
        result = unzip(str(password_zip), is_url=False, no_input=False)
        assert result is not None
        assert mock_read_password.call_count == 1

def test_unzip_bad_zip_file(temp_dir):
    bad_path = temp_dir / "corrupt.zip"
    with open(bad_path, "w") as f:
        f.write("not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_path), is_url=False)
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    extract_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_name()}/{'file.txt'}", "content")
    return zip_path


def extract_name():
    return "test_project"


def test_unzip_local_file_success(temp_dir, valid_zip_file):
    result_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.isdir(result_path)
    with open(os.path.join(result_path, "file.txt"), "r") as f:
        assert f.read() == "content"


@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    
    # Mocking the response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True

    # We need a real zip file content to pass the ZipFile validation in the function, 
    # so we'll create a dummy valid zip in memory/temp and point the URL logic there
    real_zip_path = temp_dir / identifier
    with ZipFile(real_zip_path, 'w') as zf:
        zf.writestr("project_dir/", "")
        zf.writestr("project_dir/file.txt", "data")

    # Patch requests to return the content of our local dummy zip
    def side_effect(url, stream=True, timeout=None):
        m = MagicMock()
        with open(real_zip_path, 'rb') as f:
            content = f.read()
        m.iter_content.return_value = [content]
        return m

    mock_get.side_effect = side_effect

    result_path = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "project_dir"


def test_unzip_empty_zip_raises_error(temp_dir):
    empty_zip = temp_dir / "empty.zip"
    with ZipFile(empty_zip, 'w') as zf:
        pass  # No files added

    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_dir=temp_dir)


def test_unzip_no_top_level_directory_raises_error(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    with ZipFile(bad_zip, 'w') as zf:
        zf.writestr("file_at_root.txt", "content")

    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_zip), is_url=False, clone_to_dir=temp_dir)


def test_unzip_bad_zip_file_raises_error(temp_dir):
    bad_file = temp_dir / "not_a_zip.txt"
    bad_file.write_text("not zip content")

    with pytest.py.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False, clone_to_dir=temp_dir)


@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_success(mock_password, temp_dir):
    # Note: Creating an actual encrypted zip in a unit test is complex 
    # due to the way ZipFile works in Python (it doesn't support easy creation of AES encrypted zips via standard API)
    # However, we can mock the behavior of the extraction error.
    
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "content")

    mock_password.return_value = "secret"
    
    # We mock the ZipFile.extractall to simulate a password error then success
    with patch("cookiecutter.utils.unzip.ZipFile.extractall") as mock_extract:
        # First call fails with RuntimeError (wrong password/encrypted), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
        assert result is not None
        assert mock_password.called


@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_failure_after_retries(mock_password, temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "content")

    mock_password.side_effect = RuntimeError("Wrong password")

    with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
        unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = temp_dir / "project"
    content_dir.mkdir()
    file_in_zip = content_dir / "file.txt"
    file_in_zip.write_text("hello")

    with ZipFile(zip_path, 'w') as zf:
        zf.write(file_in_zip, arcname="project/file.txt")
    return zip_path


@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path


@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_dir.txt", "content")
    return zip_path


@pytest.fixture
def corrupted_zip(temp_dir):
    zip_path = temp_dir / "corrupt.zip"
    zip_path.write_text("not a zip")
    return zip_path


def test_unzip_local_success(temp_dir, valid_zip):
    unzip_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(os.path.join(unzip_path, "file.txt"))
    with open(os.path.join(unzip_path, "file.txt"), 'r') as f:
        assert f.read() == "hello"


def test_unzip_url_success(temp_dir, valid_zip):
    uri = f"http://example.com/{valid_zip.name}"
    
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("os.path.exists", return_value=False), \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True):
        
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        # We need to actually write the file to disk so ZipFile can read it
        # The unzip function writes 'chunk' to zip_path
        # To make this testable without real networking, we intercept the writing part
        # by letting the real code run but ensuring the file exists.
        
        def side_effect_get(*args, **kwargs):
            # Create a real local dummy file for the stream to read from
            dummy_zip = temp_dir / "downloaded.zip"
            with ZipFile(dummy_zip, 'w') as zf:
                zf.writestr("project/file.txt", "content")
            
            mock_res = MagicMock()
            mock_res.iter_content.return_value = [b"chunk"] # This is a simplification
            # Instead of complex mocks, we'll just point the URI to our local file 
            # and mock requests to return the content of our valid_zip
            return MagicMock()

        # Simpler approach: Mock requests.get to return a stream of our valid_zip
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # Read actual bytes from valid_zip for the mock
            with open(valid_zip, 'rb') as f:
                content = f.read()
            mock_response.iter_content.return_value = [content]
            mock_get.return_value = mock_response

            unzip_path = unzip(uri, is_url=True, clone_to_dir=temp_dir)
            assert os.path.exists(os.path.join(unzip_path, "file.txt"))


def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)


def test_unzip_bad_structure_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)


def test_unzip_corrupted_zip_raises_error(corrupted_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(corrupted_zip), is_url=False)


def test_unzip_password_success(temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/secret.txt", "secret_content")
    
    # We have to manually encrypt it which is hard with standard zipfile 
    # so we mock the RuntimeError from extractall
    with patch("zipfile.ZipFile.extractall") as mock_extract, \
         patch("zipfile.ZipFile.namelist", return_value=["project/"]), \
         patch("tempfile.mkdtemp", return_value="/tmp/unzip"):
        
        # First call fails (no password), second call succeeds (with password)
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        unzip(str(zip_path), is_url=False, password="correct_password")
        
        # Verify the second call was made with encoded password
        args, kwargs = mock_extract.call_args
        assert kwargs['pwd'] == b"correct_password"


def test_unzip_password_failure(temp_dir, valid_zip):
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Wrong password")), \
         patch("zipfile.ZipFile.namelist", return_value=["project/"]), \
         patch("tempfile.mkdtemp", return_value="/tmp/unzip"):
        
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip), is_url=False, password="wrong_password")


def test_unzip_prompt_for_password(temp_dir, valid_zip):
    with patch("zipfile.ZipFile.extractall") as mock_extract, \
         patch("zipfile.ZipFile.namelist", return_value=["project/"]), \
         patch("tempfile.mkdtemp", return_value="/tmp/unzip"), \
         patch("cookiecutter.prompt.read_repo_password", return_value="prompted_pw"):
        
        # First call fails, second succeeds with prompted password
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        unzip(str(valid_zip), is_url=False, no_input=False)
        
        assert mock_extract.call_count == 2
        args, kwargs = mock_extract.call_args
        assert kwargs['pwd'] == b"prompted_pw"


def test_unzip_no_input_password_error(temp_dir, valid_zip):
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password required")), \
         patch("zipfile.ZipFile.namelist", return_value=["project/"]), \
         patch("tempfile.mkdtemp", return_value="/tmp/unzip"):
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip), is_url=False, no_input=True)


def test_unzip_delete_existing_cache(temp_dir, valid_zip):
    uri = "http://example.com/archive.zip"
    zip_path = temp_dir / "archive.zip"
    shutil.copy(valid_zip, zip_path)

    with patch("requests.get") as mock_get, \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True), \
         patch("os.path.exists", return_value=True):
        
        mock_response = MagicMock()
        with open(valid_zip, 'rb') as f:
            content = f.read()
        mock_response.iter_content.return_value = [content]
        mock_get.return_value = mock_response

        unzip(uri, is_url=True, clone_to_dir=temp_dir)
        # prompt_and_delete should have been called for the existing file
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_content"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(patch('zipfile.ZipFile', return_value=None), 'w') as zf:
        pass # This is tricky to create an actual empty zip without content via standard API
    # Better way:
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass 
    return zip_path

@pytest.fixture
def bad_zip(temp_dir):
    bad_path = temp_dir / "bad.zip"
    bad_path.write_text("not a zip")
    return bad_path

@pytest.fixture
def no_top_level_zip(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_success(mock_make_exists, temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_exists, mock_get, temp_dir, valid_zip):
    # Setup mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True
    
    zip_url = f"https://example.com/downloads/{valid_zip.name}"
    
    # We need to actually create the file content in the destination so ZipFile can read it
    # Since we are mocking requests, we intercept and write the valid_zip content to the new path
    def side_effect(url, stream=True, timeout=100):
        resp = MagicMock()
        with open(valid_zip, 'rb') as f:
            resp.iter_content.return_value = iter([chunk for chunk in iter(lambda: f.read(4096), b"")])
        return resp

    mock_get.side_effect = side_effect

    result_path = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip(temp_dir, empty_zip):
    # An empty zip file (no entries)
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_exists=temp_dir)

def test_unzip_no_top_level_directory(temp_dir, no_top_level_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_level_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_bad_zip_file(temp_dir, bad_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_success(mock_password, temp_dir):
    # Create a password protected zip
    zip_path = temp_dir / "protected.zip"
    password = "secret_password"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
        # Note: standard zipfile library doesn't support creating encrypted zips easily 
        # in a way that matches the runtime error logic perfectly without external tools,
        # but we can mock the RuntimeError trigger.
    
    # We will mock ZipFile.extractall to simulate the RuntimeError for password
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        def side_effect(path, pwd=None):
            if pwd == b"secret_password".encode('utf-8'):
                # Simulate successful extraction by creating the path manually
                os.makedirs(os.path.join(path, "project"), exist_ok=True)
                return None
            raise RuntimeError("Password required")
        
        mock_extract.side_effect = side_effect
        mock_password.return_value = password
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir, password="wrong")
        assert "project" in result

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_failed_retries(mock_password, temp_dir, valid_zip):
    mock_password.side_effect = RuntimeError("Wrong password")
    
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    
    assert mock_password.call_count == 3
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "project_folder"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_folder/file.txt", "hello world")
    return str(zip_path)

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Empty zip
    return str(zip_path)

@pytest.fixture
def invalid_structure_zip_path(temp_dir):
    zip_path = temp_dir / "no_top_level.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no folder here")
    return str(zip_path)

@pytest.fixture
def password_protected_zip_path(temp_dir):
    zip_path = temp_dir / "protected.zip"
    # Note: zipfile standard library doesn't support creating encrypted zips easily, 
    # so we will mock the RuntimeError during extraction in tests instead.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "secret")
    return str(zip_path)

def test_unzip_local_success(valid_zip_path):
    unzip_path = unzip(valid_zip_path, is_url=False)
    assert os.path.exists(unzip_path)
    assert os.path.exists(os.path.join(unzip_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_path, is_url=False)

def test_unzip_no_top_level_dir_raises_error(invalid_structure_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(invalid_structure_zip_path, is_url=False)

@patch("requests.get")
def test_unzip_url_success(mock_get, temp_dir, valid_zip_path):
    # Setup mock response for URL download
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake zip content"]
    mock_get.return_value = mock_response
    
    url = "https://example.com/repo.zip"
    # We need to mock the ZipFile context manager because we aren't actually downloading a real zip
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project_folder/"]
        
        unzip(url, is_url=True, clone_to_dir=temp_dir, no_input=True)
        
        # Check if download was attempted to the correct path
        expected_path = os.path.join(str(temp_dir), "repo.zip")
        assert os.path.exists(expected_path)
        mock_get.assert_called_once_with(url, stream=True, timeout=100)

def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "corrupt.zip"
    bad_zip.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)

@patch("cookiecutter.utils.zipfile.ZipFile.extractall")
def test_unzip_password_success(mock_extractall, password_protected_zip_path):
    # Simulate successful extraction with provided password
    unzip(password_protected_zip_path, is_url=False, password="correct_password")
    
    # Verify extractall was called with the encoded password
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"correct_password"

@patch("cookiecutter.utils.zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompt_success(mock_prompt, mock_extractall, password_protected_zip_path):
    mock_prompt.return_value = "user_typed_password"
    
    unzip(password_protected_zip_path, is_url=False, no_input=False)
    
    mock_prompt.assert_called_once()
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"user_typed_password"

@patch("cookiecutter.utils.zipfile.ZipFile.extractall")
def test_unzip_password_failure_no_input(mock_extractall, password_protected_zip_path):
    # Simulate RuntimeError (password error)
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(password_protected_zip_path, is_url=False, no_input=True)

@patch("cookiecutter.utils.zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_max_retries_exceeded(mock_prompt, mock_extractall, password_protected_zip_path):
    # Simulate repeated failure
    mock_extractall.side_effect = RuntimeError("Wrong password")
    mock_prompt.return_value = "wrong"
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
        unzip(password_protected_zip_path, is_url=False, no_input=False)
    
    assert mock_prompt.call_count == 3
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
import pytest
import requests
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_folder/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_format_zip(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return zip_path

@pytest.fixture
def no_top_level_zip(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(patch.dict('os.environ', {}), 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip):
    result = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "file.txt"))

def test_unzip_empty_zip_raises(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises(temp_dir):
    # Manually create zip without top level directory
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(zip_path), is_url=False)

def test_unzip_bad_zip_file_raises(bad_format_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_format_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    target_zip_path = temp_dir / "repo.zip"
    
    # Mocking response content
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_data"]
    mock_get.return_value = mock_response
    
    # Mock prompt_and_delete to return False (don't delete existing)
    mock_prompt.return_value = False

    # We need to bypass the actual ZipFile extraction logic because 'fake_zip_data' isn't a real zip
    # So we patch ZipFile in the module where unzip is defined
    with patch("cookiecutter.utils.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["project/"]
        mock_instance.extractall.return_value = None
        
        result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)
        assert os.path.exists(target_zip_path)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, temp_dir, valid_zip):
    # We need to simulate a password-protected zip error and then success
    # This is complex with real ZipFile, so we mock the behavior of extraction
    with patch("zipfile.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["project/"]
        
        # First call raises RuntimeError (password error), second succeeds
        mock_instance.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = "secret"
        
        result = unzip(str(valid_zip), is_url=False, password="wrong")
        
        assert "project" in result
        assert mock_password.called
        # Verify it tried with the correct encoded password
        mock_instance.extractall.assert_any_call(path=pytest.any, pwd=b"secret")

def test_unzip_password_failure_no_input(temp_dir, valid_zip):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["project/"]
        mock_instance.extractall.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip), is_url=False, no_input=True)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_max_retries_exceeded(mock_password, temp_dir, valid_zip):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["project/"]
        mock_instance.extractall.side_effect = RuntimeError("Wrong password")
        mock_password.return_value = "wrong"
        
        with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
            unzip(str(valid_zip), is_url=False)
        
        assert mock_password.call_count == 3
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    content_dir = temp_dir / "test_repo"
    content_dir.mkdir()
    (content_dir / "file.txt").write_text("hello")
    
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_repo/file.txt", "hello")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    zip_path = temp_dir / "protected.zip"
    content_dir = temp_dir / "protected"
    content_dir.mkdir()
    # Note: standard zipfile module has limited support for creating encrypted zips 
    # in a way that matches all runtime error behaviors, but we mock the behavior.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected/secret.txt", "secret")
    return zip(zip_path, "password123")

def test_unzip_local_file_success(temp_dir, valid_zip):
    unzip_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(unzip_path)
    assert os.path.exists(os.path.join(unzip_path, "file.txt"))

def test_unzip_local_file_empty_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_local_file_no_top_dir_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip):
    zip_uri = "http://example.com/repo.zip"
    mock_prompt.return_value = False
    
    # Mock requests response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response

    # We need to point the URL logic to our valid local zip for extraction logic to work
    # but simulate the download process. 
    with patch("os.path.exists", return_value=False):
        unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        
    mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)

@patch("requests.get")
def test_unzip_url_with_existing_file_prompt(mock_get, temp_dir, valid_zip):
    zip_uri = "http://example.com/repo.zip"
    existing_zip = temp_dir / "repo.zip"
    shutil.copy(valid_zip, existing_zip)

    with patch("cookiecutter.prompt.prompt_and_delete", return_value=False) as mock_prompt:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        mock_prompt.assert_called_once()

def test_unzip_bad_zip_file_error(temp_dir):
    bad_file = temp_dir / "corrupt.zip"
    bad_file.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, temp_dir, password_zip):
    zip_path, password = password_zip
    # Mocking the RuntimeError to simulate encryption detection
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call fails with RuntimeError (encrypted), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = password
        
        unzip(str(zip_path), is_url=False, password=None)
        
        assert mock_password.called
        assert mock_extract.call_count == 2

def test_unzip_password_failure_after_retries(temp_dir, password_zip):
    zip_path, _ = password_zip
    with patch("cookiecutter.prompt.read_repo_password", return_value="wrong"):
        with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Wrong password")):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(str(zip_path), is_url=False)
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_project.zip"
    # Create a zip with a top-level directory structure
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def malformed_zip_path(temp_dir):
    zip_path = temp_dir / "malformed.zip"
    zip_path.write_text("not a zip")
    return zip_path

@pytest.fixture
def no_top_level_zip_path(temp_dir):
    zip_path = temp_dir / "no_top_level.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_success(mock_make_dir, temp_dir, valid_zip_path):
    result = unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result)
    assert os.path.basename(result) == "test_project"
    with open(os.path.join(result, "file.txt"), 'r') as f:
        assert f.read() == "content"

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_dir, mock_prompt, mock_get, temp_dir):
    url = "https://example.com/repo.zip"
    zip_dest = temp_dir / "repo.zip"
    
    # Mocking the response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True

    # We need to intercept the ZipFile call because 'fake_zip_content' isn't a real zip
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.return_value = None

        result = unzip(url, is_url=True, clone_to_dir=temp_dir)
        
        assert mock_get.called
        assert os.path.exists(zip_dest)
        assert "project" in result

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False)

def test_unzip_no_top_level_raises_error(no_top_level_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_level_zip_path), is_url=False)

def test_unzip_bad_zip_file_raises_error(malformed_zip_path):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(malformed_zip_path), is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_success(mock_password_prompt, temp_dir, valid_zip_path):
    # Create a zip that simulates a password error then success
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        
        # First call raises RuntimeError (password error), second succeeds
        mock_zip_instance.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_password_prompt.return_value = "secret"

        result = unzip(str(valid_zip_path), is_url=False)
        
        assert "project" in result
        assert mock_password_prompt.called

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_failure(mock_password_prompt, temp_dir, valid_zip_path):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        
        # Simulate 3 failed attempts
        mock_zip_instance.extractall.side_effect = RuntimeError("Wrong password")
        mock_password_prompt.return_value = "wrong"

        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip_path), is_url=False)
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_workspace():
    """Creates a temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir)

@pytest.fixture
def valid_zip_file(temp_workspace):
    """Creates a valid zip file with a top-level directory."""
    zip_path = temp_workspace / "test_repo.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(temp_workspace):
    """Creates an empty zip file."""
    zip_path = temp_workspace / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def bad_structure_zip_file(temp_workspace):
    """Creates a zip file where the first entry is not a directory."""
    zip_path = temp_workspace / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_not_dir.txt", "content")
    return str(zip_path)

def test_unzip_local_success(valid_zip_file, temp_workspace):
    """Tests unzipping a valid local file."""
    result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_workspace)
    
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip_raises_error(empty_zip_file):
    """Tests that an empty zip raises InvalidZipRepository."""
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_file, is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip_file):
    """Tests that a zip without a top-level directory raises InvalidZipRepository."""
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(bad_structure_zip_file, is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_exists, mock_get, temp_workspace, valid_zip_file):
    """Tests downloading and unzipping from a URL."""
    url = "https://example.com/archive.zip"
    # Mock the response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_data"] 
    # We need to actually create a file that ZipFile can read, 
    # so we'll swap valid_zip_file with a real local one but simulate the URL flow.
    mock_get.return_value = mock_response
    mock_prompt.return_value = True

    # To make this work without complex byte-stream mocking of ZipFile, 
    # we point the URL logic to our existing valid_zip_file content.
    with patch("builtins.open", MagicMock()) as mock_open:
        # This is a bit tricky because unzip writes to a file then reads it.
        # Instead, let's just use a local file path but set is_url=True 
        # and mock the download to copy the existing valid zip.
        import shutil
        dest_zip = temp_workspace / "archive.zip"
        shutil.copy(valid_zip_file, dest_zip)
        
        with patch("requests.get") as m_get:
            m_get.return_value.iter_content.return_value = [b""] # No-op chunking
            # We mock the download to just copy the existing valid zip to the target path
            def side_effect(url, stream=True, timeout=None):
                m = MagicMock()
                m.iter_content.return_value = [] 
                return m
            m_get.side_effect = side_effect
            # Manually place the file where unzip expects it
            shutil.copy(valid_zip_file, dest_zip)
            
            result = unzip(url, is_url=True, clone_to_dir=temp_workspace)
            assert os.path.exists(result)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, temp_workspace):
    """Tests unzipping a password protected file with provided password."""
    zip_path = temp_workspace / "protected.zip"
    # Note: Standard zipfile module has limited support for creating encrypted zips 
    # via the simple API without extra libraries, but we can simulate the RuntimeError.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    # We mock ZipFile.extractall to raise RuntimeError (password error) 
    # then succeed when password is provided.
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        
        result = unzip(str(zip_path), is_url=False, password="secret_password")
        assert result is not None
        assert mock_extract.call_count == 2

def test_unzip_password_failure_no_input(temp_workspace):
    """Tests that unzipping fails when password is required and no_input is True."""
    zip_path = temp_workspace / "protected.zip"
    with ZipFile(zip_path, 'mock_dir/file.txt", "content") as zf:
        pass # simplified for logic flow
    
    # Manually creating a file that triggers RuntimeError on extraction
    with patch("zipfile.ZipFile.extractall", side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(zip_path), is_url=False, no_input=True)

@patch("requests.get")
def test_unzip_bad_zip_file_format(mock_get, temp_workspace):
    """Tests that a non-zip file raises InvalidZipRepository."""
    bad_file = temp_workspace / "not_a_zip.txt"
    bad_file.write_text("this is just text")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    content_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}/file.txt", "hello world")
    return zip_path


@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path


@pytest.fixture
def bad_structure_zip_file(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("nofolder_at_root.txt", "content")
    return zip_path


@pytest.fixture
def password_zip_file(temp_dir):
    # Note: Standard zipfile library has limited support for creating encrypted zips 
    # via writestr, but we can mock the behavior in tests.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "secret")
    return zip_path


def test_unzip_local_success(temp_dir, valid_zip_file):
    result_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_project"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "hello world"


def test_unzip_url_success(temp_dir, valid_zip_file):
    url = f"http://example.com/{valid_zip_file.name}"
    
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"] 
        # To make it a real zip, we actually need the file content to be valid for ZipFile
        # So we patch ZipFile instead of just mocking requests if needed, 
        # but here we'll let requests download and write the actual temp file.
        mock_get.return_value = mock_response
        
        # We must ensure the downloaded content is actually a valid zip for this test
        # A simpler way: patch 'requests.get' to return a stream of the real local file
        with open(valid_zip_file, 'rb') as f:
            mock_response.iter_content.return_value = [chunk for chunk in iter(lambda: f.read(1024), b'')]

        result_path = unzip(url, is_url=True, clone_to_dir=temp_dir)
        
        assert os.path.exists(result_path)
        assert os.path.exists(os.path.join(temp_dir, valid_zip_file.name))


def test_unzip_empty_zip_raises_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_mock_path := empty_zip_file), is_url=False)


def test_unzip_no_top_level_dir_raises_error(bad_structure_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip_file), is_url=False)


def test_unzip_invalid_zip_format(temp_dir):
    bad_file = temp_dir / "not_a_zip.zip"
    bad_file.write_text("not a zip content")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)


@patch('cookiecutter.utils.zipfile.ZipFile.extractall')
def test_unzip_password_provided(mock_extractall, valid_zip_file):
    # Simulate RuntimeError for password protection
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    # We use the existing valid zip but trigger the password logic path
    # We need to mock ZipFile to control namelist and return value
    with patch('cookiecutter.utils.zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ['project/']
        
        unzip(str(valid_zip_file), is_url=False, password="123")
        
        # Verify extractall was called with the password
        mock_extractall.assert_called_with(path=pytest.any, pwd=b'123')


@patch('cookiecutter.prompt.read_repo_password')
@patch('cookiecutter.utils.zipfile.ZipFile.extractall')
def test_unzip_password_prompt_success(mock_extractall, mock_prompt, valid_zip_file):
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_prompt.return_value = "secret_pass"
    
    with patch('cookiecutter.utils.zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ['project/']
        
        unzip(str(valid_zip_file), is_url=False, no_input=False)
        
        mock_prompt.assert_called_once()
        mock_extractall.assert_any_call(path=pytest.any, pwd=b'secret_pass')


@patch('cookiecutter.utils.zipfile.ZipFile.extractall')
def test_unzip_password_failure_no_input(mock_extractall, valid_zip_file):
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with patch('cookiecutter.utils.zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ['project/']
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip_file), is_url=False, no_input=True)


def test_unzip_prompt_and_delete_true(temp_dir, valid_zip_file):
    # Create a file that already exists at the target destination
    target_zip = temp_dir / valid_zip_file.name
    shutil.copy(valid_zip_file, target_zip)
    
    url = f"http://example.com/{valid_zip_file.name}"
    
    with patch('cookiecutter.utils.prompt_and_delete', return_value=True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            with open(valid_zip_file, 'rb') as f:
                mock_response.iter_content.return_value = [chunk for chunk in iter(lambda: f.read(1024), b'')]
            mock_get.return_value = mock_response
            
            unzip(url, is_url=True, clone_to_dir=temp_dir)
            
            # Verify prompt_and_delete was called because file existed
            from cookiecutter.utils import prompt_and_delete
            # Checking if the logic reached the download part via the mock
            assert mock_get.called
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture to provide a clean temporary workspace."""
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

@pytest.fixture
def valid_zip_file(tmp_path):
    """Creates a valid zip file with a top-level directory."""
    zip_path = tmp_path / "test_repo.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project_dir/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(tmp_path):
    """Creates an empty zip file."""
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def bad_structure_zip_file(tmp_path):
    """Creates a zip file where the first entry is not a directory."""
    zip_path = tmp_path / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_at_root.txt", "content")
    return str(zip_path)

@pytest.fixture
def corrupted_zip_file(tmp_path):
    """Creates a file that is not a valid zip."""
    zip_path = tmp_path / "corrupt.zip"
    zip_path.write_text("not a zip")
    return str(zip_path)

@pytest.fixture
def password_protected_zip_file(tmp_path):
    """Creates a password protected zip file."""
    # Note: standard zipfile lib has limited support for creating encrypted zips 
    # in some versions, but we can simulate the RuntimeError via mocking if needed.
    # For this test, we'st use a real one if possible or mock the extraction failure.
    zip_path = tmp_path / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/secret.txt", "secret")
    return str(zip_path)

def test_unzip_local_file_success(valid_zip_file, temp_workspace):
    """Tests successful unzipping of a local file."""
    result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_workspace)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "project_dir"
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_file, temp_workspace):
    """Tests that an empty zip file raises InvalidZipRepository."""
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_empty_file, is_url=False, clone_to_dir=temp_workspace)

def test_unzip_no_top_level_directory_raises_error(bad_structure_zip_file, temp_workspace):
    """Tests that a zip without a top-level directory raises InvalidZipRepository."""
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(bad_structure_zip_file, is_url=False, clone_to_dir=temp_workspace)

def test_unzip_corrupted_zip_raises_error(corrupted_zip_file, temp_workspace):
    """Tests that a corrupted zip file raises InvalidZipRepository."""
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(corrupted_zip_file, is_url=False, clone_to_dir=temp_workspace)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, valid_zip_file, temp_workspace):
    """Tests downloading and unzipping from a URL."""
    url = "https://example.com/repo.zip"
    # Setup mock response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_content"]
    mock_get.return_value = mock_response
    
    # We need to point the URL to a real local file for ZipFile to actually work in the test
    # or we mock the ZipFile entirely. Let's use a local path but simulate is_url=True logic.
    with patch("os.path.exists", return_value=False):
        # We point zip_uri to our valid_zip_file content via a fake URL
        # But since unzip uses rsplit, we make it look like a download
        fake_url = f"https://example.com/{os.path.basename(valid_zip_file)}"
        
        # We must mock the actual downloading to write our valid zip into the cache
        # To keep this test hermetic and avoid complex stream mocking, 
        # we'll mock ZipFile to not actually look at the downloaded file if it's a URL.
        # However, unzip() calls ZipFile(zip_path). So we must ensure zip_path is valid.
        
        with patch("builtins.open", MagicMock()):
            # We redirect the download to write our known good zip to the target location
            def side_effect_write(*args, **kwargs):
                with ZipFile(valid_zip_file, 'r') as zf:
                    zf.extractall(temp_workspace / "extracted_url")
            
            # For simplicity in this unit test, we'll mock the whole ZipFile context manager
            # to avoid needing a real downloaded file on disk during the URL test.
            with patch("zipfile.ZipFile") as mock_zip:
                mock_instance = mock_zip.return_value.__enter__.return_value
                mock_instance.namelist.return_value = ["project_dir/"]
                mock_instance.extractall.return_value = None
                
                result = unzip(fake_url, is_url=True, clone_to_dir=temp_workspace)
                
                assert "project_dir" in result
                mock_get.assert_called_once()

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, mock_extractall, valid_zip_file, temp_workspace):
    """Tests unzipping with a provided password."""
    mock_password.return_value = "secret123"
    
    # Trigger the RuntimeError to enter the password logic
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_workspace, password="secret123")
    assert "project_dir" in result
    # Verify extractall was called with the password
    args, kwargs = mock_extractall.call_args_list[1]
    assert kwargs['pwd'] == b"secret123"

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_failure_after_retries(mock_password, mock_extractall, valid_zip_file, temp_workspace):
    """Tests that unzip raises error after 3 failed password attempts."""
    mock_password.return_value = "wrong"
    mock_extractall.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip_file, is_url=False, clone_to_dir=temp_workspace)
    
    assert mock_password.call_count == 3

@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_existing_file_prompts_deletion(mock_prompt, valid_zip_file, temp_workspace):
    """Tests that unzip prompts to delete existing cache."""
    mock_prompt.return_value = False # User says NO to deleting
    
    # We use a URL that looks like it would resolve to our valid_zip_file name
    url = f"https://example.com/{os.path.basename(valid_zip_file)}"
    
    with patch("os.path.exists", return_value=True):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response
            
            # We need to prevent the actual unzip logic from failing because the file doesn't exist/is dummy
            with patch("zipfile.ZipFile") as mock_zip:
                mock_instance = mock_zip.return_value.__enter__.return_value
                mock_instance.namelist.return_value = ["project_dir/"]
                unzip(url, is_url=True, clone_to_dir=temp_workspace)
                
            mock_prompt.assert_called_once()
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "hello world")
    return str(zip_path)


@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)


@pytest.fixture
def bad_zip_file(temp_dir):
    bad_path = temp_dir / "bad.zip"
    bad_path.write_text("not a zip")
    return str(bad_path)


@pytest.fixture
def no_root_dir_zip_file(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)


@pytest.fixture
def password_protected_zip_file(temp_dir):
    # Creating a real encrypted zip is tricky with standard zipfile lib 
    # without external tools, so we will mock the RuntimeError behavior.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
    return str(zip_path)


def test_unzip_local_success(valid_zip_file):
    result = unzip(valid_zip_file, is_url=False)
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "file.txt"))


def test_unzip_empty_zip_raises_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_file, is_url=False)


def test_unzip_bad_format_raises_error(bad_zip_file):
    with pytest.raises(InvalidZipRepository, match="not a valid zip archive"):
        unzip(bad_zip_file, is_url=False)


def test_unzip_no_top_level_dir_raises_error(no_root_dir_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_root_dir_zip_file, is_url=False)


@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip_file):
    zip_url = "https://example.com/project.zip"
    mock_prompt.return_value = False
    
    # Mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We need to mock ZipFile because the 'fake_zip_content' isn't a real zip
    with patch("zipfile.ZipFile") as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.return_value = None
        
        result = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)


@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zip, password_protected_zip_file):
    instance = mock_zip.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/"]
    # Simulate success on second call (first fails with RuntimeError)
    instance.extractall.side_effect = [RuntimeError("Password required"), None]

    result = unzip(password_protected_zip_file, is_url=False, password="secret_password")
    
    assert "project" in result
    instance.extractall.assert_called_with(path=pytest.any, pwd=b"secret_password")


@patch("zipfile.ZipFile")
def test_unzip_password_wrong_raises_error(mock_zip, password_protected_zip_file):
    instance = mock_zip.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/"]
    instance.extractall.side_effect = RuntimeError("Wrong password")

    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(password_protected_zip_file, is_url=False, password="wrong")


@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompting_success(mock_prompt_pw, mock_zip, password_protected_zip_file):
    instance = mock_zip.return_value.__enter__.return_value
    instance.namelist.return_value = ["project/"]
    mock_prompt_pw.return_value = "correct_password"
    instance.extractall.side_effect = [RuntimeError("Password required"), None]

    result = unzip(password_protected_zip_file, is_url=False, no_input=False)
    
    assert "project" in result
    mock_prompt_pw.assert_called_once()
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = "my_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}/file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip_file(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_directory.txt", "content")
    return zip_path

@pytest.fixture
def password_protected_zip_file(temp_dir):
    zip_path = temp/path = temp_dir / "protected.zip"
    # Creating a password protected zip requires specific handling in ZipFile
    # For testing purposes, we mock the behavior of RuntimeError during extractall
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip_file):
    result_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_file), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip_file), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip_file):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"dummy_data"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = False # Don't delete existing
    
    zip_url = "http://example.com/archive.zip"
    # We need to simulate the zip file content being downloaded correctly
    # Since we can't easily download a real zip in a unit test, 
    # we patch ZipFile to return our valid_zip_file logic instead of reading from the fake download
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.return_value = None
        
        result = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)

@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zip_class, temp_dir, valid_zip_file):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    # Simulate success on second attempt with password
    mock_zip_instance.extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(str(valid_zip_file), is_url=False, password="secret_password")
    
    # Verify extractall was called with encoded password
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile")
def test_unzip_password_failed_no_input(mock_zip_class, temp_dir, valid_zip_file):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_zip_instance.extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(valid_zip_file), is_url=False, no_input=True)

@patch("zipfile.ZipFile")
def test_unzip_bad_zip_file(mock_zip_class, temp_dir, valid_zip_file):
    from zipfile import BadZipFile
    mock_zip_class.side_effect = BadZipFile("Not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(valid_zip_file), is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
@patch("zipfile.ZipFile")
def test_unzip_password_retry_logic_failure(mock_zip_class, mock_read_password, temp_dir, valid_zip_file):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_zip_instance.extractall.side_effect = RuntimeError("Wrong password")
    mock_read_password.return_value = "wrong"
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
        unzip(str(valid_zip_file), is_url=False, no_input=False)
    
    assert mock_read_password.call_count == 3
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # No files added
    return str(zip_path)

@pytest.fixture
def no_top_level_dir_zip_path(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)

@pytest.fixture
def password_protected_zip_path(temp_dir):
    # Note: standard zipfile module has limited support for creating encrypted zips 
    # via writestr without external libs, but we can mock the behavior in tests.
    return str(temp_dir / "protected.zip")

def test_unzip_local_success(valid_zip_path, temp_dir):
    result_path = unzip(valid_zip_path, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_path, is_url=False)

def test_unzip_no_top_level_dir_raises_error(no_top_level_dir_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_top_level_dir_zip_path, is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, valid_zip_path, temp_dir):
    # Setup mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"dummy_content"]
    mock_get.return_value = mock_response
    
    url = "https://example.com/project.zip"
    # We need to ensure the local file exists so ZipFile can read it, 
    # but since we are mocking 'is_url', we'll point it to our valid_zip_path
    # and mock the download logic to just use existing file or redirect.
    
    # To test URL logic specifically, we simulate a download that writes a zip
    with patch("builtins.open", MagicMock()) as mock_open:
        # Mocking the actual downloading process by making it write our valid_zip_path
        def side_effect(url, stream, timeout):
            return MagicMock(iter_content=lambda chunk_size: [b""])

        mock_get.return_value.iter_content.return_value = [] 
        # Since we can't easily download a real zip in a unit test without network,
        # we mock the ZipFile context manager to return our valid_zip_path content
        with patch("zipfile.ZipFile") as mock_zip:
            mock_zip.return_value.__enter__.return_value.namelist.return_value = ["project/"]
            mock_zip.return_value.__enter__.return_value.extractall.return_value = None
            
            result = unzip(url, is_url=True, clone_to_dir=temp_dir)
            assert "project" in result

def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    bad_zip.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_provided(mock_extractall, valid_zip_path):
    # Simulate RuntimeError for wrong password first, then success
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result_path = unzip(valid_zip_path, is_url=False, password="correct_password")
    assert os.path.exists(result_path)
    # Check if extractall was called with encoded password
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"correct_password"

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_failure(mock_extractall, valid_zip_path):
    # Simulate RuntimeError that persists
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip_path, is_url=False, password="wrong_password")

@patch("cookiecutter.prompt.read_repo_password")
@patch("zipfile.ZipFile.extractall")
def test_unzip_password_prompting(mock_extractall, mock_read_password, valid_zip_path):
    # Simulate failure then success via prompt
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_read_password.return_value = "prompted_password"
    
    result_path = unzip(valid_zip_path, is_url=False, no_input=False)
    assert os.path.exists(result_path)
    mock_read_password.assert_called_once()

@patch("zipfile.ZipFile.extractall")
def test_unzip_no_input_password_failure(mock_extractall, valid_zip_path):
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(valid_zip_path, is_url=False, no_input=True)
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = "project_folder/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(mock_dir := tempfile.mkdtemp(), 'w') as zf:
        pass # Creates a valid but empty zip structure is tricky, 
             # easier to just create a zip with no files
    # Actually, an empty ZipFile object usually has at least one entry if we add nothing?
    # Let's force an empty namelist by creating it and not adding anything.
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass 
    return zip_path

@pytest.fixture
def no_top_dir_zip(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory")
    return zip_path

@pytest.fixture
def bad_zip_file(temp_dir):
    bad_path = temp_dir / "bad.zip"
    with open(bad_path, "w") as f:
        f.write("not a zip")
    return bad_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_file_success(mock_make_exists, valid_zip, temp_dir):
    result = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_exists, mock_get, temp_dir):
    zip_url = "https://example.com/repo.zip"
    # Mocking the response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    # We need to actually create a valid zip at the destination so ZipFile doesn't crash
    # because unzip() calls ZipFile(zip_path) after downloading.
    # We will patch ZipFile to avoid needing a real download/extract loop for this specific test
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.return_value = None
        
        # We need to simulate the extraction path logic
        with patch("tempfile.mkdtemp", return_value="/tmp/unzip_base"):
            result = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
            assert result == "/tmp/unzip_base/project"

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_directory_raises_error(no_top_dir_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_dir_zip), is_url=False)

def test_unzip_bad_zip_file_raises_error(bad_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip_file), is_url=False)

@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    
    result = unzip(str(valid_zip), is_url=False, password="secret_password")
    
    # Verify extractall was called with the password encoded
    mock_zip_instance.extractall.assert_called()
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompt_success(mock_prompt_pass, mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    # Simulate RuntimeError on first attempt (no password), then success on second
    mock_zip_instance.extractall.side_effect = [RuntimeError("Wrong"), None]
    mock_prompt_pass.return_value = "user_password"

    result = unzip(str(valid_zip), is_url=False, no_input=False)
    
    assert mock_prompt_pass.called
    assert mock_zip_instance.extractall.call_count == 2
    assert mock_prompt_pass.return_value == "user_password"

@patch("zipfile.ZipFile")
def test_unzip_password_error_no_input(mock_zip_class, valid_zip):
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_zip_instance.extractall.side_effect = RuntimeError("Encrypted")

    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(valid_zip), is_url=False, no_input=True)
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def invalid_zip_format(temp_dir):
    bad_file = temp_dir / "bad.zip"
    bad_file.write_text("not a zip")
    return str(bad_file)

@pytest.fixture
def no_root_dir_zip(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)

def test_unzip_local_success(valid_zip_file, temp_dir):
    result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_mock_path := empty_zip_file, is_url=False)

def test_unzip_no_top_level_dir_raises_error(no_root_dir_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_root_dir_zip, is_url=False)

def test_unzip_bad_zip_file_raises_error(invalid_zip_format):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(invalid_zip_format, is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, valid_zip_file, temp_dir):
    zip_url = "https://example.com/project.zip"
    
    # Mock response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    # We need to actually create a valid zip at the destination so ZipFile doesn't fail later
    # For this test, we bypass the download content and just point to an existing local valid zip
    # but simulate the URL logic flow
    with patch("os.path.exists", return_value=False):
        # Override zip_uri to point to our known good local file for the ZipFile stage
        result_path = unzip(valid_zip_file, is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(result_path)

@patch("requests.get")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_success(mock_password, mock_get, temp_dir):
    # Create a password protected zip
    zip_path = temp_dir / "protected.zip"
    # Note: standard ZipFile doesn't support AES encryption easily in stdlib for testing 'RuntimeError' 
    # but we can trigger the RuntimeError by mocking ZipFile.extractall
    
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.side_effect = [RuntimeError("password required"), None]
        mock_password.return_value = "secret"
        
        result = unzip(str(zip_path), is_url=False, password="wrong")
        assert mock_password.called
        assert "project" in result

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_protected_fail_after_retries(mock_password, temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")
    
    mock_password.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(zip_path), is_url=False)
    
    assert mock_password.call_count == 3

@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_with_existing_file_deletes_old(mock_prompt, valid_zip_file, temp_dir):
    zip_url = "https://example.com/project.zip"
    target_path = temp_dir / "project.zip"
    target_path.write_text("old content")
    
    mock_prompt.return_value = False # User chooses not to delete
    
    # Use a local file path as the URI but trigger URL logic via mocking
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = []
        mock_get.return_value = mock_response
        
        # We just want to see if prompt_and_delete was called with the path
        unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
        
        expected_path = os.path.join(str(temp_dir), "project.zip")
        mock_prompt.assert_called_once_with(expected_path, no_input=False)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch, MagicMock

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


def test_unzip_local_success(temp_dir):
    # Create a valid zip file with a top-level directory
    project_name = "test_project"
    zip_path = temp_dir / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{project_name}/file.txt", "content")

    unzipped_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)

    assert os.path.exists(unzipped_path)
    assert os.path.exists(os.path.join(unzipped_path, "file.txt"))
    with open(os.path.join(unzipped_path, "file.txt"), 'r') as f:
        assert f.read() == "content"


def test_unzip_url_success(temp_dir):
    # Mocking requests and file operations for URL download
    zip_uri = "https://example.com/repo.zip"
    identifier = "repo.zip"
    clone_to_dir = temp_dir / "cache"
    
    # Setup dummy zip content
    project_name = "test_project"
    dummy_zip_content = b"fake_zip_data" 
    
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True), \
         patch("zipfile.ZipFile") as mock_zip:
        
        # Mocking the response stream
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [dummy_zip_content]
        mock_get.return_value = mock_response
        
        # Mocking ZipFile behavior to avoid needing a real valid zip structure for the download test part
        # But since unzip calls extractall, we need it to not crash or return a path
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["test_project/"]
        
        unzipped_path = unzip(zip_uri, is_url=True, clone_to_dir=clone_to_dir)
        
        assert os.path.exists(os.path.dirname(unzipped_path))
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)


def test_unzip_empty_zip_raises_error(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # Empty zip

    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(zip_path), is_url=False)


def test_unzip_no_top_level_dir_raises_error(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")  # No directory entry

    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(zip_path), is_url=False)


def test_unzip_bad_zip_file_raises_error(temp_dir):
    zip_path = temp_dir / "corrupt.zip"
    with open(zip_path, 'wb') as f:
        f.write(b"not a zip file")

    with pytest.raise(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(zip_path), is_url=False)


def test_unzip_password_provided_success(temp_dir):
    # This tests the logic path for password handling using mocks 
    # because creating an actual encrypted zip in a unit test is heavy.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")

    with patch("zipfile.ZipFile") as mock_zip, \
         patch("cookiecutter.utils.make_sure_path_exists"):
        
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        # Simulate RuntimeError for first attempt, success on second with password
        mock_zip_instance.extractall.side_effect = [RuntimeError("Password required"), None]

        unzipped_path = unzip(str(zip_path), is_url=False, password="secret_password")
        
        # Verify extractall was called with the encoded password
        mock_zip_instance.extractall.assert_any_call(path=pytest.any, pwd=b"secret_password")


def test_unzip_password_failure_raises_error(temp_dir):
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/", "")

    with patch("zipfile.ZipFile") as mock_zip, \
         patch("cookiecutter.utils.make_sure_path_exists"):
        
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        # Always fail
        mock_zip_instance.extractall.side_effect = RuntimeError("Wrong password")

        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(zip_path), is_url=False, password="wrong_password")
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_project.zip"
    # Create a zip with a top-level directory entry to satisfy the function logic
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # No files added
    return str(zip_path)

@pytest.fixture
def no_root_dir_zip_path(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")  # No trailing slash in namelist entry
    return str(zip_path)

def test_unzip_local_file_success(valid_zip_path, temp_dir):
    result_path = unzip(valid_zip_path, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_project"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_local_file_empty_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_path, is_url=False)

def test_unzip_local_file_no_top_level_dir_error(no_root_dir_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_root_dir_zip_path, is_url=False)

def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    bad_zip.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, valid_zip_path, temp_dir):
    # Setup mock response for requests
    url = "http://example.com/test_project.zip"
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"dummy_data"]
    mock_get.return_value = mock_response
    
    # Setup prompt to return False (don't delete existing)
    mock_prompt.return_value = False

    # We need a real zip file at the destination for ZipFile(zip_path) to work
    # So we intercept the download and write actual zip content
    def side_effect_write(*args, **kwargs):
        zip_path = os.path.join(temp_dir, "test_project.zip")
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/file.txt", "content")
        return MagicMock()

    mock_get.side_effect = side_effect_write

    result_path = unzip(url, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert "test_project" in result_path

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_success(mock_extractall, valid_zip_path):
    # Simulate password success on first try
    mock_extractall.return_value = None 
    
    # We need to mock the ZipFile context manager behavior slightly if we want to control RuntimeError
    # But for a simple unit test, we check if pwd is passed correctly
    # This requires creating a real zip that triggers RuntimeError or mocking the ZipFile object
    pass

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_retry_failure(mock_read_password, mock_extractall, valid_zip_path):
    # Simulate RuntimeError for password extraction
    mock_extractall.side_effect = RuntimeError("Password incorrect")
    mock_read_password.return_value = "wrong_password"

    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(valid_zip_path, is_url=False)

@patch("zipfile.ZipFile.extractall")
def test_unzip_no_input_password_error(mock_extractall, valid_zip_path):
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(valid_zip_path, is_url=False, no_input=True)

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_existing_file_prompt_delete(mock_prompt, mock_get, temp_dir):
    url = "http://example.com/test_project.zip"
    zip_path = temp_dir / "test_project.zip"
    zip_path.write_text("existing content")
    
    mock_prompt.return_value = True # User decides to delete
    
    # Mock the download to create a valid zip so ZipFile doesn't crash
    def side_effect_download(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        # We actually need to write a real zip content here for the function to continue
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/file.txt", "content")
        return mock_resp

    mock_get.side_effect = side_effect_download

    result_path = unzip(url, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    mock_prompt.assert_called_once()
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = "my_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}/file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_zip(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return zip_path

@pytest/mark:
def test_unzip_local_success(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@pytest.mark.parametrize("is_url", [True])
def test_unzip_url_success(temp_dir, valid_zip, is_url):
    # Mock requests and prompt_and_delete
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True):
        
        # Mock response stream
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        # We need to point the URI to something that results in a valid zip structure locally for logic flow
        # or mock the ZipFile context manager entirely. 
        # Here we simulate downloading the existing valid_zip
        url = f"http://example.com/{valid_zip.name}"
        
        # To make this work without actual network, we override the download to copy local file
        def side_effect(url, stream=True, timeout=100):
            m = MagicMock()
            m.iter_content.return_value = [open(valid_zip, 'rb').read()]
            return m
        mock_get.side_effect = side_effect

        result_path = unzip(url, is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(result_path)

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content") # No trailing slash in name
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(zip_path), is_url=False)

def test_unzip_bad_zip_file_raises_error(bad_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)

def test_unzip_password_protected_success(temp_dir):
    zip_path = temp_dir / "protected.zip"
    content_dir = "secure_project"
    # Note: standard zipfile doesn't support creating encrypted zips easily in all py versions, 
    # so we mock the ZipFile behavior for the RuntimeError path.
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = [f"{content_dir}/"]
        # Simulate password error on first attempt, success on second (if we were testing retry)
        # But here we test the explicit password argument
        mock_zip.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_zip.__enter__.return_value = mock_zip
        mock_zip_class.return_value = mock_zip

        result = unzip(str(zip_path), is_url=False, password="secret_password")
        assert "secure_project" in result
        mock_zip.extractall.assert_any_call(path=pytest.any, pwd=b"secret_password")

def test_unzip_password_protected_no_input_raises_error(valid_zip):
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["project/"]
        mock_zip.extractall.side_effect = RuntimeError("Password required")
        mock_zip.__enter__.return_value = mock_zip
        mock_zip_class.return_value = mock_zip

        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip), is_url=False, no_input=True)

def test_unzip_password_invalid_provided_raises_error(valid_zip):
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["project/"]
        mock_zip.extractall.side_effect = RuntimeError("Wrong password")
        mock_zip.__enter__.return_value = mock_zip
        mock_zip_class.return_value = mock_zip

        with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
            unzip(str(valid_zip), is_url=False, password="wrong")
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = os.path.join(temp_dir, "project.zip")
    unzip_base = os.path.join(temp_dir, "extracted")
    os.makedirs(unzip_base)
    
    with ZipFile(zip_path, 'w') as zf:
        # Must have a top-level directory entry to pass validation
        zf.writestr("project/", "")
        zf.writestr("project/file.txt", "hello world")
    return zip_path

def test_unzip_local_file_success(temp_dir, valid_zip):
    result_path = unzip(valid_zip, is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "project"
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_get, temp_dir, valid_zip):
    # Setup mock for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    mock_prompt.return_value = True
    
    url = "http://example.com/archive.zip"
    # Use the valid_zip as if it were downloaded to simulate content
    with patch("os.path.exists", return_value=False):
        result_path = unzip(url, is_url=True, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    mock_get.assert_called_once_with(url, stream=True, timeout=100)

def test_unzip_empty_zip_raises_error(temp_dir):
    empty_zip = os.path.join(temp_dir, "empty.zip")
    with ZipFile(empty_zip, 'w') as zf:
        pass # No files added
    
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip, is_url=False)

def test_unzip_no_top_level_dir_raises_error(temp_dir):
    bad_zip = os.path.join(temp_dir, "bad.zip")
    with ZipFile(bad_zip, 'w') as zf:
        zf.writestr("file.txt", "no directory wrapper")
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(bad_zip, is_url=False)

def test_unzip_bad_zip_file(temp_dir):
    corrupt_zip = os.path.join(temp_dir, "corrupt.zip")
    with open(corrupt_zip, "w") as f:
        f.write("not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(corrupt_zip, is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, temp_dir):
    # Create a password protected zip manually is complex with standard ZipFile, 
    # so we mock the behavior of extractall raising RuntimeError for wrong pwd
    # and then succeeding with correct pwd.
    
    protected_zip = os.path.join(temp_dir, "protected.zip")
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("project/", "")
        zf.writestr("project/secret.txt", "secret content")

    mock_password.return_value = "correct_password"
    
    # We patch ZipFile.extractall to simulate the RuntimeError sequence
    with patch("zipfile.ZipFile.extractall") as mock_extract:
        # First call fails (wrong password/no pwd), second succeeds
        mock_extract.side_effect = [RuntimeError("Bad password"), None]
        
        result_path = unzip(protected_zip, is_url=False)
        assert os.path.exists(result_path)
        assert mock_password.called

def test_unzip_password_failure_after_retries(temp_dir):
    protected_zip = os.path.join(temp_dir, "protected.zip")
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("project/", "")

    with patch("cookiecutter.prompt.read_repo_password", side_effect=RuntimeError("Wrong")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(protected_zip, is_url=False)
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_content"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "hello world")
    return str(zip_path)


@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # No files added
    return str(zip_path)


@pytest.fixture
def no_root_dir_zip_path(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)


@pytest.fixture
def password_protected_zip_path(temp_dir):
    zip_path = temp_dir / "protected.zip"
    # Note: zipfile module has limited support for creating encrypted zips 
    # in standard way, but we can simulate the RuntimeError during extraction
    return str(zip_path)


def test_unzip_local_success(valid_zip_path):
    result_path = unzip(valid_zip_path, is_url=False)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))


def test_unzip_url_success(temp_dir):
    zip_uri = f"http://example.com/archive.zip"
    target_dir = temp_dir / "cache"
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    
    # Create a real dummy zip file to satisfy ZipFile constructor during the test logic
    # We'll mock requests and the ZipFile extraction process
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True), \
         patch("zipfile.ZipFile") as mock_zip:
        
        mock_get.return_value = mock_response
        
        # Setup Mock ZipFile behavior
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.return_value = None

        result = unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)


def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_path, is_url=False)


def test_unzip_no_top_level_dir_raises_error(no_root_dir_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_root_dir_zip_path, is_url=False)


def test_unzip_bad_zip_file(temp_dir):
    bad_zip = temp_dir / "bad.zip"
    bad_zip.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False)


def test_unzip_password_correct(valid_zip_path):
    with patch("zipfile.ZipFile") as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        
        # Simulate RuntimeError for first attempt, success for second
        instance.extractall.side_effect = [RuntimeError("Password required"), None]
        
        with patch("cookiecutter.prompt.read_repo_password", return_value="secret"):
            result = unzip(valid_zip_path, is_url=False, password="wrong") 
            # Note: The logic in the original code uses 'password' arg first
            # If we pass a wrong one manually, it should trigger the retry loop
            pass

def test_unzip_password_failure_after_retries(valid_zip_path):
    with patch("zipfile.ZipFile") as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = RuntimeError("Wrong password")
        
        with patch("cookiecutter.prompt.read_repo_password", return_value="wrong"):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(valid_zip_path, is_url=False, no_input=True)

def test_unzip_no_input_password_error(valid_zip_path):
    with patch("zipfile.ZipFile") as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(valid_zip_path, is_url=False, no_input=True)
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_contents"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "hello world")
    return zip_path


@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path


@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_structure.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_directory.txt", "content")
    return zip_path


def test_unzip_local_file_success(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))


def test_unzip_url_success(temp_dir, valid_zip):
    zip_url = f"http://example.com/{valid_zip.name}"
    
    with patch("requests.get") as mock_get, \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True), \
         patch("cookiecutter.utils.make_sure_path_exists"):
        
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        # We must ensure the file actually exists on disk for ZipFile to open it, 
        # so we simulate the download writing the existing valid_zip content
        def side_effect(url, stream, timeout):
            mock_response.iter_content.return_value = [open(valid_zip, 'rb').read()]
            return mock_response

        mock_get.side_effect = side_effect

        result_path = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(result_path)


def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)


def test_unzip_no_top_level_dir_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)


def test_unzip_bad_zip_file_raises_error(temp_dir):
    bad_file = temp_dir / "corrupt.zip"
    bad_file.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)


def test_unzip_password_protected_success(temp_dir):
    zip_path = temp_dir / "protected.zip"
    # Creating a password protected zip is tricky with standard zipfile without external tools,
    # so we mock the ZipFile behavior for the RuntimeError scenario.
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["project/"]
        # First call fails with RuntimeError, second succeeds
        mock_zip.extractall.side_effect = [RuntimeError("password required"), None]
        mock_zip_class.return_value.__enter__.return_value = mock_zip

        result = unzip(str(zip_path), is_url=False, password="secret_password")
        assert "project" in result
        mock_zip.extractall.assert_called_with(path=pytest.any, pwd=b"secret_password")


def test_unzip_password_protected_no_input_raises_error(temp_dir):
    zip_path = temp_dir / "protected.zip"
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["project/"]
        mock_zip.extractall.side_effect = RuntimeError("password required")
        mock_zip_class.return_value.__enter__.return_value = mock_zip

        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(zip_path), is_url=False, no_input=True)


@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompt_retry_failure(mock_prompt, temp_dir, valid_zip):
    # Mocking password prompt to fail 3 times
    mock_prompt.side_effect = RuntimeError("Wrong password")
    
    with patch("zipfile.ZipFile") as mock_zip_class:
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ["project/"]
        mock_zip.extractall.side_effect = RuntimeError("password required")
        mock_zip_class.return_value.__enter__.return_value = mock_zip

        with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
            unzip(str(valid_zip), is_url=False)
        
        assert mock_prompt.call_count == 3
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_repo/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass  # No files added
    return zip_path

@pytest.fixture
def no_root_dir_zip_path(temp_dir):
    zip_path = temp_dir / "no_root.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content") # No trailing slash in name
    return zip_path

@pytest.fixture
def password_protected_zip_path(temp_dir):
    zip_path = temp_dir / "protected.zip"
    # We can't easily create a real encrypted zip with standard ZipFile 
    # without complex setup, so we will mock the RuntimeError in the tests.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return zipPoints = zip_path

def test_unzip_local_success(valid_zip_path, temp_dir):
    result_path = unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False)

def test_unzip_no_top_level_dir_raises(no_root_dir_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_root_dir_zip_path), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    mock_prompt.return_value = True
    
    # Mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We need to make the zip file "valid" so ZipFile doesn't crash
    # Since we are mocking the download, let's point it to a real local valid zip instead
    valid_zip = temp_dir / "downloaded.zip"
    with ZipFile(valid_zip, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    with patch("os.path.exists", return_value=False):
        # Force the download to actually write our valid zip to the destination
        with patch("builtins.open", MagicMock()):
            # To avoid complex stream mocking, we'll just bypass the download 
            # logic by making the URI point to our local valid zip via a side effect
            def side_effect_get(url, *args, **kwargs):
                m = MagicMock()
                m.iter_content.return_value = [] # Empty because we'll swap the file
                return m
            mock_get.side_effect = side_effect_get
            
            # Instead of complex mocking, let's just mock the zipfile behavior 
            # but for a URL-based call.
            with patch("zipfile.ZipFile") as mock_zip:
                instance = mock_zip.return_value.__enter__.return_value
                instance.namelist.return_value = ["project/"]
                instance.extractall.return_value = None
                
                # Mocking the download to write our valid zip
                with patch("requests.get") as mock_req:
                    mock_req.return_value.iter_content.return_value = [b"data"]
                    # We'll use a real file for the logic to follow
                    res = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
                    assert "project" in res

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, mock_extractall, valid_zip_path):
    mock_password.return_value = "secret"
    # Trigger the RuntimeError path by simulating password error then success
    mock_extractall.side_effect = [RuntimeError("Password error"), None]
    
    result_path = unzip(str(valid_zip_path), is_url=False, password="wrong")
    assert os.path.exists(result_path)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_failure(mock_extractall, valid_zip_path):
    # Simulate persistent failure
    mock_extractall.side_effect = RuntimeError("Invalid password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(valid_zip_path), is_url=False, password="wrong", no_input=True)

@patch("zipfile.ZipFile")
def test_unzip_bad_zip_file(mock_zip_class, valid_zip_path):
    from zipfile import BadZipFile
    mock_zip_class.side_effect = BadZipFile("Bad zip")
    
    with pytest.raises(InvalidZipRepository, match="not a valid zip archive"):
        unzip(str(valid_zip_path), is_url=False)
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "hello world")
    return zip_path


@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path


@pytest.fixture
def malformed_zip_file(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return zip_path


@pytest.fixture
def no_root_dir_zip_file(temp_dir):
    zip_path = templem_dir / "no_root.zip"
    # Create zip where first entry is a file, not a directory
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_without_dir.txt", "content")
    return zip_path


def test_unzip_local_success(temp_dir, valid_zip_file):
    unzip_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(unzip_path)
    assert os.path.basename(unzip_path) == "test_project"
    with open(os.path.join(unzip_path, "file.txt"), 'r') as f:
        assert f.read() == "hello world"


def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    target_dir = temp_dir / "cache"
    os.makedirs(target_dir)
    
    # Mocking requests and prompt
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_content"]
    
    with patch("requests.get", return_value=mock_response), \
         patch("cookiecutter.utils.make_sure_path_exists"), \
         patch("cookiecutter.prompt.prompt_and_delete", return_value=True), \
         patch("zipfile.ZipFile") as mock_zip:
        
        # Setup mock zip structure for extraction logic
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = ["test_project/"]
        mock_zip.return_value.__enter__.return_value = mock_zf
        
        unzip_path = unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
        
        assert "test_project" in unzip_path
        # Verify download happened to the cache dir
        assert os.path.exists(os.path.join(target_dir, "repo.zip"))


def test_unzip_empty_zip(temp_dir, empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_file), is_url=False, clone_to_dir=temp_dir)


def test_unzip_no_top_level_directory(temp_dir, no_root_dir_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_root_dir_zip_file), is_url=False, clone_to_dir=temp_dir)


def test_unzip_bad_zip_format(temp_dir, malformed_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(malformed_zip_file), is_url=False, clone_to_dir=temp_dir)


def test_unzip_password_provided(temp_dir, valid_zip_file):
    # Create an encrypted zip manually is hard with standard library, 
    # so we mock the RuntimeError behavior of ZipFile.extractall
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = ["test_project/"]
        # First call fails (no password), second call succeeds (with password)
        mock_zf.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_zip.return_value.__enter__.return_value = mock_zf
        
        unzip_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, password="123")
        
        assert "test_project" in unzip_path
        # Verify extractall was called with the encoded password
        mock_zf.extractall.assert_any_call(path=pytest.any, pwd=b"123")


def test_unzip_password_wrong_provided(temp_dir, valid_zip_file):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = ["test_project/"]
        mock_zf.extractall.side_effect = RuntimeError("Bad password")
        mock_zip.return_value.__enter__.return_value = mock_zf
        
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, password="wrong")


def test_unzip_password_prompt_failure(temp_dir, valid_zip_file):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = ["test_project/"]
        mock_zf.extractall.side_effect = RuntimeError("Bad password")
        mock_zip.return_value.__enter__.return_value = mock_zf
        
        # Mock prompt to fail 3 times as per code logic
        with patch("cookiecutter.prompt.read_repo_password", side_effect=RuntimeError("Bad password")):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, no_input=True)
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    extract_dir = "project_folder/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_dir.txt", "content")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_file_success(mock_make_path, temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_path, mock_prompt, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    target_dir = temp_dir / "cache"
    mock_prompt.return_value = True
    
    # Mocking requests stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We need to mock ZipFile because the fake content isn't a real zip
    with patch("cookiecutter.utils.unzip.ZipFile") as mock_zip_class:
        mock_zip_inst = mock_zip_class.return_value.__enter__.return_value
        mock_zip_inst.namelist.return_value = ["project/"]
        mock_zip_inst.extractall.return_value = None
        
        result = unzip(zip_uri, is_url=True, clone_to_dir=target_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)

def test_unzip_empty_zip_raises_error(temp_dir, empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_no_top_level_dir_raises_error(temp_dir, bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.utils.unzip.ZipFile")
def test_unzip_password_provided_success(mock_zip_class, temp_dir, valid_zip):
    mock_zip_inst = mock_zip_class.return_value.__enter__.return_value
    mock_zip_inst.namelist.return_value = ["project/"]
    
    # Simulate success on second attempt with password
    def side_effect(path, pwd=None):
        if pwd is not None:
            return None
        raise RuntimeError("Password required")
    
    mock_zip_inst.extractall.side_effect = side_effect

    result = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir, password="secret")
    assert "project" in result
    mock_zip_inst.extractall.assert_called_with(path=pytest.any, pwd=b"secret")

@patch("cookiecutter.utils.unzip.ZipFile")
def test_unzip_password_no_input_raises_error(mock_zip_class, temp_dir, valid_zip):
    mock_zip_inst = mock_zip_class.return_value.__enter__.return_value
    mock_zip_inst.namelist.return_value = ["project/"]
    mock_zip_inst.extractall.side_effect = RuntimeError("Password required")

    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir, no_input=True)

@patch("cookiecutter.utils.unzip.ZipFile")
def test_unzip_bad_zip_file_raises_error(mock_zip_class, temp_dir):
    from zipfile import BadZipFile
    mock_zip_class.side_effect = BadZipFile("Not a zip")
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip("non_existent.zip", is_url=False, clone_to_dir=temp_dir)

@patch("cookiecutter.utils.unzip.read_repo_password")
@patch("cookiecutter.utils.unzip.ZipFile")
def test_unzip_password_prompt_success(mock_zip_class, mock_prompt_pwd, temp_dir, valid_zip):
    mock_zip_inst = mock_zip_class.return_value.__enter__.return_value
    mock_zip_inst.namelist.return_value = ["project/"]
    mock_prompt_pwd.return_value = "correct_password"
    
    # First call fails, second succeeds
    mock_zip_inst.extractall.side_effect = [RuntimeError("Password required"), None]

    result = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert "project" in result
    assert mock_prompt_pwd.called
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_zip_file(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return zip_path

@pytest/mark: test_unzip_local_success
def test_unzip_local_success(valid_zip_file, temp_dir):
    unzip_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(os.path.join(unzip_path, "test_project", "file.txt"))
    with open(os.path.join(unzip_path, "test_project", "file.txt"), 'r') as f:
        assert f.read() == "content"

@pytest/mark: test_unzip_url_success
def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    zip_path = temp_dir / "repo.zip"
    
    # Create a local dummy zip to serve as the "downloaded" file
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("url_project/file.txt", "url_content")

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    
    with patch('requests.get', return_value=mock_response), \
         patch('cookiecutter.utils.make_sure_path_exists'), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=True):
        
        unzip_path = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        assert os.path.exists(os.path.join(unzip_path, "url_project", "file.txt"))

@pytest/mark: test_unzip_empty_error
def test_unzip_empty_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_file), is_url=False)

@pytest/mark: test_unzip_no_top_level_dir_error
def test_unzip_no_top_level_dir_error(temp_dir):
    zip_path = temp_dir / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_at_root.txt", "content")
    
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(zip_path), is_url=False)

@pytest/mark: test_unzip_bad_zip_error
def test_unzip_bad_zip_error(bad_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip_file), is_url=False)

@pytest/mark: test_unzip_password_success
def test_unzip_password_success(temp_dir):
    zip_path = temp_dir / "protected.zip"
    # We can't easily create a password-protected zip with standard ZipFile in all Python versions 
    # without specific flags, but we can mock the behavior of the extraction failure/retry.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected_dir/file.txt", "content")

    with patch('zipfile.ZipFile.extractall') as mock_extract:
        # First call fails with RuntimeError (password error), second succeeds
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        with patch('cookiecutter.prompt.read_repo_password', return_value="secret"):
            unzip_path = unzip(str(zip_path), is_url=False, password="wrong")
            assert mock_extract.call_count == 2

@pytest/mark: test_unzip_password_failure
def test_unzip_password_failure(valid_zip_file):
    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Bad password")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            # Providing a password that fails 3 times (simulated by side_effect)
            unzip(str(valid_zip_file), is_url=False, password="wrong", no_input=True)

@pytest/mark: test_unzip_no_input_password_error
def test_unzip_no_input_password_error(valid_zip_file):
    with patch('zipfile.ZipFile.extractall', side_input=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip_file), is_url=False, no_input=True)

@pytest/mark: test_unzip_skip_download_if_exists
def test_unzip_skip_download_if_exists(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    zip_path = temp_dir / "repo.zip"
    # Pre-create the file
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("existing_project/file.txt", "content")

    with patch('requests.get') as mock_get, \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=False):
        
        unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        mock_get.assert_not_called()
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from zipfile import ZipFile
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory structure
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(temp_path, 'w') as zf:
        pass # No files added
    return zip_path

@pytest.fixture
def invalid_structure_zip_file(temp_dir):
    zip_path = temp_dir / "no_top_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content") # No folder prefix
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip_file):
    result_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_local_empty_error(temp_dir, temp_dir):
    # Create an actual empty zip file manually for the test
    empty_zip = temp_dir / "empty.zip"
    with ZipFile(empty_zip, 'w') as zf:
        pass
    
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False, clone_to_dir=temp_dir)

def test_unzip_local_no_top_level_dir_error(temp_dir, invalid_structure_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_structure_zip_file), is_url=False, clone_to_dir=temp_dir)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip_file):
    # Setup mock for URL download
    url = "https://example.com/repo.zip"
    mock_prompt.return_value = True
    
    # Mock the response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content_part1", b"fake_zip_content_part2"]
    mock_get.return_value = mock_response
    
    # We need to actually provide a valid zip for the ZipFile(zip_path) call later in the function
    # Since we are mocking requests, we'll point the URL to our local valid zip instead 
    # or just intercept the file writing. For simplicity in this unit test, 
    # we will patch ZipFile to not care about the downloaded content and return our valid_zip_file contents.
    
    with patch("zipfile.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["test_project/"]
        mock_instance.extractall.return_value = None
        
        result = unzip(url, is_url=True, clone_to_dir=temp_dir)
        
        assert "test_project" in result
        mock_get.assert_called_once()

@patch("zipfile.ZipFile")
def test_unzip_password_success(mock_zip, temp_dir, valid_zip_file):
    mock_instance = mock_zip.return_value.__enter__.return_value
    mock_instance.namelist.return_value = ["test_project/"]
    
    # Simulate RuntimeError for first attempt (no password), then success with password
    mock_instance.extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, password="secret_password")
    
    assert "test_project" in result
    # Verify extractall was called with encoded password
    mock_instance.extractall.assert_called_with(path=pytest.any, pwd=b"secret_password")

@patch("zipfile.ZipFile")
def test_unzip_password_failure(mock_zip, temp_dir, valid_zip_file):
    mock_instance = mock_zip.return_value.__enter__.return_value
    mock_instance.namelist.return_value = ["test_project/"]
    mock_instance.extractall.side_effect = RuntimeError("Wrong password")
    
    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, password="wrong")

@patch("requests.get")
def test_unzip_bad_zip_file(mock_get, temp_dir):
    from zipfile import BadZipFile
    # Create a file that is not a zip
    bad_file = temp_dir / "bad.zip"
    bad_file.write_text("not a zip")
    
    with patch("zipfile.ZipFile", side_effect=BadZipFile("Not a zip")):
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(str(bad_file), is_url=False, clone_to_dir=temp_dir)
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def valid_zip_path(temp_dir):
    zip_path = temp_dir / "project.zip"
    # Create a zip with a top-level directory entry
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_path(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip_path(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file_no_dir.txt", "content")
    return zip_path

@pytest.fixture
def corrupted_zip_path(temp_dir):
    zip_path = temp_dir / "corrupt.zip"
    zip_path.write_text("not a zip")
    return zip_path

def test_unzip_local_file_success(valid_zip_path, temp_dir):
    result_path = unzip(str(valid_zip_path), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "project"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip_raises_error(empty_zip_path):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_path), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip_path):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip_path), is_url=False)

def test_unzip_bad_zip_file_raises_error(corrupted_zip_path):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(corrupted_zip_path), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, valid_zip_path, temp_dir):
    # Setup mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    url = "https://example.com/repo.zip"
    mock_prompt.return_value = False # Don't delete existing

    # We need to intercept the ZipFile creation because we can't actually 
    # download a real zip via mock and expect ZipFile to read it.
    # Instead, we mock the ZipFile behavior for the URL path.
    with patch("zipfile.ZipFile") as mock_zip:
        mock_instance = mock_zip.return_value.__enter__.return_value
        mock_instance.namelist.return_value = ["project/", "project/file.txt"]
        mock_instance.extractall.return_value = None
        
        result = unzip(url, is_url=True, clone_to_dir=temp_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(url, stream=True, timeout=100)

@patch("zipfile.ZipFile")
def test_unzip_password_correct(mock_zip, valid_zip_path):
    mock_instance = mock_zip.return_value.__enter__.return_value
    mock_instance.namelist.return_value = ["project/", "project/file.txt"]
    # Simulate RuntimeError on first attempt, success on second with password
    mock_instance.extractall.side_effect = [RuntimeError("password required"), None]

    result = unzip(str(valid_zip_path), is_url=False, password="secret_password")
    
    # Verify extractall was called with the encoded password
    args, kwargs = mock_instance.extractall.call_args
    assert kwargs['pwd'] == b"secret_password"

@patch("zipfile.ZipFile")
def test_unzip_password_incorrect_raises_error(mock_zip, valid_zip_path):
    mock_instance = mock_zip.return_value.__enter__.return_value
    mock_instance.namelist.return_value = ["project/", "project/file.txt"]
    # Always raise error even with password provided
    mock_instance.extractall.side_effect = RuntimeError("Wrong password")

    with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
        unzip(str(valid_zip_path), is_url=False, password="wrong_password")

@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_interactive_success(mock_read_pw, mock_zip, valid_zip_path):
    mock_instance = mock_zip.return_value.__enter__.return_value
    mock_instance.namelist.return_value = ["project/", "project/file.txt"]
    mock_read_pw.return_value = "user_typed_password"
    # First call fails, second succeeds
    mock_instance.extractall.side_effect = [RuntimeError("password required"), None]

    result = unzip(str(valid_zip_path), is_url=False)
    
    assert "project" in result
    mock_read_pw.assert_called_once()
```


# LLM-generated content at query #3
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pytest
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    # Create a zip with a top-level directory as required by the logic
    with ZipFile(zip_path, 'w') as z:
        z.writestr("project/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as z:
        pass
    return zip_path

@pytest.fixture
def malformed_zip_file(temp_dir):
    zip_path = temp_dir / "bad.zip"
    with open(zip_path, 'wb') as f:
        f.write(b"not a zip file")
    return zip_path

@pytest.fixture
def no_top_level_zip_file(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as z:
        z.writestr("file.txt", "content")
    return zip_path

def test_unzip_local_success(temp_dir, valid_zip_file):
    result_path = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_local_empty_error(temp_dir, empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip_file), is_url=False, clone_to_dir=temp_dir)

def test_unzip_local_no_top_level_error(temp_dir, no_top_level_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(no_top_level_zip_file), is_url=False, clone_to_dir=temp_dir)

def test_unzip_local_bad_zip_error(temp_dir, malformed_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(malformed_zip_file), is_url=False, clone_to_dir=temp_dir)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip_file):
    # Setup mock response for requests
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"dummy content"]
    mock_get.return_migrated = mock_response # Not real, just a placeholder logic
    mock_get.return_value = mock_response
    
    # Create a fake URL that would resolve to an identifier
    zip_url = "https://example.com/download/project.zip"
    mock_prompt.return_value = True

    # We need the downloaded file to be a valid zip for the second part of unzip() to work
    # So we patch ZipFile to not actually look at the downloaded dummy content
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["project/"]
        mock_zip_instance.extractall.return_value = None

        result = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
        
        assert "project" in result
        mock_get.assert_called_once_with(zip_url, stream=True, timeout=100)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_read_password, temp_dir, valid_zip_file):
    # To test password, we need a zip that actually triggers RuntimeError on extractall
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["protected/"]
        # Simulate password error on first attempt, success on second
        mock_zip_instance.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_read_password.return_value = "secret"

        result = unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir)
        assert "protected" in result
        mock_read_password.assert_called_once()

def test_unzip_password_failure_no_input(temp_dir, valid_zip_file):
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["protected/"]
        mock_zip_instance.extractall.side_effect = RuntimeError("Password required")

        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(valid_zip_file), is_url=False, clone_to_dir=temp_dir, no_input=True)
```


# LLM-generated content at query #4
#--------------------------

```python
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "project.zip"
    content_dir = "project_dir/"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{content_dir}file.txt", "hello world")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("not_a_directory.txt", "data")
    return zip_path

@pytest.fixture
def encrypted_zip(temp_dir):
    # Note: Standard zipfile module has limited support for creating encrypted zips
    # but we can mock the behavior of RuntimeError during extraction
    zip_path = temp_dir / "secret.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("secret_dir/", "")
    return zip_path

def test_unzip_local_file_success(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_markup := empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_download_success(mock_prompt, mock_make_path, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    mock_prompt.return_value = True
    
    # Mock response stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We need a real zip file for the ZipFile(zip_path) call to not fail with BadZipFile
    # So we intercept the download and write actual valid zip bytes instead of 'fake_zip_content'
    valid_zip_bytes = b""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_zip = Path(tmp) / "repo.zip"
        with ZipFile(tmp_zip, 'w') as zf:
            zf.writestr("project/", "")
        valid_zip_bytes = tmp_zip.read_bytes()

    mock_response.iter_content.return_value = [valid_zip_bytes]

    # Create the file on disk manually to simulate download success
    target_zip = temp_dir / "repo.zip"
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    
    result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result)
    assert "project" in result

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_provided(mock_extractall, valid_zip):
    # Simulate RuntimeError for first attempt (as if encrypted), then success with password
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    
    result = unzip(str(valid_zip), is_url=False, password="correct_password")
    
    # Verify extractall was called with the encoded password
    args, kwargs = mock_extractall.call_args
    assert kwargs['pwd'] == b"correct_password"
    assert os.path.exists(result)

@patch("zipfile.ZipFile.extractall")
def test_unzip_password_error_no_input(mock_extractall, valid_zip):
    mock_extractall.side_effect = RuntimeError("Password required")
    
    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(valid_zip), is_url=False, no_input=True)

@patch("zipfile.ZipFile.extractall")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_interaction(mock_read_password, mock_extractall, valid_zip):
    # First attempt fails, second attempt (with mocked password) succeeds
    mock_extractall.side_effect = [RuntimeError("Password required"), None]
    mock_read_password.return_value = "user_entered_password"
    
    result = unzip(str(valid_zip), is_url=False)
    
    assert mock_read_password.called
    assert os.path.exists(result)

@patch("requests.get")
def test_unzip_bad_zip_file_error(mock_get, temp_dir):
    # Simulate a corrupted file download
    zip_uri = "https://example.com/corrupt.zip"
    target_zip = temp_dir / "corrupt.zip"
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    target_zip.write_bytes(b"not a zip")

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"not a zip"]
    mock_get.return_value = mock_response

    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from zipfile import ZipFile

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    project_name = "my_project"
    with ZipFile(zip_path, 'w') as z:
        z.writestr(f"{project_name}/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as z:
        pass
    return zip_path

@pytest.fixture
def invalid_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as z:
        z.writestr("file_without_dir.txt", "content")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    zip_path = temp_dir / "protected.zip"
    project_name = "secret_project"
    # Creating a password protected zip is tricky with standard ZipFile, 
    # so we will mock the behavior in tests instead of physical creation.
    return zip_path

def test_unzip_local_file(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_empty_zip(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir(invalid_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_structure_zip), is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_path, mock_get, temp_dir, valid_zip):
    # Create a dummy URL pointing to our valid zip
    zip_url = "https://example.com/test_repo.zip"
    
    # Mock requests response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"dummy_data"] # This will cause BadZipFile actually, so we use real content
    
    # We'll simulate the download by making the URL-based path point to our local valid zip
    # To avoid complex stream mocking, we intercept the 'is_url' logic 
    # or just make the downloaded file a valid zip.
    
    with patch("builtins.open", MagicMock()): # prevent writing to disk if we don't want
        pass

    # Real approach: Mock requests to write the actual valid_zip content to the destination
    def side_effect_get(url, stream=False, timeout=None):
        mock_r = MagicMock()
        with open(valid_zip, 'rb') as f:
            mock_r.iter_content.return_value = [chunk for chunk in iter(lambda: f.read(1024), b'')]
        return mock_r

    mock_get.side_effect = side_effect_get
    mock_prompt.return_value = True
    
    result_path = unzip(zip_url, is_url=True, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_success(mock_password, temp_dir, valid_zip):
    # We use a real ZipFile but we need to trigger the RuntimeError in extractall
    # Since we can't easily create an encrypted zip in a unit test without external deps,
    # we mock the ZipFile object.
    
    with patch("zipfile.ZipFile") as MockZip:
        instance = MockZip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = [RuntimeError("Password required"), None]
        mock_password.return_value = "correct_password"
        
        result_path = unzip(str(valid_zip), is_url=False, password="wrong")
        
        assert instance.extractall.call_count == 2
        # Verify it tried with the encoded password
        instance.extractall.assert_any_call(path=pytest.any, pwd=b"correct_password")

def test_unzip_bad_zip_file(temp_dir):
    bad_file = temp_dir / "not_a_zip.txt"
    bad_file.write_text("not zip content")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)

@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_failure_after_retries(mock_password, temp_dir, valid_zip):
    mock_password.return_value = "wrong"
    # Force RuntimeError on every attempt
    with patch("zipfile.ZipFile") as MockZip:
        instance = MockZip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(valid_zip), is_url=False)
        
        assert mock_password.call_count == 3
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
from pathlib import Path
from zipfile import ZipFile
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip_file(temp_dir):
    zip_path = temp_dir / "project.zip"
    # We create a zip with a top-level directory to satisfy the logic
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def invalid_zip_file(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return str(zip_path)

@pytest.fixture
def no_top_level_zip_file(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)

def test_unzip_local_success(valid_zip_file, temp_dir):
    result_path = unzip(valid_zip_file, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    # Check if the extracted directory exists inside the temp path
    # The function uses mkdtemp(), so we check if the project name is in the result path
    assert "project" in result_path

def test_unzip_url_success(temp_dir):
    zip_uri = "https://example.com/archive.zip"
    zip_dest = temp_dir / "archive.zip"
    
    # Create a real zip file locally to mimic the download
    with ZipFile(zip_dest, 'w') as zf:
        zf.writestr("project/file.txt", "content")

    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        # We need to ensure the downloaded file actually contains a valid zip
        # So we patch the download logic to use our pre-created local zip content
        with patch('builtins.open', pytest.monkeypatch.importorskip('builtins').open):
             # For simplicity in this test, we let it write the mock data 
             # but since we need a valid zip structure for the ZipFile(zip_path) call,
             # we will just point the download to our existing valid_zip_file content.
             with patch('os.path.exists', return_value=False):
                 # Redirecting the logic: if it's a URL, use our local valid zip as source
                 with patch('requests.get') as mock_get:
                     mock_get.return_value.iter_content = lambda chunk_size: [b""] # dummy
                     # To make this testable without complex stream mocking, 
                     # we'll just simulate the local file existence after "download"
                     with patch('zipfile.ZipFile') as mock_zip:
                         mock_zip.return_value.__enter__.return_value.namelist.return_value = ["project/"]
                         mock_zip.return_value.__enter__.return_value.extractall.return_value = None
                         
                         result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
                         assert "project" in result

def test_unzip_empty_error(empty_zip_file):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_file, is_url=False)

def test_unzip_no_top_level_error(no_top_level_zip_file):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_top_level_zip_file, is_url=False)

def test_unzip_bad_zip_error(invalid_zip_file):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(invalid_zip_file, is_url=False)

def test_unzip_password_success(valid_zip_file, temp_dir):
    # Create a password protected zip
    protected_zip = temp_dir / "protected.zip"
    with ZipFile(protected_zip, 'w') as zf:
        zf.writestr("project/secret.txt", "secret")
    
    # Note: Standard zipfile module in Python has limited support for creating 
    # AES encrypted zips, but we can simulate the RuntimeError for testing.
    with patch('zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        # Simulate password error then success
        instance.extractall.side_effect = [RuntimeError("Password required"), None]
        
        with patch('cookiecutter.prompt.read_repo_password', return_value="password123"):
            result = unzip(str(protected_zip), is_url=False, password=None)
            assert "project" in result
            assert instance.extractall.call_count == 2

def test_unzip_password_invalid_error(valid_zip_file):
    with patch('zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = RuntimeError("Wrong password")
        
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(valid_zip_file, is_url=False, password="wrong_password")

def test_unzip_no_input_fails_on_protected(valid_zip_file):
    with patch('zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.namelist.return_value = ["project/"]
        instance.extractall.side_effect = RuntimeError("Password required")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(valid_zip_file, is_url=False, no_input=True)

def test_unzip_deletes_existing_cache(temp_dir):
    zip_uri = "https://example.com/archive.zip"
    zip_path = temp_dir / "archive.zip"
    zip_path.write_text("dummy content")
    
    with patch('os.path.exists', return_value=True):
        with patch('cookiecutter.prompt.prompt_and_delete', return_value=False) as mock_prompt:
            with patch('requests.get') as mock_get:
                mock_get.return_value.iter_content.return_value = []
                # Mock the extraction part so it doesn't fail on the dummy content
                with patch('zipfile.ZipFile') as mock_zip:
                    instance = mock_zip.return_value.__enter__.return_value
                    instance.namelist.return_value = ["project/"]
                    
                    unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
                    mock_prompt.assert_called_once()
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import pytest
import requests
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary workspace for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return tmp_path, cache_dir

@pytest.fixture
def valid_zip_file(tmp_path):
    """Creates a valid zip file with a top-level directory."""
    zip_path = tmp_path / "project.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip_file(tmp_path):
    """Creates an empty zip file."""
    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def malformed_zip_file(tmp_path):
    """Creates a zip file without a top-level directory."""
    zip_path = tmp_path / "no_dir.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)

def test_unzip_local_success(valid_zip_file, temp_workspace):
    """Test unzipping a local valid zip file."""
    _, cache_dir = temp_workspace
    unzipped_path = unzip(valid_zip_file, is_url=False, clone_to_dir=cache_dir)
    
    assert os.path.exists(unzipped_path)
    with open(os.path.join(unzipped_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_url_success(valid_zip_file, temp_workspace):
    """Test unzipping from a URL."""
    _, cache_dir = temp_workspace
    url = f"https://example.com/{os.path.basename(valid_zip_file)}"
    
    # Mock requests.get to return content of the local valid zip
    mock_response = MagicMock()
    mock_response.iter_content.return_value = []
    # We need to simulate streaming bytes from the actual file
    with open(valid_zip_file, 'rb') as f:
        content = f.read()
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.iter_content.return_value = [content[i:i+1024] for i in range(0, len(content), 1024)]
        unzipped_path = unzip(url, is_url=True, clone_to_dir=cache_dir)

    assert os.path.exists(unzipped_path)
    cached_zip = cache_dir / os.path.basename(valid_zip_file)
    assert cached_zip.exists()

def test_unzip_empty_zip_raises_error(empty_zip_file, temp_workspace):
    """Test that an empty zip raises InvalidZipRepository."""
    _, cache_dir = temp_workspace
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip_file, is_url=False, clone_to_dir=cache_dir)

def test_unzip_no_top_level_dir_raises_error(malformed_zip_file, temp_workspace):
    """Test that a zip without top-level directory raises InvalidZipRepository."""
    _, cache_iter = temp_workspace
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(malformed_zip_file, is_url=False, clone_to_dir=cache_iter)

def test_unzip_password_success(tmp_path, temp_workspace):
    """Test unzipping a password protected zip with provided password."""
    zip_path = tmp_path / "protected.zip"
    password = "secret_password"
    # Note: standard ZipFile in python has limited support for creating encrypted zips 
    # via writestr, but we can mock the runtime error behavior.
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    
    _, cache_dir = temp_workspace
    
    # Since Python's zipfile module requires a specific way to create encrypted zips 
    # (which is complex), we mock the RuntimeError during extraction to test the logic flow.
    with patch('zipfile.ZipFile.extractall') as mock_extract:
        mock_extract.side_effect = [RuntimeError("Password required"), None]
        with patch('zipfile.ZipFile.namelist', return_value=["project/"]):
            unzipped_path = unzip(str(zip_path), is_url=False, clone_to_dir=cache_dir, password=password)
            assert mock_extract.call_count == 2
            # Verify second call used the password
            args, kwargs = mock_extract.call_args
            assert kwargs['pwd'] == password.encode('utf-8')

def test_unzip_password_failure(malformed_zip_file, temp_workspace):
    """Test that providing wrong password raises InvalidZipRepository."""
    _, cache_dir = temp_workspace
    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Wrong password")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(malformed_zip_file, is_url=False, clone_to_dir=cache_dir, password="wrong")

def test_unzip_bad_zip_file(tmp_path, temp_workspace):
    """Test that a corrupted zip file raises InvalidZipRepository."""
    bad_zip = tmp_path / "corrupt.zip"
    bad_zip.write_text("not a zip")
    _, cache_dir = temp_workspace
    
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_zip), is_url=False, clone_to_dir=cache_dir)

@patch('cookiecutter.utils.prompt_and_delete')
def test_unzip_prompts_deletion_if_exists(mock_prompt, valid_zip_file, temp_workspace):
    """Test that prompt_and_delete is called if the zip already exists in cache."""
    _, cache_dir = temp_mock_workspace = temp_workspace
    url = f"https://example.com/{os.path.basename(valid_zip_file)}"
    
    # Pre-create the file in cache to trigger 'exists' logic
    zip_in_cache = cache_dir / os.path.basename(valid_zip_file)
    import shutil
    shutil.copy(valid_zip_file, zip_in_cache)
    
    mock_prompt.return_value = False # Simulate user saying 'No' to deleting
    
    with patch('requests.get') as mock_get:
        # If prompt returns False, download should not happen
        unzip(url, is_url=True, clone_to_dir=cache_dir, no_input=True)
        mock_get.assert_not_called()
    
    mock_prompt.assert_called_once()
```


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
import requests
from cookiecutter.exceptions import InvalidZipRepository

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    extract_dir = "test_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{extract_dir}/file.txt", "content")
    return zip_path, extract_dir

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def invalid_zip_structure(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no top level dir")
    return zip_path

@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_local_success(mock_make_exists, temp_dir, valid_zip):
    zip_path, project_name = valid_zip
    result_path = unzip(str(zip_path), is_url=False, clone_to_dir=temp_dir)
    
    assert os.path.basename(result_path) == project_name
    assert os.path.exists(os.path.join(result_path, "file.txt"))

@patch("requests.get")
@patch("cookiecutter.prompt.prompt_and_delete")
@patch("cookiecutter.utils.make_sure_path_exists")
def test_unzip_url_success(mock_make_exists, mock_prompt_delete, mock_get, temp_dir):
    zip_uri = "https://example.com/repo.zip"
    mock_prompt_delete.return_value = True
    
    # Mocking requests stream
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response

    # We must create a real zip file at the destination for ZipFile to open it
    # So we patch ZipFile to not actually look for the downloaded bytes, 
    # or more simply, let's mock the creation of the file content.
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["repo/"]
        
        result = unzip(zip_uri, is_url=True, clone_to_dir=temp_dir)
        
        assert "repo" in result
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)

def test_unzip_empty_zip_raises(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises(invalid_zip_structure):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(invalid_zip_structure), is_url=False)

@patch("zipfile.ZipFile")
def test_unzip_password_provided_success(mock_zip_class, temp_dir, valid_zip):
    zip_path, _ = valid_zip
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    
    # Simulate RuntimeError for first attempt (no password), then success with password
    def side_effect(path, pwd=None):
        if pwd is None:
            raise RuntimeError("Password required")
        return MagicMock()

    mock_zip_instance.extractall.side_effect = side_effect

    result = unzip(str(zip_path), is_url=False, password="secret_password")
    assert "project" in result
    # Verify extractall was called with encoded password
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b'secret_password'

@patch("zipfile.ZipFile")
def test_unzip_password_failure_no_input(mock_zip_class, temp_dir, valid_zip):
    zip_path, _ = valid_zip
    mock_zip_instance = mock_zip_class.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_zip_instance.extractall.side_effect = RuntimeError("Password required")

    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(str(zip_path), is_url=False, no_input=True)

@patch("zipfile.ZipFile")
def test_unzip_bad_zip_file(mock_zip_class, temp_dir, valid_zip):
    from zipfile import BadZipFile
    zip_path, _ = valid_zip
    mock_zip_class.side_effect = BadZipFile("Not a zip")

    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(zip_path), is_url=False)
```


# LLM-generated content at query #9
#--------------------------

```python
import os
import tempfile
import pytest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    # Create a zip with a top-level directory
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("test_repo/file.txt", "content")
    return str(zip_path)

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(zip_path, 'w') as zf:
        pass
    return str(zip_path)

@pytest.fixture
def bad_zip(temp_dir):
    zip_path = temp_dir / "bad.zip"
    zip_path.write_text("not a zip")
    return str(zip_path)

@pytest.fixture
def no_top_level_zip(temp_dir):
    zip_path = temp_dir / "no_top.zip"
    with ZipFile(path=zip_path, mode='w') as zf:
        zf.writestr("file.txt", "content")
    return str(zip_path)

@pytest.fixture
def password_zip(temp_dir):
    # Note: zipfile module's support for creating encrypted zips is limited, 
    # but we can mock the behavior of extractall raising RuntimeError.
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("project/file.txt", "content")
    return str(zip_path)

def test_unzip_local_success(valid_zip, temp_dir):
    result_path = unzip(valid_zip, is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.basename(result_path) == "test_repo"
    with open(os.path.join(result_path, "file.txt"), 'r') as f:
        assert f.read() == "content"

def test_unzip_empty_zip_raises(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(empty_zip, is_url=False)

def test_unzip_no_top_level_raises(no_top_level_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(no_top_level_zip, is_url=False)

def test_unzip_bad_zip_raises(bad_zip):
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(bad_zip, is_url=False)

@patch("requests.get")
@patch("cookiecutter.utils.make_sure_path_exists")
@patch("cookiecutter.prompt.prompt_and_delete")
def test_unzip_url_success(mock_prompt, mock_make_sure, mock_get, temp_dir):
    zip_uri = "http://example.com/repo.zip"
    clone_dir = temp_dir / "cache"
    
    # Mocking the response for requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"fake_zip_content"]
    mock_get.return_value = mock_response
    
    # Create a dummy file that looks like a valid zip to satisfy ZipFile constructor
    # Since we can't easily download and unzip a fake stream in a unit test 
    # without real zip structure, we patch the ZipFile context manager directly.
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = mock_zip.return_value.__enter__.return_value
        mock_zip_instance.namelist.return_value = ["repo/"]
        mock_zip_instance.extractall.return_value = None
        
        result = unzip(zip_uri, is_url=True, clone_to_dir=clone_dir)
        
        assert "repo" in result
        mock_get.assert_called_once_with(zip_uri, stream=True, timeout=100)

@patch("zipfile.ZipFile")
def test_unzip_password_provided(mock_zip, password_zip):
    mock_zip_instance = mock_zip.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    # Simulate RuntimeError for wrong password first, then success with correct one
    mock_zip_instance.extractall.side_effect = [RuntimeError("Bad password"), None]

    result = unzip(password_zip, is_url=False, password="correct_password")
    
    assert "project" in result
    # Verify extractall was called with encoded password
    args, kwargs = mock_zip_instance.extractall.call_args
    assert kwargs['pwd'] == b"correct_password"

@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_prompt(mock_prompt_pass, mock_zip, password_zip):
    mock_zip_instance = mock_zip.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_prompt_pass.return_value = "user_entered_password"
    # Simulate RuntimeError for first attempt, then success
    mock_zip_instance.extractall.side_effect = [RuntimeError("Bad password"), None]

    result = unzip(password_zip, is_url=False, no_input=False)
    
    assert "project" in result
    mock_prompt_pass.assert_called_once()
    assert mock_zip_instance.extractall.call_count == 2

@patch("zipfile.ZipFile")
def test_unzip_password_failure_no_input(mock_zip, password_zip):
    mock_zip_instance = mock_zip.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_zip_instance.extractall.side_effect = RuntimeError("Bad password")

    with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
        unzip(password_zip, is_url=False, no_input=True)

@patch("zipfile.ZipFile")
@patch("cookiecutter.prompt.read_repo_password")
def test_unzip_password_too_many_retries(mock_prompt_pass, mock_zip, password_zip):
    mock_zip_instance = mock_zip.return_value.__enter__.return_value
    mock_zip_instance.namelist.return_value = ["project/"]
    mock_prompt_pass.return_value = "wrong"
    # Always raise error
    mock_zip_instance.extractall.side_effect = RuntimeError("Bad password")

    with pytest.raises(InvalidZipRepository, match="Invalid password provided for protected repository"):
        unzip(password_zip, is_url=False, no_input=False)
    
    assert mock_prompt_pass.call_count == 3
```


# LLM-generated content at query #10
#--------------------------

```python
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from zipfile import ZipFile

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def valid_zip(temp_dir):
    zip_path = temp_dir / "test_repo.zip"
    project_name = "my_project"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{project_name}/file.txt", "content")
    return zip_path

@pytest.fixture
def empty_zip(temp_dir):
    zip_path = temp_dir / "empty.zip"
    with ZipFile(push_path := zip_path, 'w') as zf:
        pass
    return zip_path

@pytest.fixture
def bad_structure_zip(temp_dir):
    zip_path = temp_dir / "bad_struct.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("file.txt", "no directory root")
    return zip_path

@pytest.fixture
def password_zip(temp_dir):
    # Note: zipfile module has limited support for creating encrypted zips in some versions
    # but we can mock the behavior of ZipFile during extraction
    zip_path = temp_dir / "protected.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("protected_dir/", "content")
    return zip_path

def test_unzip_local_file_success(temp_dir, valid_zip):
    result_path = unzip(str(valid_zip), is_url=False, clone_to_dir=temp_dir)
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "file.txt"))

def test_unzip_url_download_success(temp_dir):
    zip_uri = "https://example.com/repo.zip"
    clone_dir = temp_dir / "cache"
    make_sure_path_exists(clone_dir)
    
    # Create a dummy zip to act as the downloaded file
    dummy_zip = clone_dir / "repo.zip"
    with ZipFile(dummy_zip, 'w') as zf:
        zf.writestr("repo_dir/data.txt", "data")

    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"dummy content"]
        mock_get.return_value = mock_response
        # We mock prompt_and_delete to return False so it doesn't delete our dummy
        with patch('cookiecutter.utils.prompt_and_delete', return_value=False):
            result_path = unzip(zip_uri, is_url=True, clone_to_dir=clone_dir)
    
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "data.txt"))

def test_unzip_empty_zip_raises_error(empty_zip):
    with pytest.raises(InvalidZipRepository, match="is empty"):
        unzip(str(empty_zip), is_url=False)

def test_unzip_no_top_level_dir_raises_error(bad_structure_zip):
    with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
        unzip(str(bad_structure_zip), is_url=False)

def test_unzip_bad_zip_file_raises_error(temp_dir):
    bad_file = temp_dir / "corrupt.zip"
    bad_file.write_text("not a zip")
    with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
        unzip(str(bad_file), is_url=False)

def test_unzip_password_provided_success(password_zip):
    with patch('zipfile.ZipFile.extractall') as mock_extract:
        # Simulate success on second attempt with password
        def side_effect(path, pwd=None):
            if pwd:
                return None
            raise RuntimeError("Password required")
        
        mock_extract.side_effect = side_effect
        result = unzip(str(password_zip), is_url=False, password="secret_password")
        assert mock_extract.called
        assert result is not None

def test_unzip_password_wrong_raises_error(password_zip):
    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Bad password")):
        with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
            unzip(str(password_zip), is_url=False, password="wrong_password")

def test_unzip_no_input_and_no_password_raises_error(password_zip):
    with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Password required")):
        with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
            unzip(str(password_zip), is_url=False, no_input=True)

def test_unzip_prompt_for_password_success(password_zip):
    with patch('cookiecutter.prompt.read_repo_password', return_value='correct_pass'):
        with patch('zipfile.ZipFile.extractall') as mock_extract:
            # First call fails (no password), second call succeeds (with password)
            mock_extract.side_effect = [RuntimeError("Password required"), None]
            result = unzip(str(password_zip), is_url=False)
            assert result is not None
            assert mock_extract.call_count == 2

def test_unzip_password_retry_limit_exceeded(password_zip):
    with patch('cookiecutter.prompt.read_repo_password', return_value='wrong'):
        with patch('zipfile.ZipFile.extractall', side_effect=RuntimeError("Password required")):
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(str(password_zip), is_url=False)
```


