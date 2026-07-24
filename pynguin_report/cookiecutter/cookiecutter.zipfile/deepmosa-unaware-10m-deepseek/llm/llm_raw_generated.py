####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open, call
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("test_project/", "")
            zipf.writestr("test_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "test_project" in result
        assert Path(result).exists()
    
    # Test 2: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zipf:
            zipf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: URL download with mock
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False), \
         patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_get.return_value = mock_response
        
        # Create a mock zip file in memory
        zip_content = b"PK\x03\x04" + b" " * 100  # Minimal zip header
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.write = Mock()
            
            # Mock the zip file extraction
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                
                with tempfile.TemporaryDirectory() as extract_dir:
                    mock_zip_instance.extractall.return_value = None
                    
                    result = unzip(
                        "http://example.com/test.zip",
                        is_url=True,
                        clone_to_dir=tmpdir,
                        no_input=True
                    )
                    
                    mock_get.assert_called_once_with(
                        "http://example.com/test.zip",
                        stream=True,
                        timeout=100
                    )
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        # Create password protected zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("test_project/", "")
            zipf.setpassword(b"secret")
        
        # Reopen with password to actually write encrypted content
        with zipfile.ZipFile(zip_path, 'a') as zipf:
            zipf.setpassword(b"secret")
            zipf.writestr("test_project/file.txt", "protected content")
        
        with patch('cookiecutter.prompt.read_repo_password', return_value="secret"):
            result = unzip(
                str(zip_path),
                is_url=False,
                clone_to_dir=tmpdir,
                no_input=False,
                password=None
            )
            assert "test_project" in result
    
    # Test 7: Password protected zip with no_input=True should raise exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected2.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("test_project/", "")
            zipf.setpassword(b"secret")
        
        with zipfile.ZipFile(zip_path, 'a') as zipf:
            zipf.setpassword(b"secret")
            zipf.writestr("test_project/file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(
                str(zip_path),
                is_url=False,
                clone_to_dir=tmpdir,
                no_input=True,
                password=None
            )
    
    # Test 8: Invalid password raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected3.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("test_project/", "")
            zipf.setpassword(b"secret")
        
        with zipfile.ZipFile(zip_path, 'a') as zipf:
            zipf.setpassword(b"secret")
            zipf.writestr("test_project/file.txt", "content")
        
        with patch('cookiecutter.prompt.read_repo_password', side_effect=["wrong", "wrong", "wrong"]):
            with pytest.raises(InvalidZipRepository):
                unzip(
                    str(zip_path),
                    is_url=False,
                    clone_to_dir=tmpdir,
                    no_input=False,
                    password=None
                )
    
    # Test 9: Existing cached file with no_input=False prompts for deletion
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=True) as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('zipfile.ZipFile') as mock_zip:
            
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.extractall.return_value = None
            
            result = unzip(
                "http://example.com/existing.zip",
                is_url=True,
                clone_to_dir=tmpdir,
                no_input=False
            )
            
            mock_prompt.assert_called_once()
    
    # Test 10: Clone_to_dir expansion with home directory
    with patch('pathlib.Path.expanduser', return_value=Path("/expanded/path")) as mock_expand, \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path, \
         patch('os.path.exists', return_value=False), \
         patch('requests.get') as mock_get, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = []
        mock_get.return_value = mock_response
        
        with patch('builtins.open', mock_open()), \
             patch('zipfile.ZipFile') as mock_zip:
            
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["test/", "test/file.txt"]
            mock_zip_instance.extractall.return_value = None
            
            unzip(
                "http://example.com/test.zip",
                is_url=True,
                clone_to_dir="~/some/path",
                no_input=True
            )
            
            mock_expand.assert_called_once()
            mock_make_path.assert_called_once()


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import os

    # Test 1: Local zip file extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert os.path.exists(os.path.join(result, "file.txt"))

    # Test 2: URL zip file download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
                mock_zip_instance.__exit__ = Mock(return_value=None)
                mock_zip.return_value = mock_zip_instance
                
                with patch('tempfile.mkdtemp') as mock_mkdtemp:
                    mock_mkdtemp.return_value = tmpdir
                    
                    result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
                    assert mock_get.called
                    assert "project" in result

    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        try:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "empty" in str(e)

    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "flat.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        try:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "top-level directory" in str(e)

    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.setpassword(b"secret")
        
        # Mock the zip extraction to simulate password protection
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
            mock_zip_instance.__exit__ = Mock(return_value=None)
            
            # First extractall raises RuntimeError (password needed)
            # Second succeeds with password
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None
            ]
            mock_zip.return_value = mock_zip_instance
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2

    # Test 6: Invalid zip file raises BadZipFile
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_bytes(b"not a zip file")
        
        try:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
            assert False, "Should have raised InvalidZipRepository"
        except InvalidZipRepository as e:
            assert "not a valid zip archive" in str(e)

    # Test 7: Clone directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new" / "nested" / "dir"
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=new_dir)
        assert new_dir.exists()

    # Test 8: URL zip with existing cached file and no_input=True
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"new content"]
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            mock_get.return_value = mock_response
            
            # Create existing cached file
            cached_file = Path(tmpdir) / "repo.zip"
            cached_file.write_bytes(b"old content")
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["project/"]
                mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
                mock_zip_instance.__exit__ = Mock(return_value=None)
                mock_zip.return_value = mock_zip_instance
                
                with patch('tempfile.mkdtemp') as mock_mkdtemp:
                    mock_mkdtemp.return_value = tmpdir
                    
                    # With no_input=True, should refresh without prompting
                    result = unzip("http://example.com/repo.zip", is_url=True, 
                                  clone_to_dir=tmpdir, no_input=True)
                    assert mock_get.called


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open, call
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "test_project" in result
        assert Path(result).exists()
    
    # Test 2: Invalid zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_bytes(b"not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository, match="empty"):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository, match="top-level directory"):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
        assert "project" in result
    
    # Test 6: Password protected zip with wrong password raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository, match="Invalid password"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong")
    
    # Test 7: URL download with mock
    with patch('requests.get') as mock_get, \
         patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_get.return_value = mock_response
        
        mock_prompt.return_value = True
        
        # Create a mock zip file in memory
        import io
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr("url_project/", "")
            zf.writestr("url_project/file.txt", "content")
        zip_data = zip_buffer.getvalue()
        
        # Mock the file writing
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.write.side_effect = None
            
            result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
            
            # Verify requests was called correctly
            mock_get.assert_called_once_with("http://example.com/repo.zip", stream=True, timeout=100)
            
            # Verify file was written
            assert mock_file.return_value.write.call_count == 2
    
    # Test 8: URL download with existing file and no_input=False
    with patch('requests.get') as mock_get, \
         patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_prompt.return_value = False  # User chooses not to download
        
        result = unzip("http://example.com/existing.zip", is_url=True, clone_to_dir=tmpdir, no_input=False)
        
        # Should not call requests.get if user chooses not to download
        mock_get.assert_not_called()
    
    # Test 9: Password protected with no_input=True and no password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository, match="Unable to unlock"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 10: Clone_to_dir expansion with tilde
    with patch('pathlib.Path.expanduser') as mock_expanduser, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_expanduser.return_value = Path(tmpdir)
        
        # Create a test zip
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("tilde_project/", "")
            zf.writestr("tilde_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir="~/some/path")
        mock_expanduser.assert_called_once()


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import pytest
    from cookiecutter.exceptions import InvalidZipRepository
    import zipfile
    import os

    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert os.path.exists(result)
        assert os.path.exists(os.path.join(result, "file.txt"))

    # Test 2: Invalid zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 3: Empty zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/secret.txt", "confidential")
            zf.setpassword(b"secret")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
        assert "project" in result

    # Test 6: Password protected zip with wrong password raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/secret.txt", "confidential")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong")

    # Test 7: URL download with mock
    with patch('requests.get') as mock_get, \
         patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_get.return_value = mock_response
        
        mock_prompt.return_value = True
        
        # Create a mock zip file in memory
        import io
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr("webproject/", "")
            zf.writestr("webproject/index.html", "<html></html>")
        zip_data = zip_buffer.getvalue()
        
        # Mock the file writing
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.write = Mock()
            
            # Mock the zip file reading
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["webproject/", "webproject/index.html"]
                
                # Create a real temporary directory for extraction
                with tempfile.TemporaryDirectory() as extract_dir:
                    mock_zip_instance.extractall = Mock()
                    
                    result = unzip(
                        "http://example.com/repo.zip",
                        is_url=True,
                        clone_to_dir=tmpdir
                    )
                    
                    mock_get.assert_called_once_with("http://example.com/repo.zip", stream=True, timeout=100)
                    mock_prompt.assert_called_once()

    # Test 8: No input mode with password protected zip raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)

    # Test 9: Clone directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "newdir"
        zip_path = Path(tmpdir) / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/test.txt", "test")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=str(clone_dir))
        assert clone_dir.exists()
        assert "project" in result

    # Test 10: Relative path handling
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        zip_path = Path(tmpdir) / "relative.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("relativeproject/", "")
            zf.writestr("relativeproject/file.txt", "data")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=".")
        assert "relativeproject" in result
        assert os.path.exists(os.path.join(result, "file.txt"))


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open
    from cookiecutter.exceptions import InvalidZipRepository

    # Test 1: Successful extraction from local zip file
    def test_local_zip_extraction():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = Path(tmpdir) / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("project/", "")
                zf.writestr("project/file.txt", "content")
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
            assert "project" in result
            assert Path(result).exists()
            assert (Path(result) / "file.txt").exists()

    # Test 2: URL download and extraction
    def test_url_zip_extraction():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the requests response
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            
            # Create a mock zip file in memory
            zip_content = b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00project/\x00\x00\x00"
            
            with patch('requests.get') as mock_get, \
                 patch('builtins.open', mock_open()) as mock_file:
                mock_get.return_value = mock_response
                mock_response.iter_content.return_value = [zip_content]
                
                # Mock the zip file extraction
                with patch('zipfile.ZipFile') as mock_zip:
                    mock_zip_instance = Mock()
                    mock_zip.return_value.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                    
                    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir)
                    assert mock_get.called
                    assert mock_file.called

    # Test 3: Empty zip file raises exception
    def test_empty_zip_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "empty.zip"
            with zipfile.ZipFile(zip_path, 'w'):
                pass  # Create empty zip
            
            with pytest.raises(InvalidZipRepository, match="is empty"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)

    # Test 4: Zip without top-level directory raises exception
    def test_no_top_level_dir_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "flat.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file.txt", "content")
            
            with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)

    # Test 5: Password protected zip with correct password
    def test_password_protected_zip():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "protected.zip"
            
            # Create password protected zip
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("project/", "")
                zf.setpassword(b"secret")
            
            # Mock the zip extraction to simulate password protection
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                
                # First extractall raises RuntimeError (password required)
                mock_zip_instance.extractall.side_effect = [
                    RuntimeError("password required"),
                    None  # Success with password
                ]
                
                result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
                assert mock_zip_instance.extractall.call_count == 2

    # Test 6: Invalid zip file raises BadZipFile exception
    def test_invalid_zip_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "invalid.zip"
            zip_path.write_text("not a zip file")
            
            with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)

    # Test 7: Existing cached file with no_input=True forces re-download
    def test_existing_cache_no_input():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test.zip"
            cache_file.write_bytes(b"cached content")
            
            with patch('requests.get') as mock_get, \
                 patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                
                unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                
                # Should download despite existing cache when no_input=True
                assert mock_get.called

    # Run all tests
    test_local_zip_extraction()
    test_url_zip_extraction()
    test_empty_zip_raises_exception()
    test_no_top_level_dir_raises_exception()
    test_password_protected_zip()
    test_invalid_zip_raises_exception()
    test_existing_cache_no_input()


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import pytest
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "test_project" in result
        assert Path(result).exists()
    
    # Test 2: URL download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock_zip_content"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                assert mock_get.called
                assert "test_project" in result
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "empty" in str(exc_info.value).lower()
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "top-level directory" in str(exc_info.value)
    
    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        # Create password protected zip (simulated)
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.extractall.side_effect = [RuntimeError(), None]  # First fails, second succeeds
            mock_zip.return_value = mock_zip_instance
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2
    
    # Test 6: Invalid zip file raises BadZipFile
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "not a valid zip archive" in str(exc_info.value)
    
    # Test 7: Existing cached file with no_input=True forces re-download
    with tempfile.TemporaryDirectory() as tmpdir:
        cached_file = Path(tmpdir) / "repo.zip"
        cached_file.write_bytes(b"existing content")
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"new_content"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                assert mock_get.called
    
    # Test 8: Ensure clone_to_dir is created if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_directory"
        zip_path = Path(tmpdir) / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=new_dir)
        assert new_dir.exists()
        assert "project" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open, call
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()
    
    # Test 2: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zipf:
            zipf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: URL download with mocked requests
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False), \
         patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Mock the zip file operations
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            
            result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
            
            mock_get.assert_called_once_with("http://example.com/repo.zip", stream=True, timeout=100)
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch('cookiecutter.prompt.read_repo_password') as mock_read_pass:
        
        zip_path = Path(tmpdir) / "protected.zip"
        password = b"secret"
        
        # Create password protected zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            # Set password (this is a simplified version)
            zipf.setpassword(password)
        
        # Test with password provided
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
        assert "project" in result
    
    # Test 7: Password protected zip with wrong password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        # Create a mock zip that raises RuntimeError for password
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            
            # First extractall raises RuntimeError (password needed)
            # Second with wrong password also raises RuntimeError
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                RuntimeError("bad password")
            ]
            
            with pytest.raises(InvalidZipRepository):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong")
    
    # Test 8: Test with no_input=True and password protected
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch('zipfile.ZipFile') as mock_zip:
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
        mock_zip_instance.extractall.side_effect = RuntimeError("password required")
        
        with pytest.raises(InvalidZipRepository):
            unzip("test.zip", is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 9: Test existing cached file with prompt to delete
    with patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=True) as mock_prompt, \
         patch('requests.get') as mock_get, \
         patch('zipfile.ZipFile') as mock_zip, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1"]
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
        
        result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
        
        mock_prompt.assert_called_once()
    
    # Test 10: Test existing cached file with no download (prompt returns False)
    with patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=False) as mock_prompt, \
         patch('zipfile.ZipFile') as mock_zip, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
        
        result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
        
        mock_prompt.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import pytest
    
    # Test 1: Local zip file extraction
    def test_local_zip_extraction():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = Path(tmpdir) / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("test_project/", "")
                zf.writestr("test_project/file.txt", "content")
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
            assert "test_project" in result
            assert Path(result).exists()
    
    # Test 2: URL zip file download and extraction
    def test_url_zip_extraction():
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('requests.get') as mock_get:
                # Mock response with zip content
                mock_response = Mock()
                mock_response.iter_content.return_value = [b"mock zip content"]
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                # Mock zip file creation
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('zipfile.ZipFile') as mock_zip:
                        mock_zip_instance = Mock()
                        mock_zip_instance.__enter__.return_value = mock_zip_instance
                        mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                        mock_zip.return_value = mock_zip_instance
                        
                        result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir)
                        assert mock_get.called
                        assert mock_file.called
    
    # Test 3: Empty zip file raises exception
    def test_empty_zip_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "empty.zip"
            with zipfile.ZipFile(zip_path, 'w'):
                pass
            
            with pytest.raises(InvalidZipRepository, match="is empty"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    def test_zip_no_top_level_dir_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "bad.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file.txt", "content")
            
            with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Password protected zip with correct password
    def test_password_protected_zip():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "protected.zip"
            
            # Create password protected zip
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("test_project/", "")
                zf.setpassword(b"secret")
                zf.writestr("test_project/file.txt", "content")
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert "test_project" in result
    
    # Test 6: Invalid zip file raises BadZipFile exception
    def test_invalid_zip_raises_exception():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "invalid.zip"
            zip_path.write_text("not a zip file")
            
            with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 7: Existing cached file with no_input=True forces refresh
    def test_existing_cache_no_input():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test.zip"
            cache_file.write_bytes(b"existing cache")
            
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.iter_content.return_value = [b"new content"]
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                with patch('zipfile.ZipFile') as mock_zip:
                    mock_zip_instance = Mock()
                    mock_zip_instance.__enter__.return_value = mock_zip_instance
                    mock_zip_instance.namelist.return_value = ["test_project/"]
                    mock_zip.return_value = mock_zip_instance
                    
                    unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                    assert mock_get.called
    
    # Test 8: Password prompt when no password provided
    def test_password_prompt():
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "protected.zip"
            
            # Create password protected zip
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("test_project/", "")
                zf.setpassword(b"secret")
                zf.writestr("test_project/file.txt", "content")
            
            with patch('cookiecutter.prompt.read_repo_password', return_value="secret"):
                result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=False)
                assert "test_project" in result
    
    # Run all tests
    test_local_zip_extraction()
    test_url_zip_extraction()
    test_empty_zip_raises_exception()
    test_zip_no_top_level_dir_raises_exception()
    test_password_protected_zip()
    test_invalid_zip_raises_exception()
    test_existing_cache_no_input()
    test_password_prompt()


# LLM-generated content at query #9
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import pytest
    
    # Test 1: Successful extraction from URL
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir='.')
        
        assert result == '/tmp/tempdir/project'
        mock_make_path.assert_called_once()
        mock_get.assert_called_once_with('http://example.com/repo.zip', stream=True, timeout=100)
        mock_zip_instance.extractall.assert_called_once_with(path='/tmp/tempdir')
    
    # Test 2: Successful extraction from local file
    with patch('os.path.abspath', return_value='/local/repo.zip'), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path:
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        result = unzip('/local/repo.zip', is_url=False, clone_to_dir='.')
        
        assert result == '/tmp/tempdir/project'
        mock_make_path.assert_called_once()
        mock_zip_instance.extractall.assert_called_once_with(path='/tmp/tempdir')
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = []
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        
        assert 'Zip repository' in str(exc_info.value)
        assert 'empty' in str(exc_info.value)
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        
        assert 'Zip repository' in str(exc_info.value)
        assert 'top-level directory' in str(exc_info.value)
    
    # Test 5: Password protected zip with provided password
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        
        def extractall_side_effect(path=None, pwd=None):
            if pwd is None:
                raise RuntimeError("Password required")
        
        mock_zip_instance.extractall.side_effect = extractall_side_effect
        mock_zip.return_value = mock_zip_instance
        
        result = unzip('http://example.com/repo.zip', is_url=True, password='secret')
        
        assert result == '/tmp/tempdir/project'
        mock_zip_instance.extractall.assert_called_with(
            path='/tmp/tempdir', pwd=b'secret'
        )
    
    # Test 6: Invalid zip file raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile', side_effect=zipfile.BadZipFile("Bad zip")), \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        
        assert 'Zip repository' in str(exc_info.value)
        assert 'not a valid zip archive' in str(exc_info.value)
    
    # Test 7: Existing cached file with no_input=True forces re-download
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=True), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'chunk1']
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        result = unzip('http://example.com/repo.zip', is_url=True, no_input=True)
        
        assert result == '/tmp/tempdir/project'
        mock_get.assert_called_once()
    
    # Test 8: Path objects as clone_to_dir parameter
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('builtins.open', mock_open()), \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path:
        
        mock_zip_instance = Mock()
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        mock_zip_instance.__enter__.return_value = mock_zip_instance
        mock_zip.return_value = mock_zip_instance
        
        clone_dir = Path('/custom/dir')
        result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir=clone_dir)
        
        assert result == '/tmp/tempdir/project'
        mock_make_path.assert_called_once_with(clone_dir.expanduser())


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest import mock
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()
    
    # Test 2: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zipf:
            zipf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            zipf.setpassword(b"secret")
        
        # Mock read_repo_password to return the password
        with mock.patch('cookiecutter.prompt.read_repo_password', return_value="secret"):
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=False)
            assert "project" in result
    
    # Test 6: Password protected zip with no_input=True raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            zipf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 7: URL download with mocked requests
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch('requests.get') as mock_get:
            mock_response = mock.Mock()
            mock_response.iter_content.return_value = [b"zip content"]
            mock_get.return_value = mock_response
            
            # Mock prompt_and_delete to return True
            with mock.patch('cookiecutter.prompt.prompt_and_delete', return_value=True):
                # Create a mock zip file in memory
                import io
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                    zipf.writestr("project/", "")
                    zipf.writestr("project/file.txt", "content")
                zip_buffer.seek(0)
                
                # Mock the file writing
                with mock.patch('builtins.open', mock.mock_open()):
                    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir)
                    assert "project" in result
    
    # Test 8: Invalid password after 3 attempts raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            zipf.setpassword(b"correct")
        
        # Mock read_repo_password to return wrong passwords
        with mock.patch('cookiecutter.prompt.read_repo_password', side_effect=["wrong1", "wrong2", "wrong3"]):
            with pytest.raises(InvalidZipRepository):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=False)
    
    # Test 9: Password provided via parameter works
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            zipf.setpassword(b"mypassword")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="mypassword")
        assert "project" in result
    
    # Test 10: Invalid password via parameter raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
            zipf.setpassword(b"mypassword")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong")


# LLM-generated content at query #2
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import pytest
    
    # Test 1: Successful extraction from URL
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        
        result = unzip('http://example.com/repo.zip', is_url=True, clone_to_dir='.')
        
        assert result == '/tmp/tempdir/project'
        mock_make_path.assert_called_once()
        mock_get.assert_called_once_with('http://example.com/repo.zip', stream=True, timeout=100)
    
    # Test 2: Successful extraction from local file
    with patch('os.path.abspath', return_value='/local/repo.zip'), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists') as mock_make_path:
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        
        result = unzip('/local/repo.zip', is_url=False, clone_to_dir='.')
        
        assert result == '/tmp/tempdir/project'
        mock_make_path.assert_called_once()
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = []
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        assert 'Zip repository' in str(exc_info.value) and 'empty' in str(exc_info.value)
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['file.txt']
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        assert 'does not include a top-level directory' in str(exc_info.value)
    
    # Test 5: Password protected zip with correct password
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        
        # First extractall raises RuntimeError (password protected)
        # Second succeeds with password
        mock_zip_instance.extractall.side_effect = [
            RuntimeError('password required'),
            None
        ]
        
        result = unzip('http://example.com/repo.zip', is_url=True, password='secret')
        
        assert result == '/tmp/tempdir/project'
        mock_zip_instance.extractall.assert_called_with(path='/tmp/tempdir', pwd=b'secret')
    
    # Test 6: Invalid password raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/']
        
        mock_zip_instance.extractall.side_effect = RuntimeError('password required')
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True, password='wrong', no_input=True)
        assert 'Invalid password' in str(exc_info.value)
    
    # Test 7: Bad zip file raises InvalidZipRepository
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile', side_effect=zipfile.BadZipFile('bad zip')), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        with pytest.raises(Exception) as exc_info:
            unzip('http://example.com/repo.zip', is_url=True)
        assert 'not a valid zip archive' in str(exc_info.value)
    
    # Test 8: Prompt to delete existing file when no_input=False
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=True), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'chunk1']
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        
        result = unzip('http://example.com/repo.zip', is_url=True, no_input=False)
        
        assert result == '/tmp/tempdir/project'
    
    # Test 9: Skip download when prompt_and_delete returns False
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=True), \
         patch('cookiecutter.prompt.prompt_and_delete', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists'):
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/', 'project/file.txt']
        
        result = unzip('http://example.com/repo.zip', is_url=True, no_input=False)
        
        assert result == '/tmp/tempdir/project'
        mock_get.assert_not_called()
    
    # Test 10: Password prompt with retry logic
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('tempfile.mkdtemp', return_value='/tmp/tempdir'), \
         patch('cookiecutter.utils.make_sure_path_exists'), \
         patch('cookiecutter.prompt.read_repo_password') as mock_read_password:
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ['project/']
        
        # First extractall raises RuntimeError, then two failed attempts, then success
        mock_zip_instance.extractall.side_effect = [
            RuntimeError('password required'),
            RuntimeError('wrong password'),
            RuntimeError('wrong password'),
            None
        ]
        mock_read_password.side_effect = ['wrong1', 'wrong2', 'correct']
        
        result = unzip('http://example.com/repo.zip', is_url=True, no_input=False)
        
        assert result == '/tmp/tempdir/project'
        assert mock_read_password.call_count == 3
    
    # Test 11: Exceed password retry limit
    with patch('requests.get'), \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         patch('cookiecutter.utils.make_sure_path_exists'), \
        


# LLM-generated content at query #3
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import pytest
    from cookiecutter.exceptions import InvalidZipRepository

    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()

    # Test 2: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_zip = Path(tmpdir) / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(invalid_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_zip = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(empty_zip, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(empty_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_zip = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(bad_zip), is_url=False, clone_to_dir=tmpdir)

    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
        assert "project" in result

    # Test 6: Password protected zip with wrong password raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong")

    # Test 7: URL download with mock
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False), \
         patch('zipfile.ZipFile') as mock_zip, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_response = Mock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_get.return_value = mock_response
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
        
        with patch('builtins.open', mock_open()):
            result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
        
        mock_get.assert_called_once_with("http://example.com/repo.zip", stream=True, timeout=100)

    # Test 8: URL download with existing file and no_input=True
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=True), \
         patch('zipfile.ZipFile') as mock_zip, \
         tempfile.TemporaryDirectory() as tmpdir:
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
        
        result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
        
        # Should not download since file exists and no_input=True
        mock_get.assert_not_called()

    # Test 9: Test with custom clone_to_dir path
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom"
        zip_path = Path(tmpdir) / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=str(custom_dir))
        assert custom_dir.exists()

    # Test 10: Password protected with no_input=True and no password raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
            zf.setpassword(b"secret")
        
        with pytest.raises(InvalidZipRepository):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)


# LLM-generated content at query #4
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import pytest
    
    # Test 1: Local zip file extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "test_project" in result
        assert Path(result).exists()
    
    # Test 2: URL zip file download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                
                result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                assert mock_get.called
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository, match="is empty"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Bad zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "corrupt.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        # Create password protected zip (simulated)
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
            
            # First extract attempt fails, second succeeds with password
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None  # Success with password
            ]
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2
    
    # Test 7: Password protected zip with no_input=True raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
            mock_zip_instance.extractall.side_effect = RuntimeError("password required")
            
            with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 8: Existing cached file with no_input=False prompts for deletion
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                
                # Create existing file
                cached_file = Path(tmpdir) / "test.zip"
                cached_file.write_text("existing content")
                
                with patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt:
                    mock_prompt.return_value = True  # User chooses to download
                    
                    result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir, no_input=False)
                    assert mock_prompt.called
    
    # Test 9: Path expansion for clone_to_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        # Test with Path object
        result = unzip(str(zip_path), is_url=False, clone_to_dir=Path(tmpdir))
        assert "test_project" in result
        
        # Test with string containing ~
        with patch('pathlib.Path.expanduser') as mock_expand:
            mock_expand.return_value = Path(tmpdir)
            result = unzip(str(zip_path), is_url=False, clone_to_dir="~/test")
            assert mock_expand.called


# LLM-generated content at query #5
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open, call
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_project/", "")
            zf.writestr("test_project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "test_project" in result
        assert Path(result).exists()
    
    # Test 2: URL download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get, \
             patch('os.path.exists', return_value=False):
            
            # Mock response with zip content
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"fake_zip_content"]
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            mock_get.return_value = mock_response
            
            # Mock zip file extraction
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
                mock_zip_instance.__exit__ = Mock(return_value=None)
                mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir)
                assert mock_get.called
                assert "test_project" in result
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository, match="is empty"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        # Create password protected zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("test_project/", "")
            zf.setpassword(b"secret")
            zf.writestr("test_project/file.txt", "content")
        
        # Reopen with password for testing
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.setpassword(b"secret")
        
        # Mock the zip extraction to simulate password requirement
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
            mock_zip_instance.__exit__ = Mock(return_value=None)
            mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
            
            # First extractall raises RuntimeError (password required)
            # Second succeeds with password
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None
            ]
            mock_zip.return_value = mock_zip_instance
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert "test_project" in result
    
    # Test 7: Password protected zip with wrong password raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
            mock_zip_instance.__exit__ = Mock(return_value=None)
            mock_zip_instance.namelist.return_value = ["test_project/", "test_project/file.txt"]
            
            # Always raise RuntimeError for wrong password
            mock_zip_instance.extractall.side_effect = RuntimeError("password required")
            mock_zip.return_value = mock_zip_instance
            
            with pytest.raises(InvalidZipRepository, match="Invalid password"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="wrong", no_input=True)
    
    # Test 8: Existing cached file with no_input=True forces refresh
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get, \
             patch('os.path.exists', return_value=True), \
             patch('cookiecutter.prompt.prompt_and_delete', return_value=True):
            
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"new_content"]
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
                mock_zip_instance.__exit__ = Mock(return_value=None)
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                assert mock_get.called
    
    # Test 9: Ensure clone_to_dir is created if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_directory"
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__ = Mock(return_value=mock_zip_instance)
            mock_zip_instance.__exit__ = Mock(return_value=None)
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip.return_value = mock_zip_instance
            
            # Create a local zip file
            zip_path = Path(tmpdir) / "test.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("project/", "")
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=str(new_dir))
            assert new_dir.exists()


# LLM-generated content at query #6
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import zipfile
    import pytest
    
    # Test 1: Local zip file extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "test content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()
    
    # Test 2: URL zip file download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                
                # Mock extractall
                def mock_extractall(path=None, pwd=None):
                    target_dir = Path(path) / "project"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    (target_dir / "file.txt").write_text("test")
                
                mock_zip_instance.extractall.side_effect = mock_extractall
                
                result = unzip("http://example.com/test.zip", is_url=True, clone_to_dir=tmpdir)
                assert "project" in result
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "empty" in str(exc_info.value).lower()
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "bad.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "top-level directory" in str(exc_info.value)
    
    # Test 5: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        # Create password protected zip
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.setpassword(b"secret")
        
        # Mock the zip file to simulate password protection
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            
            # First extractall raises RuntimeError (password needed)
            # Second succeeds with password
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None  # Success with password
            ]
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2
    
    # Test 6: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(Exception) as exc_info:
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "not a valid zip archive" in str(exc_info.value)
    
    # Test 7: Existing cached file with no_input=True forces refresh
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "cached.zip"
        cache_file.write_text("cached content")
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"new content"]
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip.return_value.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip_instance.extractall.return_value = None
                
                unzip("http://example.com/cached.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                # Should download fresh copy
                assert mock_get.called
    
    # Test 8: Test with custom clone_to_dir path
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom"
        zip_path = Path(tmpdir) / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=custom_dir)
        assert custom_dir.exists()
        assert "project" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_unzip():
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open
    import pytest
    from zipfile import ZipFile, BadZipFile
    import requests
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock zip file structure
        zip_path = Path(tmpdir) / "test.zip"
        with ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()
    
    # Test 2: URL download and extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=True)
                assert mock_get.called
    
    # Test 3: Empty zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with ZipFile(zip_path, 'w') as zipf:
            pass
        
        with pytest.raises(InvalidZipRepository, match="is empty"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "flat.zip"
        with ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Invalid zip file raises exception
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        # Create a password protected zip (simulated with mock)
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            
            # First extractall raises RuntimeError (password required)
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None  # Success with password
            ]
            mock_zip.return_value = mock_zip_instance
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2
    
    # Test 7: Password protected zip with wrong password in no_input mode
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.extractall.side_effect = RuntimeError("password required")
            mock_zip.return_value = mock_zip_instance
            
            with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 8: Existing cached file with no_input=True forces re-download
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "clone"
        clone_dir.mkdir()
        cached_file = clone_dir / "repo.zip"
        cached_file.write_bytes(b"old content")
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"new content"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip.return_value = mock_zip_instance
                
                unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=str(clone_dir), no_input=True)
                assert mock_get.called
    
    # Test 9: Path expansion for clone_to_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        home_dir = Path.home()
        test_file = home_dir / "test.zip"
        
        # Create a simple zip file in home directory
        with ZipFile(test_file, 'w') as zipf:
            zipf.writestr("project/", "")
            zipf.writestr("project/file.txt", "content")
        
        try:
            result = unzip(str(test_file), is_url=False, clone_to_dir="~")
            assert "project" in result
        finally:
            test_file.unlink(missing_ok=True)


# LLM-generated content at query #8
#--------------------------

```python
def test_unzip():
    import tempfile
    import zipfile
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, mock_open, call
    from cookiecutter.exceptions import InvalidZipRepository
    
    # Test 1: Successful extraction from local zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test zip file
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
        assert "project" in result
        assert Path(result).exists()
    
    # Test 2: Successful download and extraction from URL
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip.return_value = mock_zip_instance
                
                result = unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir)
                mock_get.assert_called_once_with("http://example.com/repo.zip", stream=True, timeout=100)
    
    # Test 3: Empty zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "empty.zip"
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        with pytest.raises(InvalidZipRepository, match="is empty"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 4: Zip without top-level directory raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "flat.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file.txt", "content")
        
        with pytest.raises(InvalidZipRepository, match="does not include a top-level directory"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 5: Invalid zip file raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "invalid.zip"
        zip_path.write_text("not a zip file")
        
        with pytest.raises(InvalidZipRepository, match="is not a valid zip archive"):
            unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir)
    
    # Test 6: Password protected zip with correct password
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        # Create password protected zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project/", "")
            zf.setpassword(b"secret")
        
        # Mock zipfile to simulate password protection
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            
            # First extractall raises RuntimeError (password needed)
            # Second succeeds with password
            mock_zip_instance.extractall.side_effect = [
                RuntimeError("password required"),
                None
            ]
            mock_zip.return_value = mock_zip_instance
            
            result = unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, password="secret")
            assert mock_zip_instance.extractall.call_count == 2
    
    # Test 7: Password protected zip with no_input=True raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        with patch('zipfile.ZipFile') as mock_zip:
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.extractall.side_effect = RuntimeError("password required")
            mock_zip.return_value = mock_zip_instance
            
            with pytest.raises(InvalidZipRepository, match="Unable to unlock password protected repository"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=True)
    
    # Test 8: Existing cached file with no_input=False prompts for deletion
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('requests.get') as mock_get, \
             patch('cookiecutter.prompt.prompt_and_delete') as mock_prompt:
            
            mock_prompt.return_value = True  # User chooses to download
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"mock zip content"]
            mock_get.return_value = mock_response
            
            # Create existing cached file
            cached_file = Path(tmpdir) / "repo.zip"
            cached_file.write_text("old content")
            
            with patch('zipfile.ZipFile') as mock_zip:
                mock_zip_instance = Mock()
                mock_zip_instance.__enter__.return_value = mock_zip_instance
                mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
                mock_zip.return_value = mock_zip_instance
                
                unzip("http://example.com/repo.zip", is_url=True, clone_to_dir=tmpdir, no_input=False)
                mock_prompt.assert_called_once_with(str(cached_file), no_input=False)
    
    # Test 9: Invalid password after 3 attempts raises InvalidZipRepository
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "protected.zip"
        
        with patch('zipfile.ZipFile') as mock_zip, \
             patch('cookiecutter.prompt.read_repo_password') as mock_read_password:
            
            mock_zip_instance = Mock()
            mock_zip_instance.__enter__.return_value = mock_zip_instance
            mock_zip_instance.namelist.return_value = ["project/", "project/file.txt"]
            mock_zip_instance.extractall.side_effect = RuntimeError("password required")
            mock_zip.return_value = mock_zip_instance
            
            mock_read_password.return_value = "wrong_password"
            
            with pytest.raises(InvalidZipRepository, match="Invalid password provided"):
                unzip(str(zip_path), is_url=False, clone_to_dir=tmpdir, no_input=False)
            
            assert mock_read_password.call_count == 3
    
    # Test 10: Ensure clone_to_dir is created if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_dir = Path(tmpdir) / "new_dir"
        zip_path = Path(tmpdir) / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("project/", "")
            zf.writestr("project/file.txt", "content")
        
        result = unzip(str(zip_path), is_url=False, clone_to_dir=non_existent_dir)
        assert non_existent_dir.exists()


