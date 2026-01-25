####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function with various inputs."""
    # Test with template name without .json extension
    result = get_file_name('/replay/dir', 'my_template')
    assert result == os.path.join('/replay/dir', 'my_template.json')
    
    # Test with template name that already has .json extension
    result = get_file_name('/replay/dir', 'my_template.json')
    assert result == os.path.join('/replay/dir', 'my_template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    replay_path = Path('/replay/dir')
    result = get_file_name(replay_path, 'template_name')
    assert result == os.path.join(replay_path, 'template_name.json')
    
    # Test with empty template name
    result = get_file_name('/replay/dir', '')
    assert result == os.path.join('/replay/dir', '.json')
    
    # Test with template name containing dots but not ending with .json
    result = get_file_name('/replay/dir', 'my.template.name')
    assert result == os.path.join('/replay/dir', 'my.template.name.json')
    
    # Test with relative path for replay_dir
    result = get_file_name('./replay', 'template')
    assert result == os.path.join('./replay', 'template.json')


# LLM-generated content at query #2
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with string path and template name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with string path and template name with .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object and template name without .json extension
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    result = get_file_name(replay_dir, 'another_template')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with Path object and template name with .json extension
    result = get_file_name(replay_dir, 'another_template.json')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with empty string replay_dir
    result = get_file_name('', 'template')
    assert result == os.path.join('', 'template.json')
    
    # Test with template name containing multiple dots
    result = get_file_name('/tmp/replay', 'my.template.name')
    assert result == os.path.join('/tmp/replay', 'my.template.name.json')


# LLM-generated content at query #3
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with string replay_dir and template_name without .json suffix
    result = get_file_name('/tmp/replay', 'my-template')
    assert result == os.path.join('/tmp/replay', 'my-template.json')
    
    # Test with template_name that already ends with .json
    result = get_file_name('/tmp/replay', 'my-template.json')
    assert result == os.path.join('/tmp/replay', 'my-template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    replay_path = Path('/tmp/replay')
    result = get_file_name(replay_path, 'my-template')
    assert result == os.path.join(replay_path, 'my-template.json')
    
    # Test with Path object and template_name with .json
    result = get_file_name(replay_path, 'my-template.json')
    assert result == os.path.join(replay_path, 'my-template.json')
    
    # Test with empty replay_dir
    result = get_file_name('', 'my-template')
    assert result == os.path.join('', 'my-template.json')
    
    # Test with special characters in template_name
    result = get_file_name('/tmp/replay', 'my-special-template')
    assert result == os.path.join('/tmp/replay', 'my-special-template.json')


# LLM-generated content at query #4
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with string replay_dir and template_name without .json suffix
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with template_name that already has .json suffix
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    replay_path = Path('/tmp/replay')
    result = get_file_name(replay_path, 'another_template')
    assert result == os.path.join(replay_path, 'another_template.json')
    
    # Test with Path object and .json suffix already present
    result = get_file_name(replay_path, 'another_template.json')
    assert result == os.path.join(replay_path, 'another_template.json')
    
    # Test with empty template name
    result = get_file_name('/tmp/replay', '')
    assert result == os.path.join('/tmp/replay', '.json')
    
    # Test with complex template names
    result = get_file_name('/home/user/replay', 'my-complex-template-name')
    assert result == os.path.join('/home/user/replay', 'my-complex-template-name.json')


# LLM-generated content at query #5
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with template name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with template name that already has .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'another_template')
    assert result == os.path.join(Path('/tmp/replay'), 'another_template.json')
    
    # Test with Path object and .json extension
    result = get_file_name(Path('/tmp/replay'), 'another_template.json')
    assert result == os.path.join(Path('/tmp/replay'), 'another_template.json')
    
    # Test with empty template name
    result = get_file_name('/tmp/replay', '')
    assert result == os.path.join('/tmp/replay', '.json')
    
    # Test with complex template name
    result = get_file_name('/home/user/replay', 'my-complex_template_name')
    assert result == os.path.join('/home/user/replay', 'my-complex_template_name.json')


# LLM-generated content at query #6
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test the dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with correct directory
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify the file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function adds .json extension when template_name doesn't have it."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump function doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_valid_json(tmp_path, mocker):
    """Test dump creates valid JSON file with proper formatting."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "version": "1.0.0"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    
    # Verify file is valid JSON and properly indented
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
        assert "  " in content  # Check for indentation
        loaded = json.loads(content)
        assert loaded == context


# LLM-generated content at query #7
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Write test file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "test_project"}
    
    # Write test file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works when template_name already ends with .json."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, template_name)
    assert result == valid_context


def test_load_invalid_json(tmp_path):
    """Test load raises JSONDecodeError for invalid json content."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write("invalid json content {]")
    
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #8
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json file and returns context."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create test json file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'other_key': 'value'}
    
    # Create test json file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works with template_name that already has .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create test json file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #9
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Call dump function
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct content
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context['cookiecutter']['project_name'] == "test_project"


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function with template name already containing .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "project_name": "test_project"
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump function calls make_sure_path_exists."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path, mocker):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "nested": {
                "key": "value"
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    content = replay_file.read_text(encoding='utf-8')
    
    # Verify indentation (indent=2 was used)
    assert '  ' in content


# LLM-generated content at query #10
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function when template_name already has .json extension."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_dir(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay" / "nested" / "dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_json_formatting(tmp_path):
    """Test dump function writes json with proper indentation."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content


# LLM-generated content at query #11
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    test_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Create test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"some_key": "some_value"}
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template_name that already has .json suffix."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    test_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


# LLM-generated content at query #12
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and returns json context from file."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_with_json_suffix(tmp_path):
    """Test load function handles template names with .json suffix."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    result = load(replay_dir, template_name)
    
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "other_key": "value"
    }
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #13
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump function with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author",
            "options": {
                "use_pytest": True,
                "python_version": "3.9"
            },
            "tags": ["web", "api"]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #14
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and returns valid context from json file."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Act
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {
        'project_name': 'my_project'
    }
    
    # Create the replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Act & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises json.JSONDecodeError for invalid json."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template'
    
    # Create an invalid json file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write('{ invalid json }')
    
    # Act & Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template_name already has .json suffix."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Act
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #15
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates JSON context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Write test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when context lacks cookiecutter key."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create context without cookiecutter key
    invalid_context = {"project_name": "my_project"}
    
    # Write test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup with non-existent file
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works correctly when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    # Create valid context
    valid_context = {"cookiecutter": {"key": "value"}}
    
    # Write test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test load
    result = load(replay_dir, template_name)
    assert result == valid_context


# LLM-generated content at query #16
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #17
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author_name': 'Test Author'
        }
    }
    
    # Create test json file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template_name already has .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create test json file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'other_key': 'some_value'
    }
    
    # Create test json file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises JSONDecodeError for invalid json content."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    
    # Create test file with invalid json
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        f.write('invalid json content {')
    
    # Execute and Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #18
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author_name": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_with_json_suffix(tmp_path):
    """Test load function with .json suffix in template name."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises json.JSONDecodeError for invalid json."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create invalid json file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write("invalid json {")
    
    # Test and Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #19
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_without_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "project_name": "my_project"
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load correctly handles template names with .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #20
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates context from JSON file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"invalid_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test load
    result = load(replay_dir, template_name)
    assert result == context


# LLM-generated content at query #21
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "My Project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path):
    """Test dump with template name that already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "My Project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_writes_formatted_json(tmp_path):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation


# LLM-generated content at query #22
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"
    assert loaded_context["cookiecutter"]["author"] == "test_author"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template names that already have .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #23
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, "nonexistent", "replay")
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)


def test_dump_with_complex_context(tmp_path):
    """Test dump function with complex nested context."""
    replay_dir = tmp_path
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "options": ["opt1", "opt2", "opt3"]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #24
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test load function
    loaded_context = load(replay_dir, template_name)
    
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template_name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context = load(replay_dir, template_name)
    
    assert loaded_context == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #25
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context missing cookiecutter key."""
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called
    from cookiecutter.replay import make_sure_path_exists as mocked
    # The function should have been called with replay_dir


# LLM-generated content at query #26
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author_name": "Test Author"
        }
    }
    
    # Create a replay file with test data
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute and Assert
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"some_key": "some_value"}
    
    # Create a replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #27
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test the load function."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test load with valid context
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"
    
    # Test load with .json suffix in template_name
    loaded_context = load(replay_dir, "test_template.json")
    assert loaded_context == context
    
    # Test load with missing cookiecutter key
    invalid_context = {"project": "test"}
    invalid_replay_file = replay_dir / "invalid_template.json"
    with open(invalid_replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f, indent=2)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")
    
    # Test load with non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "nonexistent_template")


# LLM-generated content at query #28
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from replay file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test: load returns correct context
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when context missing cookiecutter key."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create context without cookiecutter key
    invalid_context = {
        "other_key": "value"
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test: load raises ValueError
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test: load raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load handles template names with .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test: load works with .json suffix
    result = load(replay_dir, template_name)
    assert result == valid_context


# LLM-generated content at query #29
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates JSON context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project"
        }
    }
    
    # Write test data to file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create invalid context without cookiecutter key
    invalid_context = {"project_name": "test_project"}
    
    # Write test data to file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "nonexistent_template")


def test_load_with_json_suffix(tmp_path):
    """Test load handles template names with .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    valid_context = {"cookiecutter": {"key": "value"}}
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, template_name)
    assert result == valid_context


# LLM-generated content at query #30
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context1 = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    context2 = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, context1)
    dump(replay_dir, template_name, context2)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context2
    assert loaded_context["cookiecutter"]["project_name"] == "new_project"


def test_dump_with_complex_context(tmp_path):
    """Test dump with nested and complex context data."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "list": [1, 2, 3]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #31
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create the replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create the replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context


# LLM-generated content at query #32
#--------------------------

```python
def test_dump(tmp_path, monkeypatch):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump handles template names with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #33
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_without_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "test_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works correctly when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    test_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


# LLM-generated content at query #34
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        },
        "extra_key": "extra_value"
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "other_key": "value"
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name ending in .json."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #36
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json replay file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'author': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #37
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already containing .json suffix."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    # Create the replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #38
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'some_key': 'some_value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works correctly when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #39
#--------------------------

```python
def test_dump(tmp_path, monkeypatch):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_dir(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    assert not replay_dir.exists()
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_formatting(tmp_path):
    """Test dump function writes json with proper formatting (indent=2)."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #40
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_creates_replay_dir(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / "my-template.json").exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_writes_valid_json(tmp_path):
    """Test dump writes valid JSON with proper formatting."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    content = replay_file.read_text(encoding="utf-8")
    
    # Verify it's valid JSON and has indentation
    loaded = json.loads(content)
    assert loaded == context
    assert "  " in content  # Check for indentation


# LLM-generated content at query #41
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from pathlib import Path
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'author_name': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'some_key': 'some_value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works correctly when template_name already has .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_slug': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #42
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'
    assert result['cookiecutter']['author'] == 'test_author'


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #43
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "invalid_key": {
            "project_name": "my_project"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #44
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context
    assert 'cookiecutter' in loaded_context
    assert loaded_context['cookiecutter']['project_name'] == 'my_project'
    assert loaded_context['cookiecutter']['author'] == 'John Doe'


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'project_name': 'my_project'
    }
    
    # Create the replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with string replay_dir and template_name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with template_name that already has .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    result = get_file_name(replay_dir, 'another_template')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with Path object and .json extension
    result = get_file_name(replay_dir, 'another_template.json')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with empty template name
    result = get_file_name('/tmp/replay', '')
    assert result == os.path.join('/tmp/replay', '.json')
    
    # Test with template name containing multiple dots
    result = get_file_name('/tmp/replay', 'my.template.name')
    assert result == os.path.join('/tmp/replay', 'my.template.name.json')
    
    # Test with template name ending in .json but with more content after
    result = get_file_name('/tmp/replay', 'template.json.backup')
    assert result == os.path.join('/tmp/replay', 'template.json.backup.json')


# LLM-generated content at query #2
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name returns correct file path."""
    # Test with template name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with template name that already has .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template_name')
    assert result == os.path.join(Path('/tmp/replay'), 'template_name.json')
    
    # Test with different replay directories
    result = get_file_name('/home/user/.cookiecutters', 'my_project')
    assert result == os.path.join('/home/user/.cookiecutters', 'my_project.json')
    
    # Test with template name containing dots but not ending with .json
    result = get_file_name('/replay', 'my.template.name')
    assert result == os.path.join('/replay', 'my.template.name.json')


# LLM-generated content at query #3
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function."""
    # Test with template name without .json extension
    result = get_file_name('/replay', 'my-template')
    assert result == os.path.join('/replay', 'my-template.json')
    
    # Test with template name that already has .json extension
    result = get_file_name('/replay', 'my-template.json')
    assert result == os.path.join('/replay', 'my-template.json')
    
    # Test with Path object as replay_dir
    from pathlib import Path
    result = get_file_name(Path('/replay'), 'my-template')
    assert result == os.path.join(Path('/replay'), 'my-template.json')
    
    # Test with different replay directories
    result = get_file_name('/home/user/.cookiecutters', 'template-name')
    assert result == os.path.join('/home/user/.cookiecutters', 'template-name.json')
    
    # Test with template name already ending in .json
    result = get_file_name('/tmp', 'test.json')
    assert result == os.path.join('/tmp', 'test.json')
    assert not result.endswith('.json.json')


# LLM-generated content at query #4
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name function with various inputs."""
    # Test with string replay_dir and template_name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')

    # Test with template_name that already has .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')

    # Test with Path object as replay_dir
    from pathlib import Path
    replay_path = Path('/tmp/replay')
    result = get_file_name(replay_path, 'my_template')
    assert result == os.path.join(replay_path, 'my_template.json')

    # Test with Path object and .json extension
    result = get_file_name(replay_path, 'my_template.json')
    assert result == os.path.join(replay_path, 'my_template.json')

    # Test with empty template_name
    result = get_file_name('/tmp/replay', '')
    assert result == os.path.join('/tmp/replay', '.json')

    # Test with template_name containing multiple dots
    result = get_file_name('/tmp/replay', 'my.template.name')
    assert result == os.path.join('/tmp/replay', 'my.template.name.json')

    # Test with template_name ending in .json but with multiple dots
    result = get_file_name('/tmp/replay', 'my.template.json')
    assert result == os.path.join('/tmp/replay', 'my.template.json')


# LLM-generated content at query #5
#--------------------------

```python
def test_get_file_name():
    """Test get_file_name returns correct file path."""
    # Test with template name without .json extension
    result = get_file_name('/tmp/replay', 'my_template')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with template name with .json extension
    result = get_file_name('/tmp/replay', 'my_template.json')
    assert result == os.path.join('/tmp/replay', 'my_template.json')
    
    # Test with Path object instead of string
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    result = get_file_name(replay_dir, 'another_template')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with Path object and .json extension
    result = get_file_name(replay_dir, 'another_template.json')
    assert result == os.path.join(replay_dir, 'another_template.json')
    
    # Test with empty template name
    result = get_file_name('/tmp/replay', '')
    assert result == os.path.join('/tmp/replay', '.json')
    
    # Test with template name containing multiple dots
    result = get_file_name('/tmp/replay', 'my.template.name')
    assert result == os.path.join('/tmp/replay', 'my.template.name.json')


# LLM-generated content at query #6
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises json.JSONDecodeError for invalid json."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    
    # Create invalid json file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write('{ invalid json }')
    
    # Test and Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #7
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create test file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'invalid_key': 'value'}
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_nonexistent_file(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #8
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"
    assert loaded_context["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #9
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_indentation(tmp_path):
    """Test dump function writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content


# LLM-generated content at query #10
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name already containing .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"invalid_key": "value"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    from cookiecutter.replay import load
    
    replay_dir = Path(tmp_path)
    template_name = "test_template"
    context = {"cookiecutter": {"data": "test"}}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


# LLM-generated content at query #11
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my-template.json"
    assert expected_file.exists()
    
    with open(expected_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my-template.json"
    assert expected_file.exists()


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_indentation(tmp_path):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation


# LLM-generated content at query #12
#--------------------------

```python
def test_dump(tmp_path, monkeypatch):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    # Verify content is correct
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "test_project"


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "test_template.json").exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Should not add another .json suffix
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "other_key": "value"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "key": "old_value"
        }
    }
    new_context = {
        "cookiecutter": {
            "key": "new_value"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context
    assert saved_context["cookiecutter"]["key"] == "new_value"


def test_dump_with_complex_context(tmp_path):
    """Test dump handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {
                "level1": {
                    "level2": ["item1", "item2"]
                }
            },
            "list": [1, 2, 3],
            "bool": True,
            "null": None
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["nested"]["level1"]["level2"] == ["item1", "item2"]


# LLM-generated content at query #13
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_template_name_with_json_extension(tmp_path, mocker):
    """Test dump handles template names that already have .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file doesn't have double .json extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists to create replay directory."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('json.dump')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #15
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    assert replay_file.endswith('test_template.json')


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'project_name': 'test_project'
    }
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / 'new_dir'
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)


def test_dump_writes_valid_json(tmp_path):
    """Test dump function writes properly formatted json."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe',
            'version': '1.0.0'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '"cookiecutter"' in content
    assert '"project_name"' in content
    assert 'my_project' in content


# LLM-generated content at query #16
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    test_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "Test Author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    test_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #17
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    with open(expected_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()
    assert not (replay_dir / "test_template.json.json").exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context
    assert saved_context["cookiecutter"]["project_name"] == "new_project"


# LLM-generated content at query #18
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_path_object(tmp_path):
    """Test dump function works with Path object."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context_old = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    context_new = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, context_old)
    dump(replay_dir, template_name, context_new)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context_new


# LLM-generated content at query #19
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay" / "nested" / "dir"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #20
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_without_json_extension(tmp_path):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / "test-template.json").exists()
    assert not (replay_dir / "test-template.json.json").exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_complex_context(tmp_path):
    """Test dump function handles complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "complex-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #21
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test: load successfully reads the file
    result = load(replay_dir, template_name)
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template_name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    test_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(replay_dir, template_name)
    assert result == test_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when context lacks cookiecutter key."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "test_project"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #22
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'
    assert result['cookiecutter']['author'] == 'John Doe'


def test_load_with_json_extension(tmp_path):
    """Test load function with .json extension in template name."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {
        'other_key': 'some_value'
    }
    
    # Create a replay file without cookiecutter key
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent-template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #23
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        },
        "extra_field": "extra_value"
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["extra_field"] == "extra_value"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    context = {"other_key": "value"}
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)
    
    with open(expected_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, "new_dir")
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, "test_template.json"))


def test_dump_json_formatting(tmp_path):
    """Test dump function writes json with proper indentation."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    with open(expected_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #25
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'my_project'}
    
    # Create replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works when template_name already has .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    test_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


# LLM-generated content at query #26
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #27
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #28
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {"cookiecutter": {"old": "data"}}
    new_context = {"cookiecutter": {"new": "data"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump function with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #29
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and returns context from json file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'my_project'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #30
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create a replay file with valid context
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test: load should successfully read and return context
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_without_json_extension(tmp_path):
    """Test load function works with template names that don't have .json extension."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when context lacks cookiecutter key."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "invalid_template"
    context = {"project": "test"}
    
    replay_file = replay_dir / "invalid_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises JSONDecodeError for invalid json content."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = "bad_json"
    
    replay_file = replay_dir / "bad_json.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write("invalid json content {")
    
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #31
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = tmp_path / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(tmp_path, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template'
    invalid_context = {
        'project_name': 'test_project'
    }
    
    # Create replay file with invalid context
    replay_file = tmp_path / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, 'nonexistent_template')


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = tmp_path / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(tmp_path, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #32
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump function with nested and complex context data."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "number": 42,
            "boolean": True
        }
    }
    
    dump(replay_dir, template_name, context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #33
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "my-template.json").exists()


def test_dump_string_path(tmp_path):
    """Test dump works with string path."""
    replay_dir = str(tmp_path / "replay")
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = os.path.join(replay_dir, "my-template.json")
    assert os.path.exists(replay_file)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context1 = {"cookiecutter": {"key": "value1"}}
    context2 = {"cookiecutter": {"key": "value2"}}
    
    dump(replay_dir, template_name, context1)
    dump(replay_dir, template_name, context2)
    
    replay_file = replay_dir / "my-template.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context2


# LLM-generated content at query #34
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads and returns context from json file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'some_key': 'some_value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works with template_name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create test file without cookiecutter key
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name that already has .json extension."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #36
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Test: Valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    
    # Test: Missing cookiecutter key raises ValueError
    invalid_context = {
        "other_key": "value"
    }
    replay_file2 = replay_dir / "invalid_template.json"
    with open(replay_file2, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")
    
    # Test: File with .json extension
    context_with_json = {
        "cookiecutter": {"key": "value"}
    }
    replay_file3 = replay_dir / "template.json.json"
    with open(replay_file3, 'w', encoding="utf-8") as f:
        json.dump(context_with_json, f)
    
    result = load(replay_dir, "template.json")
    assert result == context_with_json
    
    # Test: Non-existent file raises FileNotFoundError
    with raises(FileNotFoundError):
        load(replay_dir, "nonexistent_template")


# LLM-generated content at query #37
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        },
        "other_key": "value"
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test: Successfully load valid context
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    
    # Test: Load with template name already ending in .json
    replay_file_json = replay_dir / f"{template_name}.json.json"
    with open(replay_file_json, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, f"{template_name}.json")
    assert result == valid_context
    
    # Test: Raise ValueError when cookiecutter key is missing
    invalid_context = {
        "other_key": "value"
    }
    
    invalid_replay_file = replay_dir / "invalid_template.json"
    with open(invalid_replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")
    
    # Test: Raise error when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "nonexistent_template")


# LLM-generated content at query #38
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with replay_dir
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct name
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    # Verify file contents
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump handles template name with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file doesn't have double .json extension
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()


def test_dump_creates_valid_json(tmp_path, mocker):
    """Test dump creates properly formatted JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "version": "1.0.0"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    
    # Verify JSON is valid and has proper indentation
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    loaded = json.loads(content)
    assert loaded == context
    assert "  " in content  # Check for indentation


# LLM-generated content at query #39
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author_name': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template_name that already has .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {
        'project_name': 'test_project'
    }
    
    # Create replay file without cookiecutter key
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #40
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    from cookiecutter.replay import load
    
    # Setup test data
    replay_dir = tmp_path
    template_name = 'test_template'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'test_author'
        }
    }
    
    # Create test file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test load function
    result = load(replay_dir, template_name)
    
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'my_project'}
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name already having .json extension."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = 'test_template.json'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create test file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test load function
    result = load(replay_dir, template_name)
    
    assert result == test_context


# LLM-generated content at query #41
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_dump():
    """Test dump function writes context to json file correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "My Project",
                "author": "Test Author"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created with correct name
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
        assert loaded_context["cookiecutter"]["project_name"] == "My Project"


def test_dump_with_json_suffix():
    """Test dump function when template_name already has .json suffix."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = "test_template.json"
        context = {
            "cookiecutter": {
                "project_name": "My Project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created without double .json suffix
        expected_file = os.path.join(replay_dir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


def test_dump_missing_cookiecutter_key():
    """Test dump raises ValueError when context missing cookiecutter key."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = "test_template"
        context = {"project_name": "My Project"}
        
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, context)


def test_dump_creates_directory():
    """Test dump creates replay directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = os.path.join(temp_dir, "nonexistent", "replay")
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "My Project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        assert os.path.exists(replay_dir)
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)


def test_dump_with_path_object():
    """Test dump works with Path object for replay_dir."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir)
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "My Project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


def test_dump_overwrites_existing_file():
    """Test dump overwrites existing replay file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = "test_template"
        old_context = {
            "cookiecutter": {
                "project_name": "Old Project"
            }
        }
        new_context = {
            "cookiecutter": {
                "project_name": "New Project"
            }
        }
        
        dump(replay_dir, template_name, old_context)
        dump(replay_dir, template_name, new_context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == new_context
        assert loaded_context["cookiecutter"]["project_name"] == "New Project"


# LLM-generated content at query #42
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already has .json suffix."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    # Act & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    # Arrange
    replay_dir = tmp_path / "non_existent_replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_nested_context(tmp_path):
    """Test dump function with complex nested context."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == context
    assert loaded_data["cookiecutter"]["nested"]["key2"] == ["item1", "item2"]


# LLM-generated content at query #43
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert - file was created with correct content
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "test_project"


def test_dump_adds_json_suffix(tmp_path, mocker):
    """Test dump adds .json suffix when template_name doesn't have it."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_no_suffix_when_json_extension_exists(tmp_path, mocker):
    """Test dump doesn't add .json suffix when template_name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context missing cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_calls_make_sure_path_exists(tmp_path, mocker):
    """Test dump calls make_sure_path_exists with replay_dir."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path, mocker):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {"key": "value"}
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    # Verify it's indented (contains newlines and spaces)
    assert '\n' in content
    assert '  ' in content


# LLM-generated content at query #44
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create the replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create the replay file with invalid context
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f, indent=2)
    
    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function handles template names with .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create the replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #45
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump handles template names with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump handles complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {
                "key1": "value1",
                "key2": [1, 2, 3]
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #46
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "project_name": "my_project"
    }
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template_name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #47
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_dump():
    """Test dump function writes context to json file correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'my_project',
                'author': 'test_author'
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created with correct name
        expected_file = os.path.join(replay_dir, 'test_template.json')
        assert os.path.exists(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
        assert loaded_context['cookiecutter']['project_name'] == 'my_project'


def test_dump_with_json_suffix():
    """Test dump function when template name already has .json suffix."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template.json'
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file doesn't have double .json extension
        expected_file = os.path.join(replay_dir, 'test_template.json')
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


def test_dump_creates_directory():
    """Test dump function creates replay directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = os.path.join(temp_dir, 'nested', 'replay', 'dir')
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        assert not os.path.exists(replay_dir)
        
        dump(replay_dir, template_name, context)
        
        assert os.path.exists(replay_dir)
        assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))


def test_dump_missing_cookiecutter_key():
    """Test dump function raises ValueError when cookiecutter key is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        context = {
            'project_name': 'my_project'
        }
        
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, context)


def test_dump_with_path_object():
    """Test dump function works with Path object as replay_dir."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir)
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'my_project'
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / 'test_template.json'
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


def test_dump_overwrites_existing_file():
    """Test dump function overwrites existing replay file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        
        # Write initial context
        context1 = {
            'cookiecutter': {
                'project_name': 'project1'
            }
        }
        dump(replay_dir, template_name, context1)
        
        # Overwrite with new context
        context2 = {
            'cookiecutter': {
                'project_name': 'project2'
            }
        }
        dump(replay_dir, template_name, context2)
        
        # Verify file contains new context
        replay_file = os.path.join(replay_dir, 'test_template.json')
        with open(replay_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context2
        assert loaded_context['cookiecutter']['project_name'] == 'project2'


# LLM-generated content at query #48
#--------------------------

```python
def test_dump(tmp_path, monkeypatch):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct name
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify content was written correctly
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "my_project"


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created without double extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "my_project"}}
    
    assert not replay_dir.exists()
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_complex_context(tmp_path):
    """Test dump handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "options": {
                "nested": {
                    "deep_value": 42
                }
            },
            "items": [1, 2, 3]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["options"]["nested"]["deep_value"] == 42


# LLM-generated content at query #49
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and returns json context correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #50
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template names that already have .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #51
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create test file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test: Load valid context
    result = load(replay_dir, template_name)
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'other_key': 'value'}
    
    # Create test file with invalid context
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test: Should raise ValueError
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    
    # Create test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test: Load should work correctly
    result = load(replay_dir, template_name)
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test: Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load raises JSONDecodeError for invalid json content."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    
    # Create test file with invalid JSON
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write('invalid json content {')
    
    # Test: Should raise json.JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #52
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "project_name": "my_project"
    }
    
    # Create replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works correctly when template_name already has .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_invalid_json(tmp_path):
    """Test load raises JSONDecodeError when file contains invalid json."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create replay file with invalid json
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        f.write("{ invalid json }")
    
    # Execute & Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #53
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already having .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == new_context


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #54
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create context without cookiecutter key
    invalid_context = {"project_name": "my_project"}
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    valid_context = {
        "cookiecutter": {"key": "value"}
    }
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, template_name)
    assert result == valid_context


# LLM-generated content at query #55
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_dump():
    """Test dump function writes context to json file correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'My Project',
                'author': 'John Doe'
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created
        expected_file = os.path.join(replay_dir, f'{template_name}.json')
        assert os.path.exists(expected_file)
        
        # Verify content is correct
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        assert loaded_data == context


def test_dump_with_json_extension():
    """Test dump function when template_name already has .json extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template.json'
        context = {
            'cookiecutter': {
                'project_name': 'My Project'
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created without double extension
        expected_file = os.path.join(replay_dir, template_name)
        assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key():
    """Test dump raises ValueError when context lacks cookiecutter key."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        context = {'project_name': 'My Project'}
        
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, context)


def test_dump_creates_directory():
    """Test dump creates replay_dir if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = os.path.join(temp_dir, 'new_dir', 'nested')
        template_name = 'test_template'
        context = {'cookiecutter': {'key': 'value'}}
        
        dump(replay_dir, template_name, context)
        
        assert os.path.exists(replay_dir)
        expected_file = os.path.join(replay_dir, f'{template_name}.json')
        assert os.path.exists(expected_file)


def test_dump_with_path_object():
    """Test dump works with Path object as replay_dir."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir)
        template_name = 'test_template'
        context = {'cookiecutter': {'key': 'value'}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f'{template_name}.json'
        assert expected_file.exists()


def test_dump_overwrites_existing_file():
    """Test dump overwrites existing replay file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = temp_dir
        template_name = 'test_template'
        context1 = {'cookiecutter': {'version': '1'}}
        context2 = {'cookiecutter': {'version': '2'}}
        
        dump(replay_dir, template_name, context1)
        dump(replay_dir, template_name, context2)
        
        replay_file = os.path.join(replay_dir, f'{template_name}.json')
        with open(replay_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context2


# LLM-generated content at query #56
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_dump():
    """Test dump function writes context to json file correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project",
                "author": "John Doe"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created with correct name
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        # Verify file contains correct content
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


def test_dump_with_json_suffix():
    """Test dump function when template_name already has .json suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify no double .json extension
        expected_file = os.path.join(replay_dir, template_name)
        assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key():
    """Test dump raises ValueError when cookiecutter key is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {
            "project_name": "my_project"
        }
        
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, context)


def test_dump_creates_replay_dir():
    """Test dump creates replay directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "replay")
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        assert os.path.exists(replay_dir)
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)


def test_dump_with_string_replay_dir():
    """Test dump works with string path for replay_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)


# LLM-generated content at query #57
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names that already end with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation


# LLM-generated content at query #58
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #59
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #60
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create context without cookiecutter key
    context = {"some_key": "some_value"}
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template names that already have .json extension."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


# LLM-generated content at query #61
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template name with .json suffix."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    assert replay_file.endswith("test_template.json")


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    mock_make_sure = mocker.patch("cookiecutter.replay.make_sure_path_exists")
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path):
    """Test dump writes properly formatted json with indentation."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #62
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify the file was created with correct name
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify the content is correctly written
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify no double .json extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function ensures replay directory exists."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with correct directory
    mock_make_sure.assert_called_once_with(replay_dir)


# LLM-generated content at query #63
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template_name already having .json extension."""
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.endswith('test_template.json')
    assert os.path.exists(replay_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, 'new_dir')
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)


def test_dump_indentation(tmp_path):
    """Test dump function writes json with proper indentation."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert '  ' in content


# LLM-generated content at query #64
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        },
        "extra_key": "extra_value"
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test: load should return the context
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_extension(tmp_path):
    """Test load function handles template names with .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    valid_context = {
        "cookiecutter": {"key": "value"}
    }
    
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when context lacks cookiecutter key."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    invalid_context = {
        "some_key": "some_value"
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test: should raise ValueError
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test: should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #65
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    test_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already containing .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    test_context = {
        "cookiecutter": {
            "project_name": "another_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #66
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json file and returns context."""
    import json
    from pathlib import Path
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'my_project'}
    
    # Create replay file with invalid context
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name ending in .json."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #67
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author_name': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_with_json_extension(tmp_path):
    """Test load function with .json extension in template name."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {'some_key': 'some_value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent-template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #68
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_load():
    """Test load function reads and returns valid context from json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'My Project',
                'author': 'Test Author'
            }
        }
        
        # Create test json file
        replay_file = os.path.join(tmpdir, f'{template_name}.json')
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        # Test load function
        result = load(tmpdir, template_name)
        
        assert result == context
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'My Project'


def test_load_missing_cookiecutter_key():
    """Test load raises ValueError when context lacks cookiecutter key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = 'test_template'
        context = {'other_key': 'value'}
        
        # Create test json file without cookiecutter key
        replay_file = os.path.join(tmpdir, f'{template_name}.json')
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        # Test load raises ValueError
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            load(tmpdir, template_name)


def test_load_file_not_found():
    """Test load raises FileNotFoundError when replay file does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = 'nonexistent_template'
        
        # Test load raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load(tmpdir, template_name)


def test_load_with_json_extension():
    """Test load function works when template_name already has .json extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = 'test_template.json'
        context = {
            'cookiecutter': {
                'project_name': 'My Project'
            }
        }
        
        # Create test json file
        replay_file = os.path.join(tmpdir, template_name)
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        # Test load function
        result = load(tmpdir, template_name)
        
        assert result == context


def test_load_with_path_object():
    """Test load function works with Path object as replay_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'My Project'
            }
        }
        
        # Create test json file
        replay_file = replay_dir / f'{template_name}.json'
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        # Test load function
        result = load(replay_dir, template_name)
        
        assert result == context


# LLM-generated content at query #69
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with replay_dir
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct name
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    # Verify file contents
    with open(expected_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function when template_name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify no double .json suffix
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_readable_json(tmp_path, mocker):
    """Test dump function creates properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    content = expected_file.read_text(encoding="utf-8")
    
    # Verify indentation is 2 spaces
    assert "  " in content
    # Verify valid JSON
    assert json.loads(content) == context


# LLM-generated content at query #70
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    # Create a valid replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test: load should successfully read and return context
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result


def test_load_missing_file(tmp_path):
    """Test load function raises error when file does not exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test: load should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "invalid_template"
    context = {"project_name": "test_project"}
    
    # Create an invalid replay file without cookiecutter key
    replay_file = replay_dir / "invalid_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test: load should raise ValueError
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function handles template name with .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create a replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test: load should handle template name with .json extension
    result = load(replay_dir, template_name)
    assert result == context


def test_load_invalid_json(tmp_path):
    """Test load function raises error for invalid json content."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "invalid_json"
    
    # Create a file with invalid json
    replay_file = replay_dir / "invalid_json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        f.write("invalid json content {")
    
    # Test: load should raise JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #71
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when context missing cookiecutter key."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file with invalid context
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works when template_name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #72
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create test file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_with_json_suffix(tmp_path):
    """Test load function with template_name already containing .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / "test_template.json.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "project_name": "my_project"
    }
    
    # Create test file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #73
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json context from file."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    
    # Create test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises JSONDecodeError when file contains invalid json."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create test file with invalid json
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        f.write("invalid json content {]")
    
    # Test and Assert
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name that already has .json extension."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #74
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Call dump function
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "test_project"


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_writes_formatted_json(tmp_path, mocker):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "nested": {
                "key": "value"
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    # Verify indentation (indent=2)
    assert "  " in content
    saved = json.loads(content)
    assert saved == context


# LLM-generated content at query #75
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and returns context from json file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load raises json.JSONDecodeError when file contains invalid json."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    
    # Create replay file with invalid json
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write('invalid json content {')
    
    # Execute & Assert
    with raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #76
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template name already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay" / "nested"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #77
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual filesystem operations
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with correct replay_dir
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_creates_json_file(tmp_path):
    """Test dump function creates a json file with correct content."""
    replay_dir = tmp_path
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    # Verify content is correct
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_adds_json_extension(tmp_path):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_no_double_extension(tmp_path):
    """Test dump function doesn't add double .json extension."""
    replay_dir = tmp_path
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    assert not (replay_dir / "my_template.json.json").exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "my_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_path_object(tmp_path):
    """Test dump works with Path objects."""
    replay_dir = tmp_path
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path
    template_name = "my_template"
    
    # Write initial context
    context1 = {"cookiecutter": {"key": "value1"}}
    dump(replay_dir, template_name, context1)
    
    # Write new context
    context2 = {"cookiecutter": {"key": "value2"}}
    dump(replay_dir, template_name, context2)
    
    # Verify only new context exists
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context2


# LLM-generated content at query #78
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads and validates context from json file."""
    import json
    from cookiecutter.replay import load
    
    # Test successful load
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load with template name already containing .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load raises JSONDecodeError for invalid json."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = 'test_template'
    
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        f.write('invalid json content {')
    
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #79
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'author': 'Test Author'
        }
    }
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    invalid_context = {
        'project_name': 'My Project'
    }
    
    # Create the replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent-template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template_name already has .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'My Project'
        }
    }
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #80
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {"key": "value"}
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #81
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author_name': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'some_other_key': 'value'
    }
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #82
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author_name": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file with invalid context
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load with template name that already has .json extension."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #83
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path):
    """Test dump function adds .json extension when not present."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function doesn't double .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_proper_json_formatting(tmp_path):
    """Test dump writes properly formatted JSON with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '  ' in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #84
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump handles template_name that already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file is created without double .json extension
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_creates_valid_json(tmp_path, mocker):
    """Test dump creates valid JSON file that can be read."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {
        "cookiecutter": {
            "name": "test",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify JSON is valid and properly formatted
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    loaded = json.loads(content)
    assert loaded == context
    assert "  " in content  # Check for indentation


# LLM-generated content at query #85
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    valid_context = {"cookiecutter": {"project_name": "my_project"}}
    
    # Create a valid replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(replay_dir, template_name)
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_invalid_json(tmp_path):
    """Test load function raises json.JSONDecodeError for invalid json."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        f.write("invalid json content {")
    
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)


# LLM-generated content at query #86
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already having .json suffix."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, "nonexistent", "replay")
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context_old = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    context_new = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, context_old)
    dump(replay_dir, template_name, context_new)
    
    replay_file = os.path.join(replay_dir, "test_template.json")
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context_new


def test_dump_with_path_object(tmp_path):
    """Test dump function works with pathlib.Path object."""
    from pathlib import Path
    
    replay_dir = Path(tmp_path)
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()


# LLM-generated content at query #87
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"some_key": "some_value"}
    
    # Create the replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works when template_name already has .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    # Create the replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #88
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with replay_dir
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct name
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify file contents
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created without double .json suffix
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function calls make_sure_path_exists."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_preserves_context_structure(tmp_path, mocker):
    """Test dump function preserves complex context structure."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["nested"]["key1"] == "value1"
    assert loaded_context["cookiecutter"]["nested"]["key2"] == ["item1", "item2"]


# LLM-generated content at query #89
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_without_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"invalid_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path):
    """Test dump handles template names that already end with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    assert not (replay_dir / "test_template.json.json").exists()


def test_dump_file_format(tmp_path):
    """Test dump writes valid json with proper formatting."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "name": "project",
            "version": "1.0.0"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation
    parsed = json.loads(content)
    assert parsed == context


# LLM-generated content at query #90
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_slug": "test_slug"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "invalid_key": "value"
    }
    
    # Create replay file with invalid context
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #91
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #92
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "my-template.json").exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names that already have .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    old_context = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    new_context = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my-template.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump function handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "list_of_dicts": [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"}
            ]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #93
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_extension(tmp_path):
    """Test load function handles template names with .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #94
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    # Setup
    monkeypatch.chdir(tmp_path)
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_extension(tmp_path):
    """Test load function works with template_name already having .json extension."""
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    # Create a replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    template_name = "nonexistent_template"
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #95
#--------------------------

```python
def test_dump(tmp_path):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify content
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created without double extension
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    # Verify indentation (indent=2 was used)
    assert "  " in content


# LLM-generated content at query #96
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and assert
    result = load(replay_dir, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}  # Missing 'cookiecutter' key
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #97
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Call dump
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct content
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Should not add another .json extension
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context missing cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "other_key": "value"
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_dir(tmp_path, mocker):
    """Test dump calls make_sure_path_exists to create replay directory."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('json.dump')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


# LLM-generated content at query #98
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'test_project'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #99
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from pathlib import Path
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name already containing .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #100
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    invalid_context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent-template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    test_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


# LLM-generated content at query #101
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates JSON context from replay file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #102
#--------------------------

```python
def test_load(tmp_path):
    """Test the load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = tmp_path / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(tmp_path, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test that load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template'
    invalid_context = {'project_name': 'test_project'}
    
    # Create replay file with invalid context
    replay_file = tmp_path / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)


def test_load_file_not_found(tmp_path):
    """Test that load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, 'nonexistent_template')


def test_load_with_json_extension(tmp_path):
    """Test load works correctly when template_name already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = tmp_path / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(tmp_path, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #103
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #104
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert os.path.exists(os.path.join(replay_dir, f"{template_name}.json"))


def test_dump_writes_valid_json(tmp_path):
    """Test dump writes valid JSON format."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    loaded = json.loads(content)
    assert loaded == context


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {"key": "value"}
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #2
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_load():
    """Test load function reads and validates json context from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test successful load with valid context
        template_name = 'test_template'
        context = {
            'cookiecutter': {
                'project_name': 'my_project',
                'author': 'John Doe'
            }
        }
        
        replay_file = os.path.join(tmpdir, f'{template_name}.json')
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        result = load(tmpdir, template_name)
        assert result == context
        assert 'cookiecutter' in result

        # Test load with template name that already has .json extension
        template_name_with_ext = 'test_template.json'
        replay_file_2 = os.path.join(tmpdir, template_name_with_ext)
        with open(replay_file_2, 'w', encoding='utf-8') as f:
            json.dump(context, f)
        
        result_2 = load(tmpdir, template_name_with_ext)
        assert result_2 == context

        # Test load raises ValueError when cookiecutter key is missing
        invalid_context = {'other_key': 'value'}
        invalid_replay_file = os.path.join(tmpdir, 'invalid_template.json')
        with open(invalid_replay_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_context, f)
        
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            load(tmpdir, 'invalid_template')

        # Test load with Path object as replay_dir
        result_3 = load(Path(tmpdir), template_name)
        assert result_3 == context

        # Test load raises FileNotFoundError when file doesn't exist
        with pytest.raises(FileNotFoundError):
            load(tmpdir, 'nonexistent_template')


# LLM-generated content at query #3
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    with open(expected_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function with template name already having .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "test_template.json"
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay" / "nested"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context_old = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    context_new = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, context_old)
    dump(replay_dir, template_name, context_new)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context_new


# LLM-generated content at query #4
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with replay_dir
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify the file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump handles template_name that already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file doesn't have double .json extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_valid_json(tmp_path, mocker):
    """Test dump creates valid JSON file with proper formatting."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "version": "1.0.0",
            "nested": {"key": "value"}
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    
    # Verify file content is valid JSON with indentation
    content = replay_file.read_text(encoding="utf-8")
    assert "  " in content  # Check for indentation
    
    parsed = json.loads(content)
    assert parsed == context


# LLM-generated content at query #5
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"some_key": "some_value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mocker.patch("cookiecutter.replay.make_sure_path_exists")
    
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with the replay_dir
    from cookiecutter.replay import make_sure_path_exists
    make_sure_path_exists.assert_called_once_with(replay_dir)


def test_dump_with_json_suffix_in_template_name(tmp_path):
    """Test dump doesn't add .json suffix if template_name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_indented_json(tmp_path):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {"key": "value"}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    # Verify indentation is present (indent=2)
    assert "  " in content


# LLM-generated content at query #6
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    # Call dump function
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct name
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)
    
    # Verify file contents
    with open(expected_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "My Project"


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names with .json extension."""
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "My Project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file doesn't have double .json extension
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context missing cookiecutter key."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "My Project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, "nonexistent", "nested")
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path
    template_name = "test_template"
    context_1 = {
        "cookiecutter": {
            "project_name": "Project 1"
        }
    }
    context_2 = {
        "cookiecutter": {
            "project_name": "Project 2"
        }
    }
    
    # First dump
    dump(replay_dir, template_name, context_1)
    
    # Second dump
    dump(replay_dir, template_name, context_2)
    
    # Verify file contains second context
    replay_file = os.path.join(replay_dir, "test_template.json")
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context_2
    assert saved_context["cookiecutter"]["project_name"] == "Project 2"


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = os.path.join(replay_dir, "test_template.json")
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    # Verify indentation is present (indent=2)
    assert "  " in content


# LLM-generated content at query #7
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_template_name_with_json_extension(tmp_path, mocker):
    """Test dump handles template names that already end with .json."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert - should not double the .json extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_formatted_json(tmp_path, mocker):
    """Test dump creates properly formatted json with indentation."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert - verify indentation
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check for indentation (2 spaces per level)
    assert '  ' in content


# LLM-generated content at query #8
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_suffix(tmp_path, mocker):
    """Test dump function when template_name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Should not add extra .json suffix
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function calls make_sure_path_exists."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path, mocker):
    """Test dump function formats json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "nested": {
                "key": "value"
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    # Verify indentation (indent=2)
    assert "  " in content


# LLM-generated content at query #9
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'project_slug': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function works when template_name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'author': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #10
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "test_project"


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already having .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_writes_valid_json(tmp_path):
    """Test dump writes valid JSON with proper formatting."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    saved_context = json.loads(content)
    assert saved_context == context


# LLM-generated content at query #11
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Arrange
    replay_dir = tmp_path
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)
    
    with open(expected_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already has .json suffix."""
    # Arrange
    replay_dir = tmp_path
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    # Arrange
    replay_dir = tmp_path
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    # Act & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    # Arrange
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    assert replay_dir.exists()
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted JSON with indentation."""
    # Arrange
    replay_dir = tmp_path
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = os.path.join(replay_dir, "my_template.json")
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #12
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == "my_project"
    assert result['cookiecutter']['author'] == "John Doe"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file does not exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works correctly when template_name already has .json extension."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = os.path.join(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #13
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author_name": "Test Author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump handles template names with .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file is created without double .json extension
    replay_file = tmp_path / "replay" / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists to create replay directory."""
    replay_dir = tmp_path / "nonexistent_replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('json.dump')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json replay file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author_name': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test: load should return context
    result = load(replay_dir, template_name)
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    invalid_context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test: load should raise ValueError
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent-template'
    
    # Test: load should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test: load should return context
    result = load(replay_dir, template_name)
    assert result == context


# LLM-generated content at query #15
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Act
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template names that already have .json suffix."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Act
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {
        'other_key': 'value'
    }
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Act & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    # Arrange
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #16
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    with open(expected_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context['cookiecutter']['project_name'] == "my_project"


def test_dump_without_json_suffix(tmp_path, mocker):
    """Test dump function handles template names ending with .json."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = replay_dir / template_name
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function calls make_sure_path_exists."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path, mocker):
    """Test dump function writes json with proper indentation."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '  ' in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #17
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_extension(tmp_path):
    """Test load function with template_name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f, indent=2)
    
    # Execute and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #18
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "other_key": "value"
    }
    
    # Create the replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #19
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template name already ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay" / "nested"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch("cookiecutter.replay.make_sure_path_exists")
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_with_string_path(tmp_path):
    """Test dump works with string path instead of Path object."""
    replay_dir = str(tmp_path / "replay")
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(replay_file)


# LLM-generated content at query #20
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["project_name"] == "my_project"


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already having .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context_old = {
        "cookiecutter": {
            "project_name": "old_project"
        }
    }
    context_new = {
        "cookiecutter": {
            "project_name": "new_project"
        }
    }
    
    dump(replay_dir, template_name, context_old)
    dump(replay_dir, template_name, context_new)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context["cookiecutter"]["project_name"] == "new_project"


# LLM-generated content at query #21
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct content
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.read_text(encoding="utf-8") == json.dumps(context, indent=2)


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.read_text(encoding="utf-8") == json.dumps(context, indent=2)


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump function doesn't double .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "test-template.json"
    assert expected_file.read_text(encoding="utf-8") == json.dumps(context, indent=2)


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_calls_make_sure_path_exists(tmp_path, mocker):
    """Test dump calls make_sure_path_exists with correct directory."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_with_complex_context(tmp_path, mocker):
    """Test dump handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "complex-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "level1": {
                    "level2": ["item1", "item2"]
                }
            },
            "list": [1, 2, 3],
            "bool": True,
            "null": None
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    loaded = json.loads(expected_file.read_text(encoding="utf-8"))
    assert loaded == context


# LLM-generated content at query #22
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template_name ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_complex_context(tmp_path):
    """Test dump with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "options": {
                "use_docker": True,
                "python_version": "3.9"
            },
            "dependencies": ["pytest", "black"]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    with open(replay_dir / f"{template_name}.json", encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context["cookiecutter"]["options"]["use_docker"] is True


# LLM-generated content at query #23
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test the dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to verify it's called
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = tmp_path / "replay" / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump adds .json extension when template_name doesn't have it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / "my_template.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "replay" / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "new_replay_dir" / "test_template.json"
    assert replay_file.exists()


def test_dump_preserves_context_structure(tmp_path, mocker):
    """Test dump preserves complex context structure."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "number": 42
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    with open(tmp_path / "replay" / "complex_template.json", 'r', encoding='utf-8') as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #24
#--------------------------

```python
def test_dump(tmp_path, monkeypatch):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify content
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context missing cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, monkeypatch):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_template_name_ending_in_json(tmp_path):
    """Test dump doesn't add extra .json extension if template_name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    assert not (replay_dir / f"{template_name}.json").exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            },
            "numbers": [1, 2, 3]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #25
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create the replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "project_name": "my_project"
    }
    
    # Create the replay file with invalid context
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template names that already have .json extension."""
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create the replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


# LLM-generated content at query #26
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.write_text(json.dumps(expected_context), encoding="utf-8")
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create test file
    replay_file = replay_dir / template_name
    replay_file.write_text(json.dumps(expected_context), encoding="utf-8")
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {
        "other_key": "value"
    }
    
    # Create test file
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.write_text(json.dumps(invalid_context), encoding="utf-8")
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #27
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json replay file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function with .json suffix in template name."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #28
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    expected_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(expected_file)
    
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function when template_name already has .json suffix."""
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = os.path.join(tmp_path, 'new_dir')
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))


def test_dump_writes_valid_json(tmp_path):
    """Test dump function writes valid JSON with proper formatting."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'nested': {
                'key': 'value'
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = os.path.join(replay_dir, 'test_template.json')
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    # Verify it's valid JSON
    loaded = json.loads(content)
    assert loaded == context
    # Verify indentation
    assert '  ' in content


# LLM-generated content at query #29
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "my_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_nested_context(tmp_path):
    """Test dump function handles nested context data."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "settings": {
                "debug": True,
                "database": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #30
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Test successful load with valid context
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "project_slug": "my_project"
        }
    }
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    
    # Test load with template_name already ending in .json
    template_name_with_json = "test_template.json"
    replay_file_with_json = replay_dir / template_name_with_json
    with open(replay_file_with_json, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name_with_json)
    assert result == context
    
    # Test load raises ValueError when cookiecutter key is missing
    invalid_context = {"project_name": "my_project"}
    invalid_replay_file = replay_dir / "invalid_template.json"
    with open(invalid_replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")
    
    # Test load raises FileNotFoundError for non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "nonexistent_template")


# LLM-generated content at query #31
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates json context from file."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "My Project",
            "author": "Test Author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"other_key": "value"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already containing .json extension."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #32
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path):
    """Test dump function adds .json extension when not present."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path):
    """Test dump function doesn't duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / f"{template_name}.json").exists()


def test_dump_writes_proper_json_format(tmp_path):
    """Test dump writes valid JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "version": "1.0.0"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert json.loads(content) == context
    assert "  " in content


# LLM-generated content at query #33
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {
        'cookiecutter': {
            'project_name': 'My Project',
            'author': 'Test Author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'My Project'


def test_load_with_json_extension(tmp_path):
    """Test load function with .json extension in template name."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template.json'
    context = {
        'cookiecutter': {
            'project_name': 'My Project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test-template'
    context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test-template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'non-existent-template'
    
    # Test and assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #34
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test the load function."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test successful load
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context
    assert "cookiecutter" in loaded_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"


def test_load_with_json_suffix(tmp_path):
    """Test load function when template_name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads and validates context from json file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    # Create a replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works when template_name already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test successful load
    result = load(replay_dir, template_name)
    assert result == context


# LLM-generated content at query #36
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #37
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Call dump function
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path, mocker):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Should not add another .json suffix
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists with correct directory."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_with_complex_context(tmp_path, mocker):
    """Test dump with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context


# LLM-generated content at query #38
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        },
        "other_key": "other_value"
    }
    
    # Create a test replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["other_key"] == "other_value"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    context = {
        "project_name": "my_project"
    }
    
    # Create a test replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template_name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create a test replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert result["cookiecutter"]["project_name"] == "my_project"


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #39
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Execute
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump function doesn't add duplicate .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists for replay directory."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_json_formatting(tmp_path, mocker):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {"key": "value"}
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    content = replay_file.read_text(encoding='utf-8')
    
    # Verify proper indentation (indent=2)
    assert '  ' in content


# LLM-generated content at query #40
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'project_name': 'my_project'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #41
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation in test
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)
    
    # Verify file was created with correct content
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context['cookiecutter']['project_name'] == "my_project"


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function adds .json extension when not present."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump function doesn't add duplicate .json extension."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}  # Missing 'cookiecutter' key
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists with correct replay_dir."""
    # Arrange
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    # Act
    dump(replay_dir, template_name, context)
    
    # Assert
    mock_make_sure.assert_called_once_with(replay_dir)


# LLM-generated content at query #42
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "my_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "my_template"
    invalid_context = {"project_name": "test_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function works with template name already having .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "my_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #43
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct name
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    
    # Verify file contents
    with open(expected_file, 'r', encoding='utf-8') as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_without_json_extension(tmp_path, mocker):
    """Test dump function adds .json extension if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Should not have double .json extension
    expected_file = replay_dir / "my-template.json"
    assert expected_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"project_name": "test_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_writes_with_indent(tmp_path, mocker):
    """Test dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify indentation exists (indent=2)
    assert '  ' in content


# LLM-generated content at query #44
#--------------------------

```python
def test_load(tmp_path, mocker):
    """Test load function reads json file and returns context with cookiecutter key."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    expected_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load function with template name already having .json suffix."""
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #45
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test the load function."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test: load should return the context
    result = load(replay_dir, template_name)
    assert result == context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "my_project"
    assert result["cookiecutter"]["author"] == "John Doe"


def test_load_with_json_suffix(tmp_path):
    """Test the load function with .json suffix in template_name."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "another_project"
        }
    }
    
    # Create a replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test: load should handle template names with .json suffix
    result = load(replay_dir, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test the load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "other_key": "value"
    }
    
    # Create a replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    
    # Test: load should raise ValueError
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test the load function raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test: load should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #46
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "test_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_suffix(tmp_path):
    """Test load works with template name that already has .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


# LLM-generated content at query #47
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project"
    }
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #48
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    assert not replay_dir.exists()
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_without_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_with_json_suffix(tmp_path):
    """Test dump handles template_name that already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


def test_dump_with_complex_context(tmp_path):
    """Test dump with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "key1": "value1",
                "key2": ["item1", "item2"]
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #49
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Mock make_sure_path_exists to avoid actual directory creation
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file was created with correct name
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    # Verify file contents
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context['cookiecutter']['project_name'] == 'my_project'


def test_dump_without_cookiecutter_key(tmp_path, mocker):
    """Test dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "my_project"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_with_json_extension(tmp_path, mocker):
    """Test dump handles template names that already have .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    # Verify file doesn't have double .json extension
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump calls make_sure_path_exists with correct directory."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    mock_make_sure = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_with_complex_context(tmp_path, mocker):
    """Test dump handles complex nested context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "options": {
                "use_docker": True,
                "python_version": "3.9"
            },
            "modules": ["auth", "api", "db"]
        }
    }
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert saved_context['cookiecutter']['options']['python_version'] == '3.9'


# LLM-generated content at query #50
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template_name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(expected_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #51
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'project_slug': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test load function
    result = load(replay_dir, template_name)
    
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {'some_key': 'some_value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template_name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    test_context = {'cookiecutter': {'key': 'value'}}
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Test load function
    result = load(replay_dir, template_name)
    
    assert result == test_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #52
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template name already ending in .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay" / "subdir"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    mock_make_sure = mocker.patch("cookiecutter.replay.make_sure_path_exists")
    
    dump(replay_dir, template_name, context)
    
    mock_make_sure.assert_called_once_with(replay_dir)


def test_dump_formatting(tmp_path):
    """Test dump function writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    content = replay_file.read_text(encoding="utf-8")
    
    assert "  " in content  # Check for indentation
    assert json.loads(content) == context


# LLM-generated content at query #53
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create test context with required cookiecutter key
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Write test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test: load should return the context
    result = load(replay_dir, template_name)
    assert result == test_context
    assert "cookiecutter" in result
    assert result["cookiecutter"]["project_name"] == "test_project"


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    
    # Create test context without required cookiecutter key
    test_context = {
        "project_name": "test_project"
    }
    
    # Write test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test: load should raise ValueError
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function handles template names with .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    
    test_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Write test file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Test: load should work with .json extension
    result = load(replay_dir, template_name)
    assert result == test_context


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"
    
    # Test: load should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #54
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_json_suffix(tmp_path):
    """Test load function handles template names with .json suffix."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #55
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads and validates json context from file."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test load returns correct context
    result = load(replay_dir, template_name)
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'test_project'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    import pytest
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test load raises ValueError
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load works with template name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Test load returns correct context
    result = load(replay_dir, template_name)
    assert result == context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when file does not exist."""
    from cookiecutter.replay import load
    import pytest
    
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Test load raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #56
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load raises ValueError when cookiecutter key is missing."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    invalid_context = {"project_name": "my_project"}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function with template name that already has .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "test_template.json"
    expected_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)
    
    # Test
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == expected_context


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError when replay file does not exist."""
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #57
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test dump function handles template names ending with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / template_name
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test dump creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "new_replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_writes_valid_json(tmp_path):
    """Test dump writes properly formatted json with indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "nested": {
                "key": "value"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation
    saved_context = json.loads(content)
    assert saved_context == context


# LLM-generated content at query #58
#--------------------------

```python
def test_load(tmp_path, monkeypatch):
    """Test load function reads json data from file correctly."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_json_extension(tmp_path):
    """Test load function with template name already having .json extension."""
    import json
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    test_context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create the replay file
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == test_context


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    invalid_context = {
        'project_name': 'my_project'
    }
    
    # Create the replay file without cookiecutter key
    replay_file = replay_dir / 'test_template.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    import pytest
    from cookiecutter.replay import load
    
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #59
#--------------------------

```python
def test_load(tmp_path):
    """Test load function reads json data from file correctly."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result
    assert result['cookiecutter']['project_name'] == 'my_project'
    assert result['cookiecutter']['author'] == 'John Doe'


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'other_key': 'value'}
    
    # Create replay file without cookiecutter key
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when replay file doesn't exist."""
    # Setup
    replay_dir = tmp_path
    template_name = 'nonexistent_template'
    
    # Execute & Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


def test_load_with_json_extension(tmp_path):
    """Test load function works with template name that already has .json extension."""
    import json
    
    # Setup
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    # Create replay file
    replay_file = replay_dir / template_name
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Execute
    result = load(replay_dir, template_name)
    
    # Assert
    assert result == context
    assert 'cookiecutter' in result


# LLM-generated content at query #60
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

import pytest


def test_dump():
    """Test dump function writes context to json file correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project",
                "author": "John Doe"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created with correct name
        expected_file = os.path.join(replay_dir, "test_template.json")
        assert os.path.exists(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r', encoding="utf-8") as f:
            saved_context = json.load(f)
        
        assert saved_context == context
        assert saved_context["cookiecutter"]["project_name"] == "my_project"


def test_dump_with_json_suffix():
    """Test dump function when template_name already has .json suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template.json"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        # Verify file was created without double .json suffix
        expected_file = os.path.join(replay_dir, "test_template.json")
        assert os.path.exists(expected_file)


def test_dump_missing_cookiecutter_key():
    """Test dump raises ValueError when context lacks cookiecutter key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {"project_name": "my_project"}
        
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, context)


def test_dump_creates_replay_dir():
    """Test dump creates replay directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "replay")
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, "test_template.json")
        assert os.path.exists(expected_file)


def test_dump_with_path_object():
    """Test dump works with Path object as replay_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test_template.json"
        assert expected_file.exists()


def test_dump_indentation():
    """Test dump writes json with proper indentation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "my_project",
                "author": "John Doe"
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, "test_template.json")
        with open(expected_file, 'r', encoding="utf-8") as f:
            content = f.read()
        
        # Verify indentation (should have 2 spaces)
        assert "  " in content


# LLM-generated content at query #61
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to json file correctly."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test dump function with template_name already having .json suffix."""
    replay_dir = tmp_path
    template_name = 'test_template.json'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    assert replay_file.endswith('test_template.json')


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {'project_name': 'my_project'}
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path, mocker):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / 'new_dir'
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)


def test_dump_json_formatting(tmp_path):
    """Test dump function writes properly formatted json with indent."""
    replay_dir = tmp_path
    template_name = 'test_template'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'nested': {'key': 'value'}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '  ' in content  # Check for indentation
    loaded = json.loads(content)
    assert loaded == context


# LLM-generated content at query #62
#--------------------------

```python
def test_dump(tmp_path, mocker):
    """Test dump function writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "Test Author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_creates_directory(tmp_path):
    """Test dump function creates replay directory if it doesn't exist."""
    replay_dir = tmp_path / "non_existent" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "template.json").exists()


def test_dump_with_json_suffix(tmp_path):
    """Test dump function handles template names with .json suffix."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_missing_cookiecutter_key(tmp_path):
    """Test dump function raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"project_name": "test"}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_overwrites_existing_file(tmp_path):
    """Test dump function overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "template.json"
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == new_context


def test_dump_complex_context(tmp_path):
    """Test dump function with complex nested context."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "nested": {
                "level1": {
                    "level2": ["item1", "item2"]
                }
            },
            "list": [1, 2, 3],
            "boolean": True,
            "null_value": None
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex_template.json"
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


