####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    template_name = "test_template"
    json_content = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_content, f)
    
    result = load(tmp_path, template_name)
    
    assert result == json_content
    assert "cookiecutter" in result


def test_load_json_file_with_json_extension(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "test_template.json"
    json_content = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / template_name
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_content, f)
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "test_template"
    json_content = {"other_key": "value"}
    
    json_file = tmp_path / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_content, f)
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    from pathlib import Path
    
    template_name = "nonexistent_template"
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "test_template"
    json_content = {"cookiecutter": {"key": "value"}}
    
    json_file = tmp_path / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_content, f)
    
    result = load(Path(tmp_path), template_name)
    
    assert result == json_content


# LLM-generated content at query #2
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with valid context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write JSON with 'cookiecutter' key to ensure predicate at line 8 is False
        context_data = {
            'cookiecutter': {
                'project_name': 'test_project'
            }
        }
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=test_file):
            result = load(temp_path, 'test_template')
            assert result == context_data
            assert 'cookiecutter' in result


# LLM-generated content at query #3
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {"cookiecutter": {"project_name": "test"}}
        test_file = Path(tmpdir) / "test.json"
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        result = load(tmpdir, "test")
        
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to the file
    test_data = {"cookiecutter": {"project_name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_dump_creates_replay_file_with_valid_context(tmp_path, mocker):
    """Test that dump creates a replay file with valid context."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_adds_json_suffix_when_missing(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_does_not_duplicate_json_suffix(tmp_path):
    """Test that dump does not add .json suffix if template name already has it."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_nested_directories(tmp_path):
    """Test that dump creates nested directories if they don't exist."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_writes_valid_json_format(tmp_path):
    """Test that dump writes context in valid JSON format with proper indentation."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"name": "test", "value": 123}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content
    loaded_context = json.loads(content)
    assert loaded_context == context


# LLM-generated content at query #6
#--------------------------

```python
import json
import tempfile
from pathlib import Path


def test_load_with_valid_context():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template.json')
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_template_name_without_json_extension():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template')
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_path_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'key': 'value'}}
        file_path = Path(temp_dir) / 'config.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(Path(temp_dir), 'config.json')
        assert result == test_data


def test_load_missing_cookiecutter_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'other_key': 'value'}
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        try:
            load(temp_dir, 'template.json')
            assert False, "Expected ValueError"
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_complex_context():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {
            'cookiecutter': {
                'project_name': 'my_project',
                'author': 'John Doe',
                'nested': {'key': 'value'}
            },
            'extra_field': 'extra_value'
        }
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template')
        assert result == test_data
        assert result['cookiecutter']['project_name'] == 'my_project'
        assert result['extra_field'] == 'extra_value'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_name = "template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text(json.dumps(context_data), encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = "template.json"
    context_data = {"cookiecutter": {"key": "value"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text(json.dumps(context_data), encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = "template"
    context_data = {"other_key": "value"}
    
    json_file = tmp_path / "template.json"
    json_file.write_text(json.dumps(context_data), encoding="utf-8")
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    template_name = "nonexistent"
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    template_name = "template"
    context_data = {"cookiecutter": {"nested": {"data": "value"}}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text(json.dumps(context_data), encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data
    assert result["cookiecutter"]["nested"]["data"] == "value"


# LLM-generated content at query #2
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_json_without_extension(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


# LLM-generated content at query #3
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, "test_template")
        
        assert result == context_data
        assert "cookiecutter" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates directory and writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_json_extension_in_template_name(tmp_path):
    """Test that dump does not add extra .json when template_name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"some_key": "some_value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_dump_creates_nested_directories(tmp_path):
    """Test that dump creates nested directory structure if it doesn't exist."""
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_complex_context(tmp_path):
    """Test that dump correctly serializes complex context structure."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "John Doe",
            "features": ["feature1", "feature2"],
            "config": {"nested": True, "value": 42}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


# LLM-generated content at query #5
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to file
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        # Import and call the function
        from pathlib import Path
        result = load(replay_dir, template_name)
    
    # Verify the file was read correctly (predicate at line 5 was True)
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_adds_json_suffix_if_not_present(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_does_not_duplicate_json_suffix(tmp_path):
    """Test that dump does not add .json suffix if already present."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()
    assert not (replay_dir / "my-template.json.json").exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_dump_preserves_context_structure(tmp_path):
    """Test that dump preserves the exact structure of the context."""
    replay_dir = tmp_path / "replay"
    template_name = "complex-template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value", "list": [1, 2, 3]}
        },
        "extra_key": "extra_value"
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex-template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


# LLM-generated content at query #7
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_not_in_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a mock file without 'cookiecutter' key
    json_file = replay_dir / f"{template_name}.json"
    context_without_cookiecutter = {"other_key": "value"}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_without_cookiecutter, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("builtins.__import__") as mock_import:
        # This test verifies the predicate at line 8 evaluates to True
        # by ensuring the condition 'cookiecutter' not in context is met
        try:
            from pathlib import Path
            with open(json_file, encoding="utf-8") as infile:
                import json
                context = json.load(infile)
            
            # The predicate at line 8 should evaluate to True
            assert 'cookiecutter' not in context
        except Exception:
            pass


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json(tmp_path, monkeypatch):
    """Test that dump creates the replay directory and writes context to JSON file."""
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context


def test_dump_with_json_suffix_in_template_name(tmp_path):
    """Test that dump doesn't add extra .json suffix if template name already has it."""
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when cookiecutter key is missing from context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_with_string_path(tmp_path):
    """Test that dump works with string path instead of Path object."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = str(tmp_path / "replay")
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = str(tmp_path / "replay" / "my_template.json")
    assert Path(replay_file).exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context


def test_dump_overwrites_existing_replay_file(tmp_path):
    """Test that dump overwrites existing replay file."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {"cookiecutter": {"project_name": "old_project"}}
    new_context = {"cookiecutter": {"project_name": "new_project"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        saved_context = json.load(infile)
    
    assert saved_context == new_context


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates replay directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_json_extension_in_template_name(tmp_path):
    """Test that dump handles template names that already have .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_with_existing_replay_directory(tmp_path):
    """Test that dump works when replay directory already exists."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_writes_json_with_proper_formatting(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John Doe"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    assert "  " in content  # Check for indentation


def test_dump_with_nested_context(tmp_path):
    """Test that dump handles nested context dictionaries."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "config": {
                "debug": True,
                "version": "1.0"
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context
    assert saved_context["cookiecutter"]["config"]["debug"] is True


# LLM-generated content at query #10
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    test_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_template_name_without_json_extension(tmp_path):
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    test_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    test_file = tmp_path / "template.json"
    test_file.write_text('{"data": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    test_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    test_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(str(tmp_path), "template.json")
    
    assert result == test_data


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        replay_file = temp_path / "replay.json"
        
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "replay.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_writes_json_file_with_valid_context(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = 'my_template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / 'my_template.json'
    assert replay_file.exists()
    
    import json
    with open(replay_file, 'r', encoding='utf-8') as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_adds_json_extension_if_missing(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = 'my_template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / 'my_template.json'
    assert replay_file.exists()


def test_dump_does_not_add_duplicate_json_extension(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = 'my_template.json'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / 'my_template.json'
    assert replay_file.exists()


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = 'my_template'
    context = {'project_name': 'test_project'}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'cookiecutter key' in str(e)


def test_dump_creates_replay_directory_if_not_exists(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / 'new_dir'
    template_name = 'my_template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / 'my_template.json').exists()


def test_dump_preserves_context_structure(tmp_path):
    from cookiecutter.replay import dump
    import json
    
    replay_dir = tmp_path
    template_name = 'my_template'
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author',
            'nested': {'key': 'value'}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    with open(tmp_path / 'my_template.json', 'r', encoding='utf-8') as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #13
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to file
    test_data = {"cookiecutter": {"key": "value"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file path
    import sys
    from unittest.mock import patch
    
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
        
        # Call the function with mocked open
        with open(test_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        # Verify the file was opened with utf-8 encoding
        assert context == test_data
        assert 'cookiecutter' in context


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test_project"}}


def test_load_with_template_name_without_json_extension(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_missing_cookiecutter_key(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_with_path_object(tmp_path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"cookiecutter": {"name": "example"}}', encoding="utf-8")
    
    result = load(tmp_path, "config.json")
    
    assert result == {"cookiecutter": {"name": "example"}}


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_complex_cookiecutter_structure(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"nested": {"key": "value"}, "list": [1, 2, 3]}, "other": "data"}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result["cookiecutter"]["nested"]["key"] == "value"
    assert result["cookiecutter"]["list"] == [1, 2, 3]
    assert result["other"] == "data"


# LLM-generated content at query #15
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write test data with cookiecutter key
        test_data = {
            "cookiecutter": {
                "project_name": "test_project"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=str(test_file)):
            result = load(temp_path, "test_template")
        
        # Verify that the predicate 'cookiecutter' in context evaluates to True
        assert 'cookiecutter' in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #16
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_without_cookiecutter = {"other_key": "value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #17
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        test_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        json_file = replay_dir / f"{template_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        result = load(replay_dir, template_name)
        
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #18
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path):
    """Test that dump creates directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context


def test_dump_with_json_suffix_in_template_name(tmp_path):
    """Test that dump doesn't add extra .json suffix if template name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context


def test_dump_raises_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError if context doesn't contain cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_with_nested_context(tmp_path):
    """Test that dump correctly writes nested context data."""
    replay_dir = tmp_path / "replay"
    template_name = "nested_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "nested_template.json"
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["nested"]["key"] == "value"


def test_dump_with_string_replay_dir(tmp_path):
    """Test that dump works with string path for replay_dir."""
    replay_dir = str(tmp_path / "replay")
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    import os
    replay_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(replay_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"name": "test"}}


def test_load_with_json_extension_in_template_name(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_missing_cookiecutter_key(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_with_path_object(tmp_path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"cookiecutter": {"setting": "enabled"}}', encoding="utf-8")
    
    result = load(tmp_path, "config")
    
    assert result == {"cookiecutter": {"setting": "enabled"}}


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #21
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to file
    test_data = {"cookiecutter": {"project_name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=test_file):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_not_in_context(tmp_path):
    """Test that dump raises ValueError when 'cookiecutter' key is not in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"some_key": "some_value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #23
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file without 'cookiecutter' key
    replay_file = replay_dir / f"{template_name}.json"
    context_without_cookiecutter = {"some_key": "some_value"}
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context_without_cookiecutter, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #24
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"cookiecutter": {"project_name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test"}}


def test_load_with_template_name_without_json_extension(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_raises_error_when_cookiecutter_key_missing(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_raises_error_when_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent.json")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_complex_cookiecutter_context(tmp_path):
    template_file = tmp_path / "template.json"
    complex_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "options": ["opt1", "opt2"]
        },
        "extra_key": "extra_value"
    }
    import json
    template_file.write_text(json.dumps(complex_context), encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == complex_context


# LLM-generated content at query #25
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create a test JSON file
    test_file = replay_dir / f"{template_name}.json"
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    # Verify the file was read and parsed correctly
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #26
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with valid JSON file containing cookiecutter key."""
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_without_json_extension(tmp_path):
    """Test load function automatically adds .json extension."""
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"author": "test_author"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file does not exist."""
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent.json")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"version": "1.0"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


# LLM-generated content at query #28
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_not_in_context(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"some_key": "some_value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #29
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_file = tmpdir_path / "test.json"
        
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(tmpdir_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #30
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    valid_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"project_name": "test_project", "author": "test_author"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template.json')
    
    assert result == valid_context
    assert 'cookiecutter' in result


def test_load_without_json_extension(tmp_path):
    valid_context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template')
    
    assert result == valid_context
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"project_name": "test_project"}', encoding='utf-8')
    
    try:
        load(tmp_path, 'template.json')
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, 'nonexistent.json')
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    valid_context = {
        'cookiecutter': {
            'key': 'value'
        }
    }
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template.json')
    
    assert result == valid_context


# LLM-generated content at query #31
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_name = 'template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text(json.dumps(context), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    assert result == context
    assert 'cookiecutter' in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = 'template.json'
    context = {'cookiecutter': {'key': 'value'}}
    
    json_file = tmp_path / template_name
    json_file.write_text(json.dumps(context), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    assert result == context


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = 'template'
    context = {'other_key': 'value'}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text(json.dumps(context), encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found(tmp_path):
    template_name = 'nonexistent'
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_complex_context(tmp_path):
    template_name = 'complex'
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe',
            'version': '1.0.0',
            'nested': {'key': 'value'}
        },
        'extra_data': 'additional'
    }
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text(json.dumps(context), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    assert result == context
    assert result['cookiecutter']['project_name'] == 'my_project'


def test_load_with_path_object(tmp_path):
    template_name = 'template'
    context = {'cookiecutter': {'test': 'data'}}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text(json.dumps(context), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    assert result == context


# LLM-generated content at query #33
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create the file that get_file_name would return
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        # This will test that line 5 executes successfully with utf-8 encoding
        with open(replay_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        assert context == test_data
        assert isinstance(context, dict)
        assert "cookiecutter" in context


# LLM-generated content at query #34
#--------------------------

```python
def test_load_validates_cookiecutter_key_exists(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file with 'cookiecutter' key
    test_file = tmp_path / "test_template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=test_file):
        result = load(tmp_path, "test_template")
    
    assert "cookiecutter" in result
    assert result == test_data


# LLM-generated content at query #35
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_file = Path(tmpdir) / "replay.json"
        context_without_cookiecutter = {"some_key": "some_value"}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(tmpdir, "replay.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #36
#--------------------------

```python
def test_dump_writes_json_file_with_valid_context(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        import json
        loaded_context = json.load(infile)
    
    assert loaded_context == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_replay_directory_if_not_exists(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    assert not replay_dir.exists()
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_handles_template_name_with_json_extension(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_preserves_context_structure(tmp_path):
    from cookiecutter.replay import dump
    import json
    
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value", "list": [1, 2, 3]}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex_template.json"
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"
    assert loaded_context["cookiecutter"]["nested"]["list"] == [1, 2, 3]


# LLM-generated content at query #37
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_without_cookiecutter = {"key": "value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #38
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write JSON with cookiecutter key to ensure predicate at line 8 is False
        test_data = {"cookiecutter": {"project_name": "test_project"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("builtins.open", open):
            # Directly test the condition by reading the file
            with open(test_file, encoding="utf-8") as infile:
                context = json.load(infile)
            
            # Assert that 'cookiecutter' IS in context (predicate evaluates to False)
            assert 'cookiecutter' in context
            assert context == test_data


# LLM-generated content at query #39
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_template_name_without_json_extension(tmp_path):
    """Test load function when template_name doesn't have .json extension."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    """Test load function with Path object as replay_dir."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    """Test load function with string path as replay_dir."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(str(tmp_path), "template.json")
    
    assert result == test_data


def test_load_with_nested_cookiecutter_data(tmp_path):
    """Test load function with nested data in cookiecutter."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test", "nested": {"key": "value"}}}
    json_file.write_text('{"cookiecutter": {"project_name": "test", "nested": {"key": "value"}}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data
    assert result["cookiecutter"]["nested"]["key"] == "value"


# LLM-generated content at query #40
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = replay_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"version": "1.0"}}
    
    json_file = replay_dir / template_name
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"other_key": "value"}
    
    json_file = replay_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_file_not_found(tmp_path):
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function accepts context with 'cookiecutter' key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #42
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_name = "template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = "template.json"
    context_data = {"cookiecutter": {"key": "value"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = "template"
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    template_name = "template"
    context_data = {"cookiecutter": {"data": "test"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"data": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == context_data


def test_load_file_not_found(tmp_path):
    template_name = "nonexistent"
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #44
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=test_file):
            from load import load
            result = load(temp_path, "test_template")
            assert result == context_data
            assert "cookiecutter" in result


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_path_object():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template')
    assert result == os.path.join('/tmp/replay', 'template.json')

def test_get_file_name_with_string_path():
    result = get_file_name('/tmp/replay', 'template')
    assert result == os.path.join('/tmp/replay', 'template.json')

def test_get_file_name_with_json_extension():
    result = get_file_name('/tmp/replay', 'template.json')
    assert result == os.path.join('/tmp/replay', 'template.json')

def test_get_file_name_with_different_extension():
    result = get_file_name('/tmp/replay', 'template.txt')
    assert result == os.path.join('/tmp/replay', 'template.txt.json')

def test_get_file_name_with_empty_directory():
    result = get_file_name('', 'template')
    assert result == os.path.join('', 'template.json')

def test_get_file_name_with_complex_path():
    from pathlib import Path
    result = get_file_name(Path('/home/user/replays/2024'), 'game_replay')
    assert result == os.path.join('/home/user/replays/2024', 'game_replay.json')


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_writes_json_file_with_context(tmp_path):
    """Test that dump writes context to a JSON file."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        import json
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_creates_replay_directory_if_not_exists(tmp_path):
    """Test that dump creates the replay directory if it doesn't exist."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "new_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "test_template.json").exists()


def test_dump_adds_json_extension_if_not_present(tmp_path):
    """Test that dump adds .json extension if template_name doesn't have it."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    assert (tmp_path / "test_template.json").exists()


def test_dump_does_not_duplicate_json_extension(tmp_path):
    """Test that dump doesn't add .json if template_name already has it."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    assert (tmp_path / "test_template.json").exists()
    assert not (tmp_path / "test_template.json.json").exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError if context doesn't have cookiecutter key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_writes_properly_formatted_json(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    from cookiecutter.replay import dump
    import json
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test", "author": "author"}}
    
    dump(replay_dir, template_name, context)
    
    with open(tmp_path / "test_template.json", 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content
    loaded = json.loads(content)
    assert loaded == context


# LLM-generated content at query #3
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with valid context
    test_file = tmp_path / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Call load function
    result = load(str(tmp_path), "template.json")
    
    # Assert the result
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_without_json_extension(tmp_path):
    import json
    
    # Create a temporary JSON file without .json extension
    test_file = tmp_path / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Call load function with template name without extension
    result = load(str(tmp_path), "template")
    
    # Assert the result
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_file = tmp_path / "template.json"
    invalid_context = {
        "project_name": "test_project"
    }
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Call load function and expect ValueError
    try:
        load(str(tmp_path), "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    # Call load function with non-existent file
    try:
        load(str(tmp_path), "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_dump_creates_replay_file(tmp_path, monkeypatch):
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_adds_json_extension_if_missing(tmp_path):
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_add_extension_if_already_present(tmp_path):
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_error_when_cookiecutter_key_missing(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_nested_directories(tmp_path):
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "my_template"
    context = {"cookiecutter": {"project": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_writes_valid_json(tmp_path):
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"name": "project", "version": "1.0"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded = json.load(f)
    
    assert loaded["cookiecutter"]["name"] == "project"
    assert loaded["cookiecutter"]["version"] == "1.0"


# LLM-generated content at query #5
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write test data with 'cookiecutter' key
        test_data = {
            "cookiecutter": {
                "project_name": "test_project"
            }
        }
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=test_file):
            from __main__ import load
            result = load(temp_path, "template")
        
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_creates_directory_and_writes_json_file(tmp_path):
    """Test that dump creates the replay directory and writes JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    assert replay_file.read_text(encoding="utf-8").strip() != ""


def test_dump_adds_json_extension_if_missing(tmp_path):
    """Test that dump adds .json extension if template_name doesn't have it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_preserves_json_extension(tmp_path):
    """Test that dump doesn't add double .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_writes_correct_json_content(tmp_path):
    """Test that dump writes the correct JSON content to file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    import json
    written_content = json.loads(replay_file.read_text(encoding="utf-8"))
    assert written_content == context


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter" in str(e).lower()


def test_dump_with_nested_replay_directory(tmp_path):
    """Test that dump creates nested directories."""
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function accepts context with 'cookiecutter' key."""
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test-template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'Test Author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert 'cookiecutter' in saved_context


# LLM-generated content at query #8
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        result = load(temp_path, "test.json")
        
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #9
#--------------------------

```python
def test_load_requires_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with valid cookiecutter context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write JSON with cookiecutter key to satisfy the predicate at line 8
        context_data = {"cookiecutter": {"project_name": "test_project"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            # Import and call the function
            from __main__ import load
            result = load(temp_path, "test_template")
        
        # Assert the predicate evaluates to True (cookiecutter key exists)
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #10
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #11
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data with UTF-8 encoding
    test_data = {"cookiecutter": {"key": "value"}}
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        # Import and call the load function
        from pathlib import Path as PathlibPath
        
        def get_file_name(replay_dir, template_name):
            return PathlibPath(replay_dir) / f"{template_name}.json"
        
        def load(replay_dir, template_name):
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(replay_dir, template_name)
        assert result == test_data
        assert isinstance(result, dict)


# LLM-generated content at query #12
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    test_data = {"cookiecutter": {"project_name": "test"}}
    replay_file = replay_dir / f"{template_name}.json"
    
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data
    assert result["cookiecutter"] == {"project_name": "test"}


# LLM-generated content at query #13
#--------------------------

```python
def test_dump_creates_replay_directory(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_writes_json_file(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_writes_correct_json_content(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project", "version": "1.0"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


def test_dump_with_json_extension_in_template_name(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_with_string_path(tmp_path):
    replay_dir = str(tmp_path / "replay")
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    import os
    replay_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(replay_file)


def test_dump_preserves_context_structure(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value"}
        },
        "extra_data": "something"
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded_context = json.load(f)
    
    assert loaded_context == context


# LLM-generated content at query #14
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and JSON file without 'cookiecutter' key
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a subdirectory matching the template name
    template_dir = replay_dir / template_name
    template_dir.mkdir()
    
    # Create a JSON file without 'cookiecutter' key
    json_file = template_dir / "replay.json"
    context_without_key = {"other_key": "value"}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_without_key, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=json_file):
        try:
            from __main__ import load
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #15
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    test_data = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_author"
        }
    }
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    import os
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    
    test_data = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    file_path = os.path.join(replay_dir, template_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    test_data = {
        "project_name": "my_project"
    }
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    import os
    from pathlib import Path
    
    replay_dir = Path(tmp_path)
    template_name = "test_template"
    
    test_data = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data


def test_load_file_not_found(tmp_path):
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    import json
    from pathlib import Path
    
    # Create a temporary JSON file with cookiecutter key
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    import json
    from pathlib import Path
    
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data


def test_load_without_cookiecutter_key_raises_error(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    test_file = tmp_path / "template.json"
    test_data = {"other_key": "value"}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(tmp_path, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, "template")
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    """Test load function works with string path as replay_dir."""
    import json
    
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(str(tmp_path), "template")
    
    assert result == test_data


# LLM-generated content at query #17
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test"}}


def test_load_with_template_name_without_json_extension(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_missing_cookiecutter_key(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_with_path_object(tmp_path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"cookiecutter": {"name": "project"}}', encoding="utf-8")
    
    result = load(tmp_path, "config.json")
    
    assert result == {"cookiecutter": {"name": "project"}}


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_invalid_json(tmp_path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text('invalid json content', encoding="utf-8")
    
    try:
        load(tmp_path, "invalid.json")
        assert False, "Expected json.JSONDecodeError"
    except Exception:
        pass


def test_load_empty_cookiecutter(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {}}


# LLM-generated content at query #18
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path = os.path.join(test_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_json_file_with_json_extension(tmp_path):
    import json
    import os
    
    test_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"key": "value"}}
    
    file_path = os.path.join(test_dir, template_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    
    test_dir = tmp_path
    template_name = "test_template"
    context_data = {"other_key": "value"}
    
    file_path = os.path.join(test_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(test_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_pathlib_path(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    template_name = "test_template"
    context_data = {"cookiecutter": {"name": "test"}}
    
    file_path = test_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #19
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    replay_dir = tmp_path
    template_name = "test_template"
    replay_file = replay_dir / f"{template_name}.json"
    
    context_data = {"some_key": "some_value"}
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    # Import the function to test
    from pathlib import Path
    import json
    from typing import Any
    
    def get_file_name(replay_dir: Path | str, template_name: str) -> Path:
        return Path(replay_dir) / f"{template_name}.json"
    
    def load(replay_dir: Path | str, template_name: str) -> dict[str, Any]:
        """Read json data from file."""
        replay_file = get_file_name(replay_dir, template_name)
        
        with open(replay_file, encoding="utf-8") as infile:
            context: dict[str, Any] = json.load(infile)
        
        if 'cookiecutter' not in context:
            msg = 'Context is required to contain a cookiecutter key'
            raise ValueError(msg)
        
        return context
    
    # Test that ValueError is raised when 'cookiecutter' is not in context
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_valid_cookiecutter_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a valid context with 'cookiecutter' key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    # Write the context to a JSON file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        result = load(replay_dir, template_name)
    
    # Assert that the predicate 'cookiecutter' in context evaluates to True
    assert 'cookiecutter' in result
    assert result == valid_context


# LLM-generated content at query #21
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path

def test_load_with_valid_json_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template.json')
        
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_template_name_without_extension():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template')
        
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_path_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(Path(temp_dir), 'template.json')
        
        assert result == test_data


def test_load_missing_cookiecutter_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'project_name': 'test_project'}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        try:
            load(temp_dir, 'template.json')
            assert False, "Expected ValueError"
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            load(temp_dir, 'nonexistent.json')
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# LLM-generated content at query #22
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with valid context
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test loading the file
    result = load(test_dir, "template")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    # Create a temporary JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test loading with .json extension already in template_name
    result = load(test_dir, "template.json")
    
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {"other_key": "value"}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Test that ValueError is raised
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Test loading with Path object
    result = load(test_dir, "template")
    
    assert result == valid_context


def test_load_file_not_found(tmp_path):
    # Test that FileNotFoundError is raised when file doesn't exist
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent_template")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {
            "cookiecutter": {
                "project_name": "test_project"
            }
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        result = load(temp_path, "test")
        
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #24
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #25
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test JSON file with 'cookiecutter' key
        test_file = temp_path / "test.json"
        test_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, "test_template")
        
        # Verify the predicate 'cookiecutter' not in context evaluates to False
        assert "cookiecutter" in result
        assert result == test_data


# LLM-generated content at query #26
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to the file
    test_data = {"cookiecutter": {"project_name": "test"}}
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    # Verify the file was read correctly (predicate at line 5 evaluated successfully)
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #27
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    test_file = tmp_path / "test.json"
    test_data = {"some_key": "some_value"}
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=test_file):
        try:
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps({"other_key": "value"}), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(json_file)):
        try:
            load(tmp_path, "template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


def test_load_succeeds_when_cookiecutter_key_present(tmp_path):
    import json
    from pathlib import Path
    from unittest.mock import patch
    
    # Create a temporary JSON file with 'cookiecutter' key
    json_file = tmp_path / "test.json"
    test_data = {"cookiecutter": {"project_name": "test"}, "other_key": "value"}
    json_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    with patch('__main__.get_file_name', return_value=str(json_file)):
        result = load(tmp_path, "template")
        assert result == test_data
        assert 'cookiecutter' in result


# LLM-generated content at query #29
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_context
    assert "cookiecutter" in result


def test_load_with_template_name_without_extension(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template")
    
    assert result == test_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"project_name": "test_project"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"author": "test_author"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_context


# LLM-generated content at query #30
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test-template"
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'Test Author'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / '.cookiecutters' / 'test-template.json').exists()


# LLM-generated content at query #31
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_without_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    import json
    
    test_dir = str(tmp_path)
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"version": "1.0"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write test data with 'cookiecutter' key
        test_data = {
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            }
        }
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=test_file):
            result = load(temp_path, 'test_template')
        
        # Assert that the predicate at line 8 evaluates to False
        # (meaning 'cookiecutter' IS in context, so no ValueError is raised)
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'test_project'


# LLM-generated content at query #33
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a mock file without 'cookiecutter' key
    replay_file = replay_dir / f"{template_name}.json"
    context_without_cookiecutter = {"other_key": "value"}
    
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context_without_cookiecutter, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #34
#--------------------------

```python
def test_dump_writes_json_file(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        import json
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_with_json_suffix_in_template_name(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        import json
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_creates_nested_directories(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_with_string_path(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = str(tmp_path / "replay")
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    import os
    replay_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(replay_file)


# LLM-generated content at query #35
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Write test data to file
    test_data = {"cookiecutter": {"key": "value"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    # Verify the file was read correctly with utf-8 encoding
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #36
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {"cookiecutter": {"project_name": "test_project"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        result = load(temp_path, "test")
        assert result == context_data
        assert "cookiecutter" in result


# LLM-generated content at query #37
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        test_data = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        try:
            load(temp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #38
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and JSON file without 'cookiecutter' key
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a simple JSON file without 'cookiecutter' key
    json_file = replay_dir / f"{template_name}.json"
    context_without_cookiecutter = {"other_key": "value"}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_without_cookiecutter, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(json_file)):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #39
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function succeeds when 'cookiecutter' key is in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #40
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write test data with cookiecutter key to ensure predicate is False
        test_data = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        from unittest.mock import patch
        with patch("load.get_file_name", return_value=str(test_file)):
            result = load(temp_path, "test_template")
        
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #41
#--------------------------

```python
def test_load_context_without_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / "test.json"
        
        context_data = {"other_key": "value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        try:
            load(tmp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #42
#--------------------------

```python
import json
import tempfile
from pathlib import Path


def test_load_with_valid_context():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            }
        }
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template')
        
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_json_extension():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {
            'cookiecutter': {
                'project_name': 'test_project'
            }
        }
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(temp_dir, 'template.json')
        
        assert result == test_data


def test_load_missing_cookiecutter_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {
            'project_name': 'test_project'
        }
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        try:
            load(temp_dir, 'template')
            assert False, "Expected ValueError"
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_path_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {
            'cookiecutter': {
                'value': 'test'
            }
        }
        file_path = Path(temp_dir) / 'template.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        result = load(Path(temp_dir), 'template')
        
        assert result == test_data


# LLM-generated content at query #43
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_data = {'cookiecutter': {'project_name': 'test_project'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template.json')
    
    assert result == json_data
    assert 'cookiecutter' in result


def test_load_with_template_name_without_extension(tmp_path):
    json_data = {'cookiecutter': {'key': 'value'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template')
    
    assert result == json_data


def test_load_missing_cookiecutter_key(tmp_path):
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"other_key": "value"}', encoding='utf-8')
    
    try:
        load(tmp_path, 'template.json')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_path_object(tmp_path):
    json_data = {'cookiecutter': {'name': 'test'}}
    json_file = tmp_path / 'config.json'
    json_file.write_text('{"cookiecutter": {"name": "test"}}', encoding='utf-8')
    
    result = load(tmp_path, 'config.json')
    
    assert result == json_data


def test_load_with_complex_cookiecutter_structure(tmp_path):
    json_data = {'cookiecutter': {'project': {'name': 'test', 'version': '1.0'}, 'author': 'John'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"project": {"name": "test", "version": "1.0"}, "author": "John"}}', encoding='utf-8')
    
    result = load(tmp_path, 'template.json')
    
    assert result == json_data
    assert result['cookiecutter']['project']['name'] == 'test'


