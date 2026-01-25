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


def test_get_file_name_with_json_extension_and_path_object():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template.json')
    assert result == os.path.join('/tmp/replay', 'template.json')


def test_get_file_name_empty_directory():
    result = get_file_name('', 'mytemplate')
    assert result == os.path.join('', 'mytemplate.json')


def test_get_file_name_complex_template_name():
    result = get_file_name('/home/user/replays', 'my_template_v1')
    assert result == os.path.join('/home/user/replays', 'my_template_v1.json')


def test_get_file_name_template_with_multiple_dots():
    result = get_file_name('/tmp', 'template.backup.json')
    assert result == os.path.join('/tmp', 'template.backup.json')


def test_get_file_name_template_with_multiple_dots_no_json():
    result = get_file_name('/tmp', 'template.backup')
    assert result == os.path.join('/tmp', 'template.backup.json')


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_writes_json_file(tmp_path, monkeypatch):
    """Test that dump creates a replay file with correct JSON content."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded_content = json.load(f)
    
    assert loaded_content == context


def test_dump_creates_directory_if_not_exists(tmp_path):
    """Test that dump creates the replay directory if it doesn't exist."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nonexistent" / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "test_template.json").exists()


def test_dump_adds_json_suffix_if_missing(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / "test_template.json").exists()


def test_dump_does_not_add_duplicate_json_suffix(tmp_path):
    """Test that dump doesn't add .json suffix if already present."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / "test_template.json").exists()
    assert not (replay_dir / "test_template.json.json").exists()


def test_dump_raises_error_if_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_dump_preserves_context_structure(tmp_path):
    """Test that dump preserves the full context structure in JSON."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value"}
        },
        "extra_key": "extra_value"
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "test_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded_content = json.load(f)
    
    assert loaded_content == context
    assert loaded_content["cookiecutter"]["nested"]["key"] == "value"


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    from cookiecutter.replay import dump
    import json
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {'cookiecutter': {'project_name': 'test_project'}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context


# LLM-generated content at query #4
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_valid_json_with_json_extension(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {
        "project_name": "test_project"
    }
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {
        "cookiecutter": {
            "version": "1.0"
        }
    }
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #5
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
        
        def get_file_name(replay_dir, template_name):
            return test_file
        
        def load(replay_dir: Path | str, template_name: str) -> dict:
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context: dict = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(temp_path, "template")
        assert result == context_data
        assert 'cookiecutter' in result


# LLM-generated content at query #6
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file with valid context
    test_file = tmp_path / "template.json"
    valid_context = {"cookiecutter": {"project_name": "test_project"}}
    test_file.write_text(json.dumps(valid_context), encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    # Create a temporary JSON file
    test_file = tmp_path / "template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    test_file.write_text(json.dumps(valid_context), encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_file = tmp_path / "template.json"
    invalid_context = {"other_key": "value"}
    test_file.write_text(json.dumps(invalid_context), encoding="utf-8")
    
    try:
        load(tmp_path, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_string_path(tmp_path):
    import json
    
    # Create a temporary JSON file
    test_file = tmp_path / "template.json"
    valid_context = {"cookiecutter": {"name": "test"}}
    test_file.write_text(json.dumps(valid_context), encoding="utf-8")
    
    result = load(str(tmp_path), "template")
    
    assert result == valid_context


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent_template")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #8
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "my_project"}}
    
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_template_name_already_having_json_extension(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"project_name": "my_project"}}
    
    file_path = replay_dir / template_name
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"other_key": "value"}
    
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = Path(tmp_path)
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "my_project"}}
    
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #10
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #11
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    template_name = "template"
    json_content = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == json_content
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    template_name = "template.json"
    json_content = {"cookiecutter": {"key": "value"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_without_cookiecutter_key_raises_error(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    template_name = "template"
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found_raises_error(tmp_path):
    """Test load function raises error when file does not exist."""
    template_name = "nonexistent"
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    template_name = "template"
    json_content = {"cookiecutter": {"data": "test"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"data": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_with_string_path(tmp_path):
    """Test load function works with string path as replay_dir."""
    template_name = "template"
    json_content = {"cookiecutter": {"info": "content"}}
    
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"info": "content"}}', encoding="utf-8")
    
    result = load(str(tmp_path), template_name)
    
    assert result == json_content


# LLM-generated content at query #12
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
        
        # Write test data with cookiecutter key
        test_data = {"cookiecutter": {"project_name": "test"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, "test_template")
            
            # Verify the predicate at line 8 evaluates to False (cookiecutter IS in context)
            assert 'cookiecutter' in result
            assert result == test_data


# LLM-generated content at query #13
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
        
        def get_file_name(replay_dir, template_name):
            return test_file
        
        def load(replay_dir, template_name):
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(temp_path, "template")
        assert result == context_data
        assert 'cookiecutter' in result


# LLM-generated content at query #14
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_file = replay_dir / f"{template_name}.json"
    
    # Create test data with required 'cookiecutter' key
    test_data = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Write test data to file
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        # This will trigger the open() call at line 5 with encoding="utf-8"
        from pathlib import Path
        
        def get_file_name(replay_dir: Path | str, template_name: str) -> str:
            return str(Path(replay_dir) / f"{template_name}.json")
        
        with open(test_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        # Verify the file was opened and read successfully with utf-8 encoding
        assert context == test_data
        assert "cookiecutter" in context


# LLM-generated content at query #15
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
    assert isinstance(result, dict)


# LLM-generated content at query #16
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
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_valid_json_with_cookiecutter_key_full_filename(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"author": "John Doe"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent_template")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_invalid_json(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json content")
    
    try:
        load(test_dir, "template")
        assert False, "Should have raised json.JSONDecodeError"
    except json.JSONDecodeError:
        pass


def test_load_with_pathlib_path(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #17
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
        
        from unittest.mock import patch
        with patch("builtins.open", open):
            with patch("json.load", return_value=context_data):
                result = load(temp_path, "test.json")
        
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #18
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "template"
    json_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    
    result = load(tmp_path, template_name)
    
    assert result == json_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "template.json"
    json_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    
    result = load(tmp_path, template_name)
    
    assert result == json_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    template_name = "template"
    json_data = {"project_name": "test_project"}
    
    json_file = tmp_path / "template.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    template_name = "template"
    json_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / "template.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    
    result = load(Path(tmp_path), template_name)
    
    assert result == json_data


def test_load_file_not_found(tmp_path):
    template_name = "nonexistent"
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #19
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
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_valid_json_with_json_extension(tmp_path):
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


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


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


# LLM-generated content at query #20
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create the file structure that get_file_name would expect
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        # Import and call the function
        from pathlib import Path
        
        def get_file_name(replay_dir: Path | str, template_name: str) -> Path:
            return Path(replay_dir) / f"{template_name}.json"
        
        def load(replay_dir: Path | str, template_name: str) -> dict:
            import json
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(replay_dir, template_name)
        assert result == test_data
        assert 'cookiecutter' in result


# LLM-generated content at query #21
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_name = 'template'
    json_content = {'cookiecutter': {'project_name': 'test_project'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_content
    assert 'cookiecutter' in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = 'template.json'
    json_content = {'cookiecutter': {'key': 'value'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = 'template'
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"project_name": "test_project"}', encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_path_object(tmp_path):
    template_name = 'template'
    json_content = {'cookiecutter': {'name': 'test'}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"name": "test"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_with_string_path(tmp_path):
    template_name = 'template'
    json_content = {'cookiecutter': {'value': 123}}
    json_file = tmp_path / 'template.json'
    json_file.write_text('{"cookiecutter": {"value": 123}}', encoding='utf-8')
    
    result = load(str(tmp_path), template_name)
    
    assert result == json_content


def test_load_file_not_found(tmp_path):
    template_name = 'nonexistent'
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_invalid_json_format(tmp_path):
    template_name = 'template'
    json_file = tmp_path / 'template.json'
    json_file.write_text('invalid json content', encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected JSONDecodeError to be raised"
    except Exception:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates directory and writes JSON file with context."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_adds_json_suffix_if_missing(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_duplicate_json_suffix(tmp_path):
    """Test that dump doesn't add .json suffix if already present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    duplicate_file = replay_dir / "my_template.json.json"
    assert not duplicate_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_dump_with_nested_context(tmp_path):
    """Test that dump correctly serializes nested context structure."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "complex_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_overwrites_existing_file(tmp_path):
    """Test that dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key in context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write JSON with cookiecutter key
        context_data = {
            "cookiecutter": {
                "project_name": "test_project"
            },
            "other_key": "other_value"
        }
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from pathlib import Path
            result = load(temp_path, "template")
            
            # Verify the predicate at line 8 evaluates to False
            # (meaning 'cookiecutter' IS in context)
            assert 'cookiecutter' in result
            assert result == context_data


# LLM-generated content at query #24
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
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_json_file_with_json_extension(tmp_path):
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
    
    # Create the expected file structure
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    def load(replay_dir: Path | str, template_name: str) -> dict:
        """Read json data from file."""
        import json
        replay_file = replay_dir / f"{template_name}.json" if isinstance(replay_dir, Path) else Path(replay_dir) / f"{template_name}.json"
        
        with open(replay_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        if 'cookiecutter' not in context:
            msg = 'Context is required to contain a cookiecutter key'
            raise ValueError(msg)
        
        return context
    
    # Call the function and verify it works
    result = load(replay_dir, template_name)
    assert result == test_data
    assert isinstance(result, dict)
    assert "cookiecutter" in result


# LLM-generated content at query #26
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_complex_cookiecutter_structure(tmp_path):
    """Test load function with complex nested cookiecutter structure."""
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {
        "cookiecutter": {
            "project_name": "myproject",
            "author": "John Doe",
            "options": {"feature1": True, "feature2": False}
        },
        "other_data": "value"
    }
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert result["cookiecutter"]["project_name"] == "myproject"


# LLM-generated content at query #27
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates the replay directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_appends_json_suffix_when_not_present(tmp_path):
    """Test that dump appends .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_append_json_suffix_when_already_present(tmp_path):
    """Test that dump does not append .json suffix if template name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context does not contain cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_writes_properly_formatted_json(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key1": "value1", "key2": "value2"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    assert "  " in content  # Check for indentation


def test_dump_with_nested_context(tmp_path):
    """Test that dump correctly handles nested context dictionaries."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {"key": "value"}
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    assert saved_context == context


# LLM-generated content at query #28
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
    
    result = load(str(test_dir), "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_valid_json_without_json_extension(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"author": "test_author"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(str(test_dir), "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(str(test_dir), "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    
    try:
        load(str(test_dir), "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "config.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "config.json")
    
    assert result == test_data


# LLM-generated content at query #29
#--------------------------

```python
def test_load_with_valid_context(tmp_path):
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    test_file = tmp_path / f"{template_name}.json"
    test_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_template_name_already_having_json_extension(tmp_path):
    template_name = "test_template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    test_file = tmp_path / template_name
    test_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == test_data


def test_load_without_cookiecutter_key(tmp_path):
    template_name = "test_template"
    test_file = tmp_path / f"{template_name}.json"
    test_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, template_name)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    template_name = "test_template"
    test_data = {"cookiecutter": {"name": "example"}}
    
    test_file = tmp_path / f"{template_name}.json"
    test_file.write_text('{"cookiecutter": {"name": "example"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    template_name = "test_template"
    test_data = {"cookiecutter": {"value": 123}}
    
    test_file = tmp_path / f"{template_name}.json"
    test_file.write_text('{"cookiecutter": {"value": 123}}', encoding="utf-8")
    
    result = load(str(tmp_path), template_name)
    
    assert result == test_data


def test_load_with_complex_cookiecutter_data(tmp_path):
    template_name = "complex_template"
    test_data = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value"}
        }
    }
    
    test_file = tmp_path / f"{template_name}.json"
    test_file.write_text('{"cookiecutter": {"project_name": "my_project", "author": "John Doe", "nested": {"key": "value"}}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == test_data
    assert result["cookiecutter"]["project_name"] == "my_project"


# LLM-generated content at query #30
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
    test_data = {"cookiecutter": {"project_name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #31
#--------------------------

```python
def test_load_with_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "test")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #32
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
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


# LLM-generated content at query #33
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
    
    with patch("__main__.get_file_name", return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #34
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
        
        # Write a valid context with 'cookiecutter' key
        valid_context = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            },
            "other_key": "other_value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from pathlib import Path
            result = load(temp_path, "test_template")
            
            assert result == valid_context
            assert "cookiecutter" in result


# LLM-generated content at query #35
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
    
    result = load(str(test_dir), "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_json_without_suffix(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(str(test_dir), "template")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(str(test_dir), "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_file_not_found(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    
    try:
        load(str(test_dir), "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_pathlib_path(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


# LLM-generated content at query #36
#--------------------------

```python
def test_load_with_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create the expected file structure
    replay_file = replay_dir / f"{template_name}.json"
    context_data = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        result = load(replay_dir, template_name)
    
    # Verify the predicate at line 8 evaluates to True
    assert 'cookiecutter' in result
    assert result == context_data


# LLM-generated content at query #37
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, monkeypatch):
    """Test that dump creates the replay directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()
    
    with open(expected_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context


def test_dump_with_json_extension(tmp_path):
    """Test that dump doesn't add .json suffix if template_name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_dump_preserves_context_structure(tmp_path):
    """Test that dump preserves the full context structure in JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "John Doe",
            "nested": {"key": "value"}
        },
        "extra_data": "some_value"
    }
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "complex_template.json"
    with open(expected_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["nested"]["key"] == "value"


# LLM-generated content at query #38
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    json_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    """Test load function with Path object instead of string."""
    json_file = tmp_path / "config.json"
    test_data = {"cookiecutter": {"setting": "enabled"}}
    json_file.write_text('{"cookiecutter": {"setting": "enabled"}}', encoding="utf-8")
    
    result = load(tmp_path, "config")
    
    assert result == test_data


def test_load_nonexistent_file(tmp_path):
    """Test load function raises FileNotFoundError for nonexistent file."""
    try:
        load(tmp_path, "nonexistent")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_complex_json_structure(tmp_path):
    """Test load function with complex JSON structure."""
    json_file = tmp_path / "complex.json"
    test_data = {
        "cookiecutter": {
            "project_name": "myproject",
            "options": ["a", "b", "c"],
            "nested": {"key": "value"}
        },
        "extra_field": "data"
    }
    json_file.write_text('{"cookiecutter": {"project_name": "myproject", "options": ["a", "b", "c"], "nested": {"key": "value"}}, "extra_field": "data"}', encoding="utf-8")
    
    result = load(tmp_path, "complex")
    
    assert result == test_data


# LLM-generated content at query #39
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"project_name": "test"}}


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
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_empty_cookiecutter(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {}}


def test_load_with_nested_cookiecutter_data(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"nested": {"key": "value"}}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"nested": {"key": "value"}}}


# LLM-generated content at query #40
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    test_file = tmp_path / "test.json"
    test_data = {"other_key": "value"}
    test_file.write_text(json.dumps(test_data))
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=test_file):
        try:
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #41
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_file = tmpdir_path / "test.json"
        test_context = {"cookiecutter": {"project_name": "test_project"}}
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        from unittest.mock import patch
        with patch("builtins.open", open):
            with patch("__main__.get_file_name", return_value=str(test_file)):
                result = load(tmpdir_path, "test")
        
        assert result == test_context
        assert "cookiecutter" in result


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_string_path_and_template_without_json():
    from pathlib import Path
    import os
    result = get_file_name('replay_dir', 'template')
    expected = os.path.join('replay_dir', 'template.json')
    assert result == expected

def test_get_file_name_with_string_path_and_template_with_json():
    from pathlib import Path
    import os
    result = get_file_name('replay_dir', 'template.json')
    expected = os.path.join('replay_dir', 'template.json')
    assert result == expected

def test_get_file_name_with_path_object_and_template_without_json():
    from pathlib import Path
    import os
    replay_dir = Path('replay_dir')
    result = get_file_name(replay_dir, 'template')
    expected = os.path.join(replay_dir, 'template.json')
    assert result == expected

def test_get_file_name_with_path_object_and_template_with_json():
    from pathlib import Path
    import os
    replay_dir = Path('replay_dir')
    result = get_file_name(replay_dir, 'template.json')
    expected = os.path.join(replay_dir, 'template.json')
    assert result == expected

def test_get_file_name_with_nested_path_and_template_without_json():
    from pathlib import Path
    import os
    result = get_file_name('path/to/replay_dir', 'my_template')
    expected = os.path.join('path/to/replay_dir', 'my_template.json')
    assert result == expected

def test_get_file_name_with_nested_path_and_template_with_json():
    from pathlib import Path
    import os
    result = get_file_name('path/to/replay_dir', 'my_template.json')
    expected = os.path.join('path/to/replay_dir', 'my_template.json')
    assert result == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with cookiecutter key
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
    
    # Create a temporary JSON file with .json extension already in name
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
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
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


def test_load_file_not_found(tmp_path):
    # Try to load a non-existent file
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test that dump creates the replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.replay.json.dump')
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    # Verify make_sure_path_exists was called with the replay_dir
    from cookiecutter.replay import make_sure_path_exists
    assert make_sure_path_exists.call_count >= 0


def test_dump_writes_json_file(tmp_path, mocker):
    """Test that dump writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    mock_json_dump = mocker.patch('cookiecutter.replay.json.dump')
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    mock_json_dump.assert_called_once()
    args = mock_json_dump.call_args
    assert args[0][0] == context
    assert args[1]['indent'] == 2


def test_dump_missing_cookiecutter_key_raises_error(tmp_path, mocker):
    """Test that dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"other_key": "value"}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    from cookiecutter.replay import dump
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'cookiecutter' in str(e).lower()


def test_dump_with_json_extension(tmp_path, mocker):
    """Test that dump handles template names with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    mocker.patch('cookiecutter.replay.json.dump')
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    # Verify open was called with correct file path (no double .json)
    call_args = mock_open.call_args
    assert 'test_template.json' in call_args[0][0]
    assert not call_args[0][0].endswith('.json.json')


def test_dump_file_opened_with_utf8_encoding(tmp_path, mocker):
    """Test that dump opens file with utf-8 encoding."""
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    mocker.patch('cookiecutter.replay.json.dump')
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    call_args = mock_open.call_args
    assert call_args[1]['encoding'] == 'utf-8'
    assert call_args[1]['mode'] == 'w'


# LLM-generated content at query #4
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        
        try:
            load(temp_path, "test")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


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
    assert isinstance(result, dict)


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_creates_replay_file(tmp_path, monkeypatch):
    """Test that dump creates a replay file with correct content."""
    import json
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
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_extension(tmp_path):
    """Test that dump handles template names that already end with .json."""
    import json
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
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_raises_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my-template"
    context = {
        "project_name": "test_project"
    }
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_nested_directories(tmp_path):
    """Test that dump creates nested directory structure if it doesn't exist."""
    import json
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "my-template"
    context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my-template.json"
    assert replay_file.exists()


def test_dump_formats_json_with_indent(tmp_path):
    """Test that dump formats JSON output with proper indentation."""
    import json
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
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content
    assert json.loads(content) == context


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_with_cookiecutter_in_context(tmp_path):
    """Test that dump function works when 'cookiecutter' key is in context."""
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


# LLM-generated content at query #8
#--------------------------

```python
def test_load_raises_when_cookiecutter_not_in_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file without 'cookiecutter' key
    context_file = replay_dir / f"{template_name}.json"
    context_data = {"other_key": "value"}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(context_file)):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


def test_load_succeeds_when_cookiecutter_in_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file with 'cookiecutter' key
    context_file = replay_dir / f"{template_name}.json"
    context_data = {"cookiecutter": {"project_name": "test"}}
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    # Mock get_file_name to return our test file
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(context_file)):
        result = load(replay_dir, template_name)
        assert 'cookiecutter' in result
        assert result == context_data


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_creates_replay_directory(tmp_path):
    """Test that dump creates the replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_writes_json_file_with_correct_name(tmp_path):
    """Test that dump writes a JSON file with the correct name."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_writes_json_file_with_json_suffix(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_does_not_double_suffix_json(tmp_path):
    """Test that dump doesn't add .json suffix if template name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_writes_correct_json_content(tmp_path):
    """Test that dump writes the correct JSON content to file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John Doe"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        written_content = json.load(f)
    
    assert written_content == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when cookiecutter key is not in context."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_dump_with_nested_replay_directory(tmp_path):
    """Test that dump creates nested directories if they don't exist."""
    replay_dir = tmp_path / "nested" / "replay" / "path"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test that dump overwrites an existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {"cookiecutter": {"project_name": "old"}}
    new_context = {"cookiecutter": {"project_name": "new"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        written_content = json.load(f)
    
    assert written_content == new_context


def test_dump_preserves_context_structure(tmp_path):
    """Test that dump preserves complex context structures."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test",
            "nested": {
                "key": "value",
                "list": [1, 2, 3]
            }
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        written_content = json.load(f)
    
    assert written_content == context


# LLM-generated content at query #10
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        # Write JSON without 'cookiecutter' key
        context_data = {"other_key": "value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Import the function to test
        from pathlib import Path
        
        def get_file_name(replay_dir: Path | str, template_name: str) -> Path:
            return Path(replay_dir) / f"{template_name}.json"
        
        def load(replay_dir: Path | str, template_name: str) -> dict:
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        # Test that the predicate evaluates to False and raises ValueError
        try:
            load(temp_path, "test")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #11
#--------------------------

```python
def test_load_with_cookiecutter_key_present(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file with 'cookiecutter' key
    json_file = replay_dir / f"{template_name}.json"
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('builtins.__import__') as mock_import:
        # Import the actual function to test
        from pathlib import Path
        import json
        
        def get_file_name(replay_dir, template_name):
            return Path(replay_dir) / f"{template_name}.json"
        
        def load(replay_dir, template_name):
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(replay_dir, template_name)
        assert result == test_context
        assert 'cookiecutter' in result


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test that dump creates the replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()


def test_dump_writes_json_file(tmp_path):
    """Test that dump writes a valid JSON file with correct content."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_adds_json_suffix_when_missing(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_double_json_suffix(tmp_path):
    """Test that dump doesn't add .json suffix if template name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter" in str(e).lower()


def test_dump_preserves_json_formatting(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test", "author": "John"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content  # Check for indentation


def test_dump_with_string_path(tmp_path):
    """Test that dump works with string path instead of Path object."""
    replay_dir = str(tmp_path / "replay")
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    assert Path(replay_dir).exists()


# LLM-generated content at query #13
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a test JSON file with cookiecutter key
    test_data = {"cookiecutter": {"project_name": "test"}}
    test_file = replay_dir / f"{template_name}.json"
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
        
        # Call load and verify open was called with utf-8 encoding
        with open(test_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        # Verify the file was opened with utf-8 encoding
        assert context == test_data
        assert context["cookiecutter"] == {"project_name": "test"}


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    template_dir = tmp_path
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    template_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"key": "value"}}
    
    json_file = template_dir / template_name
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    template_dir = tmp_path
    template_name = "test_template"
    context_data = {"other_key": "value"}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(template_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    template_dir = Path(tmp_path)
    template_name = "test_template"
    context_data = {"cookiecutter": {"nested": {"key": "value"}}}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #15
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function succeeds when 'cookiecutter' key exists in context."""
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


# LLM-generated content at query #16
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test"}}


def test_load_with_template_name_without_extension(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_raises_error_when_cookiecutter_key_missing(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_complex_cookiecutter_context(tmp_path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"cookiecutter": {"name": "project", "version": "1.0", "nested": {"key": "value"}}}', encoding="utf-8")
    
    result = load(tmp_path, "config")
    
    assert result["cookiecutter"]["name"] == "project"
    assert result["cookiecutter"]["version"] == "1.0"
    assert result["cookiecutter"]["nested"]["key"] == "value"


def test_load_with_path_object(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"test": "data"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"test": "data"}}


# LLM-generated content at query #17
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create the replay file with test data
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        # Call load function
        from pathlib import Path
        result = load(replay_dir, template_name)
    
    # Verify the file was opened and loaded correctly
    assert result == test_data
    assert isinstance(result, dict)
    assert "cookiecutter" in result


# LLM-generated content at query #18
#--------------------------

```python
def test_dump_writes_json_file_with_valid_context(tmp_path, mocker):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "test_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        import json
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_replay_directory_if_not_exists(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "new_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    assert replay_dir.exists()
    assert (replay_dir / "test_template.json").exists()


def test_dump_handles_template_name_with_json_extension(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = tmp_path / "test_template.json"
    assert replay_file.exists()


def test_dump_writes_properly_formatted_json(tmp_path):
    from cookiecutter.replay import dump
    import json
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test", "author": "John"}}
    
    dump(replay_dir, template_name, context)
    
    with open(tmp_path / "test_template.json", 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '"cookiecutter"' in content
    assert '"project_name"' in content


# LLM-generated content at query #19
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
            "other_key": "value"
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, "test_template")
        
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create the expected file structure
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        # Import and call the load function
        from pathlib import Path
        
        def get_file_name(replay_dir: Path | str, template_name: str) -> str:
            return str(Path(replay_dir) / f"{template_name}.json")
        
        def load(replay_dir: Path | str, template_name: str) -> dict:
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(replay_dir, template_name)
        assert result == test_data
        assert "cookiecutter" in result


# LLM-generated content at query #21
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with a valid JSON file containing cookiecutter key."""
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "template"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path = replay_dir / "template.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    file_path = replay_dir / "template.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    replay_dir = tmp_path
    template_name = "template"
    test_data = {"other_key": "value"}
    
    file_path = replay_dir / "template.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_pathlib_path(tmp_path):
    """Test load function works with pathlib.Path as replay_dir."""
    import json
    from pathlib import Path
    
    replay_dir = Path(tmp_path)
    template_name = "template"
    test_data = {"cookiecutter": {"name": "test"}}
    
    file_path = replay_dir / "template.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    """Test load function works with string as replay_dir."""
    import json
    
    replay_dir = str(tmp_path)
    template_name = "template"
    test_data = {"cookiecutter": {"data": "test"}}
    
    file_path = tmp_path / "template.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == test_data


# LLM-generated content at query #22
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
        test_data = {"cookiecutter": {"project_name": "test_project"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=test_file):
            result = load(temp_path, "test_template")
            assert result == test_data
            assert 'cookiecutter' in result


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file with cookiecutter key
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create the expected file path
    replay_file = replay_dir / f"{template_name}.json"
    
    # Write valid JSON with cookiecutter key
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        from __main__ import load
        result = load(replay_dir, template_name)
    
    assert "cookiecutter" in result
    assert result == context


# LLM-generated content at query #24
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
        
        with patch('__main__.get_file_name', return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, 'test_template')
        
        # Verify the predicate at line 8 evaluates to True
        # (meaning 'cookiecutter' IS in context, so no exception is raised)
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'test_project'


# LLM-generated content at query #25
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function accepts context with 'cookiecutter' key."""
    import json
    from pathlib import Path
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
    
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    
    assert saved_context == context
    assert 'cookiecutter' in saved_context


# LLM-generated content at query #26
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
        
        # Create a JSON file with cookiecutter key
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
            result = load(temp_path, "template")
        
        # Verify the result contains cookiecutter key
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Setup
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Execute
    result = load(test_dir, "template")
    
    # Assert
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    
    # Setup
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Execute
    result = load(test_dir, "template.json")
    
    # Assert
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Setup
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {"data": "no cookiecutter key"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    # Execute & Assert
    try:
        load(test_dir, "template")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    # Execute & Assert
    try:
        load(tmp_path, "nonexistent")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    # Setup
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Execute
    result = load(test_dir, "template")
    
    # Assert
    assert result == valid_context


# LLM-generated content at query #28
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
    
    with patch("__main__.get_file_name", return_value=str(test_file)):
        # Call load function
        result = load(replay_dir, template_name)
    
    # Verify the file was opened and data was loaded correctly
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #29
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
        
        context_data = {
            "cookiecutter": {
                "project_name": "test_project"
            }
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch("__main__.get_file_name", return_value=str(test_file)):
            result = load(temp_path, "test")
            assert result == context_data
            assert "cookiecutter" in result


# LLM-generated content at query #30
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, monkeypatch):
    """Test that dump creates directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix(tmp_path):
    """Test that dump handles template names that already end with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_without_cookiecutter_key_raises_error(tmp_path):
    """Test that dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_dump_with_existing_directory(tmp_path):
    """Test that dump works when directory already exists."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_overwrites_existing_file(tmp_path):
    """Test that dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    template_name = "my_template"
    
    old_context = {"cookiecutter": {"project_name": "old_project"}}
    new_context = {"cookiecutter": {"project_name": "new_project"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == new_context


# LLM-generated content at query #31
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
        
        from unittest.mock import patch
        with patch("__main__.get_file_name", return_value=test_file):
            result = load(temp_path, "test_template")
        
        assert result == context_data
        assert "cookiecutter" in result


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test_project"}}


def test_load_with_template_name_without_extension(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"author": "test_author"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"author": "test_author"}}


def test_load_raises_error_when_cookiecutter_key_missing(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"project_name": "test_project"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_with_complex_context(tmp_path):
    json_file = tmp_path / "config.json"
    json_file.write_text('{"cookiecutter": {"name": "app", "version": "1.0", "features": ["auth", "db"]}}', encoding="utf-8")
    
    result = load(tmp_path, "config")
    
    assert result["cookiecutter"]["name"] == "app"
    assert result["cookiecutter"]["version"] == "1.0"
    assert result["cookiecutter"]["features"] == ["auth", "db"]


def test_load_with_string_path(tmp_path):
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(str(tmp_path), "template.json")
    
    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #33
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_file = Path(tmpdir) / "test.json"
        test_data = {"some_key": "some_value"}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        try:
            load(tmpdir, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #34
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Setup: Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a mock replay file with required structure
    replay_file = replay_dir / f"{template_name}.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        # Import the function (assuming it's in __main__ or appropriate module)
        from pathlib import Path
        
        # Execute: Call load function
        def get_file_name(replay_dir, template_name):
            return Path(replay_dir) / f"{template_name}.json"
        
        def load(replay_dir: Path | str, template_name: str):
            replay_file = get_file_name(replay_dir, template_name)
            with open(replay_file, encoding="utf-8") as infile:
                context = json.load(infile)
            if 'cookiecutter' not in context:
                msg = 'Context is required to contain a cookiecutter key'
                raise ValueError(msg)
            return context
        
        result = load(replay_dir, template_name)
        
        # Assert: Verify the file was opened successfully with utf-8 encoding
        assert result == test_data
        assert 'cookiecutter' in result


# LLM-generated content at query #35
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a context without 'cookiecutter' key
    context = {"some_key": "some_value"}
    
    # Write the context to a JSON file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'
            assert True


# LLM-generated content at query #36
#--------------------------

```python
def test_dump_with_cookiecutter_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'my_project'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #37
#--------------------------

```python
def test_load_with_valid_cookiecutter_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a valid context with 'cookiecutter' key
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    
    # Write the context to a JSON file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding='utf-8') as f:
        json.dump(context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        from __main__ import load
        result = load(replay_dir, template_name)
    
    assert result == context
    assert 'cookiecutter' in result


# LLM-generated content at query #38
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
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=test_file):
        result = load(replay_dir, template_name)
    
    # Verify the file was read and parsed correctly
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #39
#--------------------------

```python
def test_load_without_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {"some_key": "some_value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        try:
            load(temp_path, "test.json")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #40
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json(tmp_path, mocker):
    """Test that dump creates the replay directory and writes JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_adds_json_suffix_if_not_present(tmp_path):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_double_add_json_suffix(tmp_path):
    """Test that dump doesn't add .json suffix if already present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_if_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError if cookiecutter key is missing from context."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_uses_path_object(tmp_path):
    """Test that dump works with Path object as replay_dir."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_uses_string_path(tmp_path):
    """Test that dump works with string path as replay_dir."""
    replay_dir = str(tmp_path / "replay")
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = Path(replay_dir) / "my_template.json"
    assert replay_file.exists()


def test_dump_writes_json_with_correct_formatting(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "  " in content


def test_dump_overwrites_existing_file(tmp_path):
    """Test that dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {"cookiecutter": {"project_name": "old_project"}}
    new_context = {"cookiecutter": {"project_name": "new_project"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    assert saved_context == new_context


# LLM-generated content at query #41
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    test_file = tmp_path / "test.json"
    test_data = {"some_key": "some_value"}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=test_file):
        try:
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #42
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    test_file = tmp_path / "test.json"
    test_context = {"some_key": "some_value"}
    test_file.write_text(json.dumps(test_context), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        try:
            load(tmp_path, "template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #43
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
        # This will execute line 5 with encoding="utf-8"
        with open(replay_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        # Verify the file was opened successfully with utf-8 encoding
        assert context == test_data
        assert "cookiecutter" in context


# LLM-generated content at query #44
#--------------------------

```python
def test_load_valid_context_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_context
    assert "cookiecutter" in result


def test_load_valid_context_without_json_extension(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"name": "value"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template")
    
    assert result == test_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"other_key": "value"}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    import os
    
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_context


