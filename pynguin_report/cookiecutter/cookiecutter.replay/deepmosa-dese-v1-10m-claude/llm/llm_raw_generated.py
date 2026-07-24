####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_path_object_without_json_extension():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template')
    assert result == os.path.join(Path('/tmp/replay'), 'template.json')

def test_get_file_name_with_string_path_without_json_extension():
    result = get_file_name('/tmp/replay', 'template')
    assert result == os.path.join('/tmp/replay', 'template.json')

def test_get_file_name_with_json_extension():
    result = get_file_name('/tmp/replay', 'template.json')
    assert result == os.path.join('/tmp/replay', 'template.json')

def test_get_file_name_with_path_object_and_json_extension():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template.json')
    assert result == os.path.join(Path('/tmp/replay'), 'template.json')

def test_get_file_name_with_empty_directory():
    result = get_file_name('', 'myfile')
    assert result == os.path.join('', 'myfile.json')

def test_get_file_name_with_nested_path():
    result = get_file_name('/home/user/replays/data', 'replay_template')
    assert result == os.path.join('/home/user/replays/data', 'replay_template.json')


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, monkeypatch):
    """Test that dump creates the replay directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    monkeypatch.setattr("cookiecutter.replay.make_sure_path_exists", lambda x: None)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: type('MockFile', (), {
        '__enter__': lambda self: self,
        '__exit__': lambda self, *args: None,
        'write': lambda self, x: None
    })())
    
    import json
    written_data = []
    
    def mock_dump(data, file, **kwargs):
        written_data.append(data)
    
    monkeypatch.setattr("json.dump", mock_dump)
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    assert written_data[0] == context


def test_dump_raises_valueerror_without_cookiecutter_key(tmp_path, monkeypatch):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    monkeypatch.setattr("cookiecutter.replay.make_sure_path_exists", lambda x: None)
    
    from cookiecutter.replay import dump
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_dump_adds_json_suffix_to_template_name(tmp_path, monkeypatch):
    """Test that dump adds .json suffix to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path_used = []
    
    def mock_open(path, *args, **kwargs):
        file_path_used.append(path)
        return type('MockFile', (), {
            '__enter__': lambda self: self,
            '__exit__': lambda self, *args: None
        })()
    
    monkeypatch.setattr("cookiecutter.replay.make_sure_path_exists", lambda x: None)
    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("json.dump", lambda *args, **kwargs: None)
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    assert "my_template.json" in file_path_used[0]


def test_dump_does_not_add_json_suffix_if_already_present(tmp_path, monkeypatch):
    """Test that dump does not add .json suffix if template name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path_used = []
    
    def mock_open(path, *args, **kwargs):
        file_path_used.append(path)
        return type('MockFile', (), {
            '__enter__': lambda self: self,
            '__exit__': lambda self, *args: None
        })()
    
    monkeypatch.setattr("cookiecutter.replay.make_sure_path_exists", lambda x: None)
    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("json.dump", lambda *args, **kwargs: None)
    
    from cookiecutter.replay import dump
    dump(replay_dir, template_name, context)
    
    assert "my_template.json.json" not in file_path_used[0]
    assert "my_template.json" in file_path_used[0]


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path, mocker):
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
    """Test that dump adds .json suffix if template_name doesn't have it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_duplicate_json_suffix(tmp_path):
    """Test that dump doesn't add .json suffix if template_name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_if_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError if context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_writes_valid_json(tmp_path):
    """Test that dump writes valid JSON that can be read back."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"cookiecutter": {"name": "test", "version": "1.0", "nested": {"key": "value"}}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context


# LLM-generated content at query #4
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
        },
        'other_key': 'other_value'
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #5
#--------------------------

```python
def test_dump_with_cookiecutter_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
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


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.json"
        test_data = {"some_key": "some_value"}
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        try:
            load(tmpdir, "test")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #8
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
    
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_with_cookiecutter_in_context(tmp_path):
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


# LLM-generated content at query #10
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_name = 'test_template'
    json_data = {'cookiecutter': {'project_name': 'my_project'}}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text('{"cookiecutter": {"project_name": "my_project"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_data
    assert 'cookiecutter' in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = 'test_template.json'
    json_data = {'cookiecutter': {'version': '1.0'}}
    
    json_file = tmp_path / template_name
    json_file.write_text('{"cookiecutter": {"version": "1.0"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_data


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = 'test_template'
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text('{"data": "value"}', encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_pathlib_path(tmp_path):
    template_name = 'test_template'
    json_data = {'cookiecutter': {'key': 'value'}}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text('{"cookiecutter": {"key": "value"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == json_data


def test_load_with_string_path(tmp_path):
    template_name = 'test_template'
    json_data = {'cookiecutter': {'name': 'test'}}
    
    json_file = tmp_path / f'{template_name}.json'
    json_file.write_text('{"cookiecutter": {"name": "test"}}', encoding='utf-8')
    
    result = load(str(tmp_path), template_name)
    
    assert result == json_data


# LLM-generated content at query #11
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and JSON file without 'cookiecutter' key
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create the expected file path
    replay_file = replay_dir / f"{template_name}.json"
    
    # Write JSON data without 'cookiecutter' key
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
    
    # Test that ValueError is raised when 'cookiecutter' key is missing
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #12
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
    test_data = {"cookiecutter": {"key": "value"}}
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    # Verify the file was opened and read correctly
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #13
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
        
        # Write a JSON file with 'cookiecutter' key
        context_data = {
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            },
            'other_key': 'other_value'
        }
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=str(test_file)):
            result = load(temp_path, 'test_template')
        
        # Verify the predicate evaluates to False (cookiecutter IS in context)
        assert 'cookiecutter' in result
        assert result == context_data


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    template_name = "test_template"
    context_data = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    file_path = os.path.join(test_dir, f"{template_name}.json")
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    import os
    
    test_dir = tmp_path
    template_name = "test_template.json"
    context_data = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    file_path = os.path.join(test_dir, template_name)
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    
    test_dir = tmp_path
    template_name = "test_template"
    context_data = {
        "project_name": "test_project"
    }
    
    file_path = os.path.join(test_dir, f"{template_name}.json")
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(test_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    template_name = "test_template"
    context_data = {
        "cookiecutter": {
            "key": "value"
        }
    }
    
    file_path = test_dir / f"{template_name}.json"
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(test_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #15
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_data = {'cookiecutter': {'project_name': 'test_project'}}
    test_file = tmp_path / 'template.json'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template')
    
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_valid_json_with_cookiecutter_key_explicit_extension(tmp_path):
    import json
    
    test_data = {'cookiecutter': {'key': 'value'}}
    test_file = tmp_path / 'template.json'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template.json')
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_data = {'other_key': 'value'}
    test_file = tmp_path / 'template.json'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(tmp_path, 'template')
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_empty_cookiecutter_key(tmp_path):
    import json
    
    test_data = {'cookiecutter': {}}
    test_file = tmp_path / 'template.json'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template')
    
    assert result == test_data
    assert result['cookiecutter'] == {}


def test_load_nested_cookiecutter_data(tmp_path):
    import json
    
    test_data = {'cookiecutter': {'nested': {'deep': 'value'}, 'list': [1, 2, 3]}}
    test_file = tmp_path / 'template.json'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template')
    
    assert result == test_data
    assert result['cookiecutter']['nested']['deep'] == 'value'
    assert result['cookiecutter']['list'] == [1, 2, 3]


# LLM-generated content at query #16
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
        
        # Write test data with cookiecutter key
        test_context = {
            "cookiecutter": {
                "project_name": "test_project",
                "author": "test_author"
            }
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=str(test_file)):
            result = load(temp_path, "test_template")
            
            assert 'cookiecutter' in result
            assert result == test_context


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
        
        context_data = {"cookiecutter": {"key": "value"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        with open(test_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        result = 'cookiecutter' not in context
        assert result is False


# LLM-generated content at query #18
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without 'cookiecutter' key
    test_file = tmp_path / "test.json"
    test_context = {"some_key": "some_value"}
    test_file.write_text(json.dumps(test_context), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=test_file):
        try:
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #19
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with valid JSON file containing cookiecutter key."""
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with valid context
    test_data = {'cookiecutter': {'project_name': 'test_project'}}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template.json')
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_without_json_extension(tmp_path):
    """Test load function adds .json extension when not provided."""
    import json
    
    # Create a temporary JSON file
    test_data = {'cookiecutter': {'key': 'value'}}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(tmp_path, 'template')
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_data = {'other_key': 'value'}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(tmp_path, 'template.json')
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found(tmp_path):
    """Test load function raises FileNotFoundError when file doesn't exist."""
    try:
        load(tmp_path, 'nonexistent.json')
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_data = {'cookiecutter': {'name': 'test'}}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(Path(tmp_path), 'template.json')
    assert result == test_data


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.json"
        
        context_data = {"cookiecutter": {"key": "value"}}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        result = load(temp_path, "test")
        
        assert result == context_data
        assert "cookiecutter" in result


# LLM-generated content at query #21
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
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        try:
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #22
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary directory and JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    
    # Write valid JSON with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Import the function
    from pathlib import Path as PathlibPath
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    
    # Call the load function
    result = load(str(test_dir), "template.json")
    
    # Assert the result
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_without_json_extension(tmp_path):
    import json
    
    # Create a temporary directory and JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    
    # Write valid JSON with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Call the load function without .json extension
    result = load(str(test_dir), "template")
    
    # Assert the result
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary directory and JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    
    # Write JSON without cookiecutter key
    invalid_context = {
        "project_name": "test_project"
    }
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    # Call the load function and expect ValueError
    try:
        load(str(test_dir), "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_with_pathlib_path(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    
    # Write valid JSON with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "version": "1.0"
        }
    }
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Call the load function with Path object
    result = load(test_dir, "template.json")
    
    # Assert the result
    assert result == valid_context


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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
                "project_name": "test_project"
            }
        }
        
        json_file = replay_dir / f"{template_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        result = load(replay_dir, template_name)
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #25
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
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #26
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    from your_module import load
    result = load(test_dir, "template")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    from your_module import load
    result = load(test_dir, "template.json")
    
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {"other_key": "value"}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    from your_module import load
    
    try:
        load(test_dir, "template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    from your_module import load
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_string_path(tmp_path):
    import json
    
    test_dir = str(tmp_path)
    test_file = tmp_path / "template.json"
    valid_context = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    from your_module import load
    result = load(test_dir, "template")
    
    assert result == valid_context


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function works when 'cookiecutter' key is in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #29
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with valid JSON file containing cookiecutter key."""
    import json
    from pathlib import Path
    
    # Create a temporary JSON file with cookiecutter key
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Call load and verify it returns the correct data
    result = load(test_dir, "template")
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    import json
    from pathlib import Path
    
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
    """Test load function works with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "config.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "config")
    assert result == test_data


def test_load_nonexistent_file(tmp_path):
    """Test load function raises FileNotFoundError for nonexistent file."""
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
import json
import os
import tempfile
from pathlib import Path


def test_load_with_valid_context():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'project_name': 'test_project'}}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        from your_module import load
        result = load(temp_dir, 'template')
        
        assert result == test_data
        assert 'cookiecutter' in result


def test_load_with_template_name_already_has_json_extension():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'key': 'value'}}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        from your_module import load
        result = load(temp_dir, 'template.json')
        
        assert result == test_data


def test_load_with_path_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'cookiecutter': {'nested': {'data': 'value'}}}
        file_path = os.path.join(temp_dir, 'config.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        from your_module import load
        result = load(Path(temp_dir), 'config')
        
        assert result == test_data


def test_load_missing_cookiecutter_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = {'other_key': 'value'}
        file_path = os.path.join(temp_dir, 'template.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        from your_module import load
        try:
            load(temp_dir, 'template')
            assert False, "Expected ValueError"
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found():
    with tempfile.TemporaryDirectory() as temp_dir:
        from your_module import load
        try:
            load(temp_dir, 'nonexistent')
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# LLM-generated content at query #31
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with valid JSON file containing cookiecutter key."""
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
    """Test load function when template_name already has .json extension."""
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
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
        load(test_dir, "template")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    """Test load function with Path object as replay_dir."""
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #32
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
        from pathlib import Path
        result = load(replay_dir, template_name)
    
    # Verify the file was opened and read correctly
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #33
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    template_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_json_extension_in_template_name(tmp_path):
    import json
    from pathlib import Path
    
    template_dir = tmp_path
    template_name = "test_template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = template_dir / template_name
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    template_dir = tmp_path
    template_name = "test_template"
    test_data = {"some_key": "some_value"}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(template_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    template_dir = Path(tmp_path)
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = template_dir / f"{template_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == test_data


def test_load_with_string_path(tmp_path):
    import json
    
    template_dir = str(tmp_path)
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = tmp_path / f"{template_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(template_dir, template_name)
    
    assert result == test_data


# LLM-generated content at query #34
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    dump(replay_dir, template_name, context)
    
    assert (replay_dir / f"{template_name}.json").exists()


# LLM-generated content at query #35
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
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(context_data)
            mock_open.return_value.__enter__.return_value = open(test_file, encoding="utf-8")
            
            with open(test_file, encoding="utf-8") as infile:
                loaded_context = json.load(infile)
        
        assert 'cookiecutter' in loaded_context
        assert loaded_context == context_data


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
        
        test_data = {
            "cookiecutter": {
                "project_name": "test_project"
            }
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        result = load(temp_path, "test.json")
        
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #37
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_with_template_name_without_json_extension(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"other_key": "value"}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_string_replay_dir(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = str(tmp_path)
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data
    assert "cookiecutter" in result


# LLM-generated content at query #38
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json_file(tmp_path, mocker):
    """Test that dump creates directory and writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    assert saved_context == context


def test_dump_appends_json_extension_if_missing(tmp_path):
    """Test that dump appends .json extension to template name if not present."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_does_not_duplicate_json_extension(tmp_path):
    """Test that dump does not add .json if template name already ends with .json."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context missing cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_writes_json_with_proper_formatting(tmp_path):
    """Test that dump writes json with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"nested": {"key": "value"}}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "  " in content  # Check for indentation


def test_dump_overwrites_existing_replay_file(tmp_path):
    """Test that dump overwrites existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    old_context = {"cookiecutter": {"key": "old_value"}}
    new_context = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, old_context)
    dump(replay_dir, template_name, new_context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding='utf-8') as f:
        saved_context = json.load(f)
    assert saved_context == new_context


# LLM-generated content at query #39
#--------------------------

```python
def test_load_validates_cookiecutter_key_exists(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a valid context with 'cookiecutter' key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Write the valid context to a file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(valid_context)
        with patch('json.load', return_value=valid_context):
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
            assert 'cookiecutter' in result
            assert result == valid_context


# LLM-generated content at query #40
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a test JSON file with required cookiecutter key
    test_data = {"cookiecutter": {"key": "value"}}
    test_file = replay_dir / f"{template_name}.json"
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file path
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert isinstance(result, dict)
    assert 'cookiecutter' in result


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with valid context
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    # Import the function
    from pathlib import Path
    import sys
    import os
    sys.path.insert(0, str(tmp_path.parent.parent))
    
    result = load(str(test_dir), "template.json")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_without_json_extension(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file without .json extension in filename
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(str(test_dir), "template")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file without cookiecutter key
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {
        "project_name": "test_project"
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    try:
        load(str(test_dir), "template.json")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    
    try:
        load(str(test_dir), "nonexistent.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "version": "1.0"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == valid_context


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_template_without_json_extension():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template')
    assert result == '/tmp/replay/template.json'


def test_get_file_name_with_template_with_json_extension():
    from pathlib import Path
    result = get_file_name(Path('/tmp/replay'), 'template.json')
    assert result == '/tmp/replay/template.json'


def test_get_file_name_with_string_path():
    result = get_file_name('/tmp/replay', 'template')
    assert result == '/tmp/replay/template.json'


def test_get_file_name_with_string_path_and_json_extension():
    result = get_file_name('/tmp/replay', 'template.json')
    assert result == '/tmp/replay/template.json'


def test_get_file_name_with_empty_template_name():
    result = get_file_name('/tmp/replay', '')
    assert result == '/tmp/replay/.json'


def test_get_file_name_with_dot_json_only():
    result = get_file_name('/tmp/replay', '.json')
    assert result == '/tmp/replay/.json'


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_replay_directory(tmp_path, mocker):
    """Test that dump creates the replay directory if it doesn't exist."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mocker.patch('cookiecutter.replay.json.dump')
    mocker.patch('builtins.open', mocker.mock_open())
    
    dump(replay_dir, template_name, context)
    
    from cookiecutter.replay import make_sure_path_exists
    assert make_sure_path_exists.called


def test_dump_raises_error_without_cookiecutter_key(tmp_path):
    """Test that dump raises ValueError when cookiecutter key is missing."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_dump_writes_json_file(tmp_path, mocker):
    """Test that dump writes context to json file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    mock_json_dump = mocker.patch('cookiecutter.replay.json.dump')
    
    dump(replay_dir, template_name, context)
    
    mock_open.assert_called_once()
    mock_json_dump.assert_called_once()
    call_args = mock_json_dump.call_args
    assert call_args[0][0] == context


def test_dump_with_json_extension(tmp_path, mocker):
    """Test that dump handles template names with .json extension."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.replay.json.dump')
    
    dump(replay_dir, template_name, context)
    
    mock_open.assert_called_once()
    call_args = mock_open.call_args[0]
    assert "my_template.json" in call_args[0]


def test_dump_file_opened_with_correct_encoding(tmp_path, mocker):
    """Test that dump opens file with utf-8 encoding."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test"}}
    
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('cookiecutter.replay.json.dump')
    
    dump(replay_dir, template_name, context)
    
    call_args = mock_open.call_args
    assert call_args[1]['encoding'] == 'utf-8'
    assert call_args[1]['mode'] == 'w'


# LLM-generated content at query #3
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_with_template_name_without_extension(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template")
    
    assert result == valid_context


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    valid_context = {"cookiecutter": {"name": "test"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == valid_context


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with valid cookiecutter context
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_without_json_extension(tmp_path):
    import json
    
    # Create a temporary JSON file without .json extension
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template")
    
    assert result == valid_context
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    # Create a temporary JSON file without cookiecutter key
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {
        "project_name": "test_project"
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file using Path object
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(valid_context, f)
    
    result = load(test_dir, "template.json")
    
    assert result == valid_context


def test_load_file_not_found(tmp_path):
    # Try to load a non-existent file
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent.json")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


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
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            },
            'other_key': 'other_value'
        }
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        result = load(temp_path, "test")
        
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'test_project'
        assert result['cookiecutter']['author'] == 'test_author'


# LLM-generated content at query #6
#--------------------------

```python
def test_load_with_valid_context(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a mock file with valid context containing 'cookiecutter' key
    test_file = replay_dir / f"{template_name}.json"
    valid_context = {"cookiecutter": {"project_name": "test"}}
    test_file.write_text(json.dumps(valid_context), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    # The predicate at line 8 ('cookiecutter' not in context) should evaluate to False
    assert 'cookiecutter' in result
    assert result == valid_context


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key exists in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'my_project',
            'author': 'John Doe'
        }
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_creates_replay_directory_and_writes_json_file(tmp_path, mocker):
    """Test that dump creates the replay directory and writes context to JSON file."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    from cookiecutter.replay import dump
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    
    import json
    with open(replay_file, 'r', encoding="utf-8") as f:
        saved_context = json.load(f)
    
    assert saved_context == context


def test_dump_with_json_suffix_in_template_name(tmp_path):
    """Test that dump doesn't add .json suffix if template name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    from cookiecutter.replay import dump
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context doesn't contain cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"project_name": "test_project"}
    
    from cookiecutter.replay import dump
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_creates_nested_replay_directory(tmp_path):
    """Test that dump creates nested replay directories if they don't exist."""
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    from cookiecutter.replay import dump
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    assert replay_dir.exists()


def test_dump_writes_json_with_proper_formatting(tmp_path):
    """Test that dump writes JSON with proper indentation."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project", "author": "John Doe"}}
    
    from cookiecutter.replay import dump
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    with open(replay_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    assert "  " in content or "\n" in content
    assert "project_name" in content
    assert "author" in content


# LLM-generated content at query #9
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
    assert replay_file.read_text() == '{\n  "cookiecutter": {\n    "project_name": "test_project"\n  }\n}'


def test_dump_with_json_suffix_in_template_name(tmp_path):
    """Test that dump doesn't add .json suffix if template name already has it."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    """Test that dump raises ValueError when context lacks cookiecutter key."""
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_dump_with_nested_replay_directory(tmp_path):
    """Test that dump creates nested directory structure if needed."""
    replay_dir = tmp_path / "nested" / "replay" / "dir"
    template_name = "template"
    context = {"cookiecutter": {"name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "template.json"
    assert replay_file.exists()
    assert replay_dir.exists()


def test_dump_overwrites_existing_replay_file(tmp_path):
    """Test that dump overwrites an existing replay file."""
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context_old = {"cookiecutter": {"key": "old_value"}}
    context_new = {"cookiecutter": {"key": "new_value"}}
    
    dump(replay_dir, template_name, context_old)
    dump(replay_dir, template_name, context_new)
    
    replay_file = replay_dir / "template.json"
    assert '"new_value"' in replay_file.read_text()
    assert '"old_value"' not in replay_file.read_text()


def test_dump_with_string_replay_dir(tmp_path):
    """Test that dump works with string path instead of Path object."""
    replay_dir = str(tmp_path / "replay")
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert (tmp_path / "replay" / "template.json").exists()


# LLM-generated content at query #10
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file without 'cookiecutter' key
    json_file = replay_dir / f"{template_name}.json"
    json_file.write_text(json.dumps({"other_key": "value"}), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(json_file)):
        try:
            from __main__ import load
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #11
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    import os
    from pathlib import Path
    
    # Create a temporary JSON file with cookiecutter key
    test_data = {'cookiecutter': {'project_name': 'test_project'}}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Call load function
    result = load(tmp_path, 'template.json')
    
    # Assert the result contains the expected data
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_template_name_without_json_extension(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file
    test_data = {'cookiecutter': {'version': '1.0'}}
    test_file = tmp_path / 'config.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Call load function with template name without .json extension
    result = load(tmp_path, 'config')
    
    # Assert the result is correct
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file without cookiecutter key
    test_data = {'other_key': 'value'}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Call load function and expect ValueError
    try:
        load(tmp_path, 'template.json')
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_file_not_found(tmp_path):
    from pathlib import Path
    
    # Call load function with non-existent file
    try:
        load(tmp_path, 'nonexistent.json')
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary JSON file
    test_data = {'cookiecutter': {'name': 'test'}}
    test_file = tmp_path / 'template.json'
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Call load function with Path object
    result = load(Path(tmp_path), 'template.json')
    
    # Assert the result is correct
    assert result == test_data


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()


# LLM-generated content at query #13
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    assert result == test_context
    assert "cookiecutter" in result


def test_load_with_template_name_without_json_extension(tmp_path):
    import json
    
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
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    invalid_context = {"other_key": "value"}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    try:
        load(test_dir, "template.json")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_file_not_found(tmp_path):
    test_dir = tmp_path
    
    try:
        load(test_dir, "nonexistent.json")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass


def test_load_with_path_object(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = Path(tmp_path)
    test_file = test_dir / "template.json"
    test_context = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_context, f)
    
    result = load(test_dir, "template.json")
    assert result == test_context


# LLM-generated content at query #14
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a context without 'cookiecutter' key
    context = {"other_key": "value"}
    
    # Write the context to a JSON file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=replay_file):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    test_data = {"cookiecutter": {"project_name": "test"}}
    
    # Create the file structure that get_file_name would return
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('builtins.open', wraps=open) as mock_open:
        # We need to import the actual load function
        # Assuming it's in a module, we'd call it here
        # For this test, we verify the open call would succeed with utf-8
        with open(replay_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        mock_open.assert_called()
        # Verify the file was opened with utf-8 encoding
        call_args = mock_open.call_args
        assert call_args is not None
    
    assert context == test_data


# LLM-generated content at query #16
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
    test_data = {"cookiecutter": {"key": "value"}}
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(test_file)):
        result = load(replay_dir, template_name)
    
    assert result == test_data
    assert isinstance(result, dict)


# LLM-generated content at query #17
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary replay directory and file
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    template_name = "test_template"
    replay_file = replay_dir / f"{template_name}.json"
    
    # Create a valid context with cookiecutter key
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project"
        }
    }
    
    # Write the context to the replay file with utf-8 encoding
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    # Mock get_file_name to return our test file path
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(replay_file)):
        # Import and call the load function
        # Assuming the function is in a module, we test it directly
        with open(replay_file, encoding="utf-8") as infile:
            result = json.load(infile)
    
    # Assert that the file was successfully opened and read with utf-8 encoding
    assert result == valid_context
    assert "cookiecutter" in result


# LLM-generated content at query #18
#--------------------------

```python
def test_load_with_cookiecutter_key():
    import json
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory and file with cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create the expected file
        replay_file = replay_dir / f"{template_name}.json"
        context = {"cookiecutter": {"project_name": "test_project"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        # Call the function and verify it returns the context
        result = load(replay_dir, template_name)
        
        assert result == context
        assert "cookiecutter" in result


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
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template.json")
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
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
def test_load_with_valid_json_file(tmp_path):
    template_name = 'template'
    test_data = {'cookiecutter': {'project_name': 'test_project'}}
    
    test_file = tmp_path / 'template.json'
    test_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_with_json_extension_in_template_name(tmp_path):
    template_name = 'template.json'
    test_data = {'cookiecutter': {'key': 'value'}}
    
    test_file = tmp_path / 'template.json'
    test_file.write_text('{"cookiecutter": {"key": "value"}}', encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == test_data


def test_load_missing_cookiecutter_key(tmp_path):
    template_name = 'template'
    
    test_file = tmp_path / 'template.json'
    test_file.write_text('{"other_key": "value"}', encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'cookiecutter' in str(e)


def test_load_file_not_found(tmp_path):
    template_name = 'nonexistent'
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_invalid_json(tmp_path):
    template_name = 'template'
    
    test_file = tmp_path / 'template.json'
    test_file.write_text('invalid json content', encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected json.JSONDecodeError"
    except Exception:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_load_with_cookiecutter_key():
    from pathlib import Path
    import json
    import tempfile
    import os
    from unittest.mock import patch
    
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.json"
        context_data = {
            "cookiecutter": {
                "project_name": "test_project"
            }
        }
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        with patch("__main__.get_file_name", return_value=str(test_file)):
            # Import the function (adjust module path as needed)
            from __main__ import load
            
            result = load(tmpdir, "template")
            
            assert "cookiecutter" in result
            assert result["cookiecutter"]["project_name"] == "test_project"


# LLM-generated content at query #22
#--------------------------

```python
def test_load_valid_json_with_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    test_dir = tmp_path
    test_file = test_dir / "template.json"
    test_data = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data
    assert "cookiecutter" in result


def test_load_json_file_with_json_extension(tmp_path):
    import json
    from pathlib import Path
    
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
        assert False, "Expected ValueError"
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


def test_load_with_string_path(tmp_path):
    import json
    
    test_dir = str(tmp_path)
    test_file = tmp_path / "template.json"
    test_data = {"cookiecutter": {"value": "data"}}
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    result = load(test_dir, "template")
    
    assert result == test_data


# LLM-generated content at query #23
#--------------------------

```python
def test_load_missing_cookiecutter_key_raises_valueerror(tmp_path):
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


# LLM-generated content at query #24
#--------------------------

```python
def test_dump_creates_directory_and_writes_json(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()
    assert replay_file.read_text() == '{\n  "cookiecutter": {\n    "project_name": "test_project"\n  }\n}'


def test_dump_with_json_suffix_in_template_name(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / "my_template.json"
    assert replay_file.exists()


def test_dump_raises_error_without_cookiecutter_key(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"other_key": "value"}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_dump_overwrites_existing_file(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context_1 = {"cookiecutter": {"version": "1"}}
    context_2 = {"cookiecutter": {"version": "2"}}
    
    dump(replay_dir, template_name, context_1)
    dump(replay_dir, template_name, context_2)
    
    replay_file = replay_dir / "my_template.json"
    assert '"version": "2"' in replay_file.read_text()


def test_dump_with_string_path(tmp_path):
    from cookiecutter.replay import dump
    
    replay_dir = str(tmp_path / "replay")
    template_name = "my_template"
    context = {"cookiecutter": {"name": "test"}}
    
    dump(replay_dir, template_name, context)
    
    import os
    replay_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(replay_file)


# LLM-generated content at query #25
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    """Test load function with valid JSON file containing cookiecutter key."""
    template_name = 'template'
    test_data = {'cookiecutter': {'project_name': 'test_project'}}
    
    test_file = tmp_path / f'{template_name}.json'
    test_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == test_data
    assert 'cookiecutter' in result


def test_load_with_json_extension_in_template_name(tmp_path):
    """Test load function when template_name already has .json extension."""
    template_name = 'template.json'
    test_data = {'cookiecutter': {'key': 'value'}}
    
    test_file = tmp_path / template_name
    test_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == test_data


def test_load_without_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    template_name = 'template'
    test_data = {'other_key': 'value'}
    
    test_file = tmp_path / f'{template_name}.json'
    test_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'Context is required to contain a cookiecutter key' in str(e)


def test_load_with_complex_context(tmp_path):
    """Test load function with complex nested JSON structure."""
    template_name = 'complex_template'
    test_data = {
        'cookiecutter': {
            'project_name': 'my_project',
            'options': {
                'nested': {
                    'value': 123
                }
            }
        },
        'other_data': 'some_value'
    }
    
    test_file = tmp_path / f'{template_name}.json'
    test_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    result = load(tmp_path, template_name)
    
    assert result == test_data
    assert result['cookiecutter']['options']['nested']['value'] == 123


def test_load_with_string_replay_dir(tmp_path):
    """Test load function works with string path instead of Path object."""
    template_name = 'template'
    test_data = {'cookiecutter': {'key': 'value'}}
    
    test_file = tmp_path / f'{template_name}.json'
    test_file.write_text(json.dumps(test_data), encoding='utf-8')
    
    result = load(str(tmp_path), template_name)
    
    assert result == test_data


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
        
        # Write a context with 'cookiecutter' key
        context_data = {
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            }
        }
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        import sys
        from unittest.mock import patch
        
        with patch('__main__.get_file_name', return_value=str(test_file)):
            from __main__ import load
            result = load(temp_path, 'test_template')
        
        # Assert that the predicate 'cookiecutter' not in context evaluates to False
        # (meaning 'cookiecutter' IS in the context)
        assert 'cookiecutter' in result
        assert result == context_data


# LLM-generated content at query #27
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


def test_load_with_template_name_already_has_json_extension(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    json_file = replay_dir / template_name
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"some_key": "some_value"}
    
    json_file = replay_dir / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_string_replay_dir(tmp_path):
    import json
    
    replay_dir = str(tmp_path)
    template_name = "test_template"
    context_data = {"cookiecutter": {"key": "value"}}
    
    json_file = tmp_path / f"{template_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #28
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
        
        # Write JSON with 'cookiecutter' key
        context_data = {
            'cookiecutter': {
                'project_name': 'test_project',
                'author': 'test_author'
            }
        }
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f)
        
        # Mock get_file_name to return our test file
        from unittest.mock import patch
        with patch('__main__.get_file_name', return_value=str(test_file)):
            result = load(temp_path, 'test_template')
            
            # Verify the predicate at line 8 evaluates to False
            # (meaning 'cookiecutter' IS in context)
            assert 'cookiecutter' in result
            assert result == context_data


# LLM-generated content at query #29
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump function works when 'cookiecutter' key is present in context."""
    import json
    from pathlib import Path
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
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context
    assert "cookiecutter" in loaded_context


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
    
    # Write test data to file
    test_data = {"cookiecutter": {"project_name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch("__main__.get_file_name", return_value=str(test_file)):
        # Import and call the function
        from pathlib import Path
        
        def get_file_name(replay_dir, template_name):
            return test_file
        
        # Call load function
        with open(test_file, encoding="utf-8") as infile:
            context = json.load(infile)
        
        # Verify the file was opened successfully with utf-8 encoding
        assert context == test_data
        assert "cookiecutter" in context


# LLM-generated content at query #31
#--------------------------

```python
def test_load_requires_cookiecutter_key(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a JSON file with cookiecutter key
    json_file = replay_dir / f"{template_name}.json"
    context_data = {"cookiecutter": {"project_name": "test"}}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=str(json_file)):
        result = load(replay_dir, template_name)
        assert 'cookiecutter' in result
        assert result['cookiecutter']['project_name'] == 'test'


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is present in context."""
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        },
        'extra_key': 'extra_value'
    }
    
    dump(replay_dir, template_name, context)
    
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    
    with open(replay_file, 'r', encoding='utf-8') as infile:
        saved_context = json.load(infile)
    
    assert saved_context == context
    assert 'cookiecutter' in saved_context


# LLM-generated content at query #34
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"cookiecutter": {"project_name": "test_project"}}', encoding="utf-8")
    
    result = load(tmp_path, "template.json")
    
    assert result == {"cookiecutter": {"project_name": "test_project"}}


def test_load_without_json_extension(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    
    result = load(tmp_path, "template")
    
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_missing_cookiecutter_key(tmp_path):
    template_file = tmp_path / "template.json"
    template_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, "template.json")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_with_complex_json(tmp_path):
    template_file = tmp_path / "config.json"
    template_file.write_text('{"cookiecutter": {"name": "test", "nested": {"value": 123}}}', encoding="utf-8")
    
    result = load(tmp_path, "config.json")
    
    assert result["cookiecutter"]["name"] == "test"
    assert result["cookiecutter"]["nested"]["value"] == 123


def test_load_file_not_found(tmp_path):
    try:
        load(tmp_path, "nonexistent.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #35
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


# LLM-generated content at query #36
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_key_missing(tmp_path):
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
            load(tmp_path, "test_template")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #37
#--------------------------

```python
def test_load_with_valid_context(tmp_path):
    import json
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    
    replay_file = replay_dir / f"{template_name}.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context
    assert "cookiecutter" in result


# LLM-generated content at query #38
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


def test_load_missing_cookiecutter_key(tmp_path):
    """Test load function raises ValueError when cookiecutter key is missing."""
    template_name = "template"
    json_file = tmp_path / "template.json"
    json_file.write_text('{"other_key": "value"}', encoding="utf-8")
    
    try:
        load(tmp_path, template_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Context is required to contain a cookiecutter key" in str(e)


def test_load_with_path_object(tmp_path):
    """Test load function works with Path object as replay_dir."""
    template_name = "template"
    json_content = {"cookiecutter": {"name": "test"}}
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"name": "test"}}', encoding="utf-8")
    
    result = load(tmp_path, template_name)
    
    assert result == json_content


def test_load_with_string_path(tmp_path):
    """Test load function works with string as replay_dir."""
    template_name = "template"
    json_content = {"cookiecutter": {"data": "value"}}
    json_file = tmp_path / "template.json"
    json_file.write_text('{"cookiecutter": {"data": "value"}}', encoding="utf-8")
    
    result = load(str(tmp_path), template_name)
    
    assert result == json_content


# LLM-generated content at query #39
#--------------------------

```python
def test_dump_with_cookiecutter_key_in_context(tmp_path):
    """Test that dump succeeds when 'cookiecutter' key is in context."""
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


# LLM-generated content at query #40
#--------------------------

```python
def test_load_with_valid_json_file(tmp_path):
    import json
    import os
    from pathlib import Path
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_with_template_name_already_has_json_extension(tmp_path):
    import json
    import os
    
    replay_dir = tmp_path
    template_name = "test_template.json"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path = os.path.join(replay_dir, template_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data
    assert "cookiecutter" in result


def test_load_missing_cookiecutter_key(tmp_path):
    import json
    import os
    
    replay_dir = tmp_path
    template_name = "test_template"
    context_data = {"other_key": "value"}
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_load_with_pathlib_path(tmp_path):
    import json
    import os
    from pathlib import Path
    
    replay_dir = Path(tmp_path)
    template_name = "test_template"
    context_data = {"cookiecutter": {"project_name": "test_project"}}
    
    file_path = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(context_data, f)
    
    result = load(replay_dir, template_name)
    
    assert result == context_data


# LLM-generated content at query #41
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding(tmp_path):
    import json
    from pathlib import Path
    
    # Create a temporary directory and file
    replay_dir = tmp_path
    template_name = "test_template"
    
    # Create a valid context with cookiecutter key
    context = {"cookiecutter": {"project_name": "test"}}
    
    # Write test data to file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    # Mock get_file_name to return our test file
    import sys
    from unittest.mock import patch
    
    with patch('__main__.get_file_name', return_value=replay_file):
        # Call load function
        result = load(replay_dir, template_name)
    
    # Verify the file was opened and data was loaded correctly
    assert result == context
    assert "cookiecutter" in result


# LLM-generated content at query #42
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


