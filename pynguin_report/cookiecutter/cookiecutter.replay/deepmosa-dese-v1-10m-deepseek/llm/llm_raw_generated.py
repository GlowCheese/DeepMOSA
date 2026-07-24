####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_json_suffix():
    replay_dir = "/test/dir"
    template_name = "test.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/test/dir/test.json"

def test_get_file_name_without_json_suffix():
    replay_dir = "/test/dir"
    template_name = "test"
    result = get_file_name(replay_dir, template_name)
    assert result == "/test/dir/test.json"

def test_get_file_name_with_path_object():
    replay_dir = Path("/test/dir")
    template_name = "test"
    result = get_file_name(replay_dir, template_name)
    assert result == str(Path("/test/dir/test.json"))


# LLM-generated content at query #2
#--------------------------

```python
def test_load_success():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    test_data = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(test_data))
    with unittest.mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == test_data

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    test_data = {'key': 'value'}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(test_data))
    with unittest.mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_json_extension():
    replay_dir = '/tmp/replay'
    template_name = 'template.json'
    test_data = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(test_data))
    with unittest.mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == test_data


# LLM-generated content at query #3
#--------------------------

def test_dump_creates_directory_if_not_exists():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    assert os.path.exists(test_dir)
    os.rmdir(test_dir)

def test_dump_raises_value_error_without_cookiecutter_key():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'key': 'value'}
    try:
        dump(test_dir, test_template, test_context)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_dump_creates_json_file():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    expected_file = os.path.join(test_dir, 'test_template.json')
    assert os.path.exists(expected_file)
    os.remove(expected_file)
    os.rmdir(test_dir)

def test_dump_handles_existing_json_suffix():
    test_dir = 'test_replay_dir'
    test_template = 'test_template.json'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    expected_file = os.path.join(test_dir, 'test_template.json')
    assert os.path.exists(expected_file)
    os.remove(expected_file)
    os.rmdir(test_dir)


# LLM-generated content at query #4
#--------------------------

```
def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    from unittest.mock import patch

    test_data = {'cookiecutter': {'project_name': 'test'}}
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    with patch('pathlib.Path.exists', return_value=True):
        result = load(temp_file_path, 'template')
        assert 'cookiecutter' in result


# LLM-generated content at query #5
#--------------------------

```python
def test_dump_raises_value_error_when_context_does_not_contain_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #6
#--------------------------

```python
def test_load_returns_context_with_cookiecutter_key():
    context = load("some_dir", "template_name")
    assert 'cookiecutter' in context


# LLM-generated content at query #7
#--------------------------

```python
def test_load_returns_dict_with_cookiecutter_key():
    test_dir = Path("test_dir")
    test_template = "test_template"
    test_data = {"cookiecutter": {"key": "value"}}
    
    with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
        result = load(test_dir, test_template)
    
    assert isinstance(result, dict)
    assert "cookiecutter" in result


# LLM-generated content at query #8
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = "test_dir"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

```
def test_load_successfully_reads_json_file(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_path.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_handles_template_name_without_json_extension(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_path.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_handles_template_name_with_json_extension(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template.json"
    file_path = replay_dir / template_name
    file_path.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")
    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}


def test_load_raises_value_error_when_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_path.write_text('{"key": "value"}', encoding="utf-8")
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_raises_file_not_found_error_when_file_does_not_exist(tmp_path):
    replay_dir = tmp_path
    template_name = "nonexistent_template"
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_load_success():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template_missing_key.json"
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_without_json_extension():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #11
#--------------------------

```python
def test_dump_raises_value_error_when_context_does_not_contain_cookiecutter_key():
    test_replay_dir = "test_dir"
    test_template_name = "test_template"
    test_context = {"key": "value"}
    try:
        dump(test_replay_dir, test_template_name, test_context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #12
#--------------------------

```python
def test_load_with_cookiecutter_key():
    context = load("test_dir", "template_name")
    assert 'cookiecutter' in context

def test_load_without_cookiecutter_key():
    try:
        load("test_dir", "invalid_template")
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #13
#--------------------------

```
def test_load_with_valid_json_file():
    replay_dir = "/tmp"
    template_name = "valid_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open("/tmp/valid_template.json", "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove("/tmp/valid_template.json")

def test_load_with_json_extension_in_name():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open("/tmp/template.json", "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove("/tmp/template.json")

def test_load_with_missing_cookiecutter_key():
    replay_dir = "/tmp"
    template_name = "invalid_template"
    invalid_context = {"key": "value"}
    with open("/tmp/invalid_template.json", "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    try:
        load(replay_dir, template_name)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    os.remove("/tmp/invalid_template.json")

def test_load_with_nonexistent_file():
    replay_dir = "/tmp"
    template_name = "nonexistent"
    try:
        load(replay_dir, template_name)
        assert False
    except FileNotFoundError:
        assert True


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_invalid_replay_file():
    replay_dir = Path("invalid_dir")
    template_name = "invalid_template"
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #15
#--------------------------

```
def test_load_without_cookiecutter_key_raises_value_error():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch

    test_data = '{"some_key": "some_value"}'
    with patch('builtins.open', mock_open(read_data=test_data)):
        try:
            load(Path("fake_dir"), "fake_template")
            assert False, "Expected ValueError but no exception was raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #16
#--------------------------

```python
def test_dump_raises_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    from cookiecutter.replay import dump

    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': {}}

    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #17
#--------------------------

```python
def test_load_with_valid_file_and_context():
    replay_dir = 'test_dir'
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    
    with open(os.path.join(replay_dir, template_name), 'w', encoding='utf-8') as file:
        json.dump(expected_context, file)
    
    context = load(replay_dir, template_name)
    assert context == expected_context

def test_load_with_invalid_context():
    replay_dir = 'test_dir'
    template_name = 'test_template_invalid.json'
    invalid_context = {'key': 'value'}
    
    with open(os.path.join(replay_dir, template_name), 'w', encoding='utf-8') as file:
        json.dump(invalid_context, file)
    
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_missing_file():
    replay_dir = 'test_dir'
    template_name = 'non_existent_template.json'
    
    try:
        load(replay_dir, template_name)
    except FileNotFoundError:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_load_returns_dict_with_cookiecutter_key():
    test_dir = Path("test_dir")
    test_name = "test_template"
    mock_file_content = '{"cookiecutter": {"key": "value"}}'
    
    def mock_get_file_name(dir, name):
        return "mock_file_path"
    
    def mock_open(file, encoding):
        from io import StringIO
        return StringIO(mock_file_content)
    
    original_get_file_name = globals().get("get_file_name")
    globals()["get_file_name"] = mock_get_file_name
    original_open = builtins.open
    builtins.open = mock_open
    
    try:
        result = load(test_dir, test_name)
        assert isinstance(result, dict)
        assert "cookiecutter" in result
    finally:
        globals()["get_file_name"] = original_get_file_name
        builtins.open = original_open


# LLM-generated content at query #19
#--------------------------

```python
def test_load_success():
    replay_dir = '/tmp/replay'
    template_name = 'test_template'
    test_data = {'cookiecutter': {'key': 'value'}}
    with open(f'{replay_dir}/{template_name}.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/replay'
    template_name = 'invalid_template'
    test_data = {'key': 'value'}
    with open(f'{replay_dir}/{template_name}.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    try:
        load(replay_dir, template_name)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_extension():
    replay_dir = '/tmp/replay'
    template_name = 'test_template.json'
    test_data = {'cookiecutter': {'key': 'value'}}
    with open(f'{replay_dir}/{template_name}', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data


# LLM-generated content at query #20
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_key_is_missing():
    replay_dir = '/path/to/replay'
    template_name = 'template_name'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #21
#--------------------------

```python
def test_load_valid_file():
    replay_dir = "test_dir"
    template_name = "valid_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as file:
        json.dump(expected_context, file)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_valid_file_without_json_suffix():
    replay_dir = "test_dir"
    template_name = "valid_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as file:
        json.dump(expected_context, file)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_invalid_file_missing_cookiecutter():
    replay_dir = "test_dir"
    template_name = "invalid_template.json"
    invalid_context = {"key": "value"}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as file:
        json.dump(invalid_context, file)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_load_nonexistent_file():
    replay_dir = "test_dir"
    template_name = "nonexistent_template.json"
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


# LLM-generated content at query #22
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = 'some_dir'
    template_name = 'some_template'
    context = {'some_key': 'some_value'}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #23
#--------------------------

```python
def test_load_valid_template():
    replay_dir = "test_dir"
    template_name = "valid_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as file:
        json.dump(expected_context, file)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_template_without_cookiecutter():
    replay_dir = "test_dir"
    template_name = "invalid_template.json"
    invalid_context = {"key": "value"}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as file:
        json.dump(invalid_context, file)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_template_with_json_suffix():
    replay_dir = "test_dir"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as file:
        json.dump(expected_context, file)
    result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #24
#--------------------------

```python
def test_load_with_invalid_replay_file():
    invalid_path = Path("/nonexistent/path")
    template = "invalid_template"
    try:
        load(invalid_path, template)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #25
#--------------------------

def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'not_cookiecutter': {}}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #26
#--------------------------

def test_dump_does_not_raise_value_error_when_context_contains_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)


# LLM-generated content at query #27
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #28
#--------------------------

def test_dump_creates_file_with_valid_context():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists

    with tempfile.TemporaryDirectory() as tmp_dir:
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}
        dump(tmp_dir, template_name, context)
        
        replay_file = Path(tmp_dir) / f"{template_name}.json"
        assert replay_file.exists()
        
        with open(replay_file, encoding="utf-8") as infile:
            loaded_context = json.load(infile)
            assert loaded_context == context


# LLM-generated content at query #29
#--------------------------

def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'missing_cookiecutter_key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #30
#--------------------------

```python
def test_dump_raises_value_error_when_context_does_not_contain_cookiecutter_key():
    context = {'not_cookiecutter': 'value'}
    template_name = 'template'
    replay_dir = 'path/to/replay'
    
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #31
#--------------------------

```python
def test_dump_writes_to_file(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = replay_dir / f"{template_name}.json"

    dump(replay_dir, template_name, context)

    assert replay_file.exists()
    with open(replay_file, encoding="utf-8") as infile:
        assert json.load(infile) == context


# LLM-generated content at query #32
#--------------------------

```python
def test_dump_creates_directory_and_writes_file():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert Path(replay_dir).exists()
    assert Path(replay_file).exists()
    
    with open(replay_file, "r", encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context
    
    Path(replay_file).unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #33
#--------------------------

```python
def test_dump_creates_file_with_context():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert Path(replay_file).exists()


# LLM-generated content at query #34
#--------------------------

```python
def test_load_raises_value_error_if_cookiecutter_not_in_context():
    context = {}
    try:
        load("/path/to/replay_dir", "template_name")
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #35
#--------------------------

def test_dump_creates_directory_if_not_exists(tmp_path):
    replay_dir = tmp_path / "nonexistent_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_raises_value_error_without_cookiecutter_key(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_dump_creates_correct_json_file(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = tmp_path / "test_template.json"
    assert expected_file.exists()
    with open(expected_file, encoding="utf-8") as f:
        content = json.load(f)
    assert content == context


def test_dump_handles_existing_json_suffix(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = tmp_path / "test_template.json"
    assert expected_file.exists()


# LLM-generated content at query #36
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    test_dir = Path('test_dir')
    test_template = 'test_template'
    invalid_context = {'some_key': 'some_value'}
    
    try:
        dump(test_dir, test_template, invalid_context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #37
#--------------------------

```python
def test_load_should_return_dict_when_file_exists_and_contains_cookiecutter_key():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    mock_file_content = {'cookiecutter': {'key': 'value'}}
    expected_result = {'cookiecutter': {'key': 'value'}}
    mock_file_path = os.path.join(replay_dir, f'{template_name}.json')
    
    with open(mock_file_path, 'w', encoding="utf-8") as mock_file:
        json.dump(mock_file_content, mock_file)
    
    result = load(replay_dir, template_name)
    assert result == expected_result

def test_load_should_raise_value_error_when_file_exists_but_missing_cookiecutter_key():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    mock_file_content = {'key': 'value'}
    mock_file_path = os.path.join(replay_dir, f'{template_name}.json')
    
    with open(mock_file_path, 'w', encoding="utf-8") as mock_file:
        json.dump(mock_file_content, mock_file)
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_should_raise_file_not_found_error_when_file_does_not_exist():
    replay_dir = '/path/to/replay'
    template_name = 'nonexistent_template'
    
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError:
        assert True


# LLM-generated content at query #38
#--------------------------

```python
def test_load_successfully_reads_valid_json_file():
    replay_dir = "test_dir"
    template_name = "valid_template"
    test_file = "test_dir/valid_template.json"
    test_content = '{"cookiecutter": {"key": "value"}}'
    
    import json
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    replay_dir_path = temp_dir / replay_dir
    replay_dir_path.mkdir()
    test_file_path = replay_dir_path / (template_name + ".json")
    test_file_path.write_text(test_content)

    def get_file_name(dir_path, name):
        return dir_path / (name + ".json")

    context = load(replay_dir_path, template_name)
    assert context == {"cookiecutter": {"key": "value"}}
    shutil.rmtree(temp_dir)

def test_load_raises_error_when_cookiecutter_key_missing():
    replay_dir = "test_dir"
    template_name = "invalid_template"
    test_file = "test_dir/invalid_template.json"
    test_content = '{"key": "value"}'
    
    import json
    from pathlib import Path
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    replay_dir_path = temp_dir / replay_dir
    replay_dir_path.mkdir()
    test_file_path = replay_dir_path / (template_name + ".json")
    test_file_path.write_text(test_content)

    def get_file_name(dir_path, name):
        return dir_path / (name + ".json")

    try:
        load(replay_dir_path, template_name)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    shutil.rmtree(temp_dir)


# LLM-generated content at query #39
#--------------------------

def test_dump_writes_file():
    replay_dir = "/tmp/test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / f"{template_name}.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #40
#--------------------------

```python
def test_dump_successfully_writes_to_file(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as infile:
        assert json.load(infile) == context


# LLM-generated content at query #41
#--------------------------

```python
def test_load_successful_with_json_suffix():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected = {"cookiecutter": {"key": "value"}}
    with patch("builtins.open", mock_open(read_data=json.dumps(expected))):
        result = load(replay_dir, template_name)
    assert result == expected

def test_load_successful_without_json_suffix():
    replay_dir = "/tmp"
    template_name = "template"
    expected = {"cookiecutter": {"key": "value"}}
    with patch("builtins.open", mock_open(read_data=json.dumps(expected))):
        result = load(replay_dir, template_name)
    assert result == expected

def test_load_missing_cookiecutter_key():
    replay_dir = "/tmp"
    template_name = "template.json"
    invalid_data = {"key": "value"}
    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_data))):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            load(replay_dir, template_name)

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "nonexistent.json"
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load(replay_dir, template_name)

def test_load_invalid_json():
    replay_dir = "/tmp"
    template_name = "invalid.json"
    with patch("builtins.open", mock_open(read_data="invalid json")):
        with pytest.raises(json.JSONDecodeError):
            load(replay_dir, template_name)


# LLM-generated content at query #42
#--------------------------

```python
def test_load_returns_context_with_cookiecutter_key():
    mock_file_content = '{"cookiecutter": {"key": "value"}}'
    mock_file_path = "mock_file.json"
    mock_open = lambda file, encoding: mock_file_content
    mock_get_file_name = lambda replay_dir, template_name: mock_file_path
    original_open = open
    original_get_file_name = get_file_name
    open = mock_open
    get_file_name = mock_get_file_name
    result = load("mock_replay_dir", "mock_template")
    open = original_open
    get_file_name = original_get_file_name
    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #43
#--------------------------

```
def test_load_with_non_existent_file():
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, mock_open

    non_existent_file = Path(tempfile.gettempdir()) / "non_existent.json"
    if os.path.exists(non_existent_file):
        os.remove(non_existent_file)

    with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
        try:
            load(non_existent_file, "template")
        except FileNotFoundError:
            pass


# LLM-generated content at query #44
#--------------------------

```
def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch

    test_data = {'cookiecutter': {'key': 'value'}}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch('builtins.open', mock_file):
        result = load(Path('/test'), 'template')
        assert 'cookiecutter' in result


# LLM-generated content at query #45
#--------------------------

def test_dump_creates_directory_if_not_exists(mocker):
    mock_make_sure_path_exists = mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    test_dir = '/test/dir'
    test_template = 'template'
    test_context = {'cookiecutter': {'key': 'value'}}
    cookiecutter.replay.dump(test_dir, test_template, test_context)
    mock_make_sure_path_exists.assert_called_once_with(test_dir)

def test_dump_raises_value_error_without_cookiecutter_key():
    test_dir = '/test/dir'
    test_template = 'template'
    test_context = {'key': 'value'}
    try:
        cookiecutter.replay.dump(test_dir, test_template, test_context)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_dump_writes_correct_json_content(mocker):
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    test_dir = '/test/dir'
    test_template = 'template'
    test_context = {'cookiecutter': {'key': 'value'}}
    cookiecutter.replay.dump(test_dir, test_template, test_context)
    mock_open().write.assert_called_once_with('{\n  "cookiecutter": {\n    "key": "value"\n  }\n}')

def test_dump_uses_correct_file_path(mocker):
    mocker.patch('cookiecutter.replay.make_sure_path_exists')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    test_dir = '/test/dir'
    test_template = 'template'
    test_context = {'cookiecutter': {'key': 'value'}}
    cookiecutter.replay.dump(test_dir, test_template, test_context)
    mock_open.assert_called_once_with('/test/dir/template.json', 'w', encoding='utf-8')


# LLM-generated content at query #46
#--------------------------

```
def test_load_successfully_reads_json_file(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_content = '{"cookiecutter": {"key": "value"}}'
    file_path.write_text(file_content)
    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}

def test_load_adds_json_extension_if_missing(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_content = '{"cookiecutter": {"key": "value"}}'
    file_path.write_text(file_content)
    result = load(replay_dir, template_name[:-5])
    assert result == {"cookiecutter": {"key": "value"}}

def test_load_raises_value_error_when_no_cookiecutter_key(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_content = '{"key": "value"}'
    file_path.write_text(file_content)
    try:
        load(replay_dir, template_name)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_handles_path_object_input(tmp_path):
    replay_dir = tmp_path
    template_name = "test_template"
    file_path = replay_dir / f"{template_name}.json"
    file_content = '{"cookiecutter": {"key": "value"}}'
    file_path.write_text(file_content)
    result = load(Path(replay_dir), template_name)
    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #47
#--------------------------

def test_dump_handles_invalid_context():
    from pathlib import Path
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    import json

    test_replay_dir = Path('/tmp/test_replay')
    test_template_name = 'test_template'
    invalid_context = {'not_cookiecutter': 'value'}

    try:
        dump(test_replay_dir, test_template_name, invalid_context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #48
#--------------------------

```python
def test_load_with_cookiecutter_key():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    context = {'cookiecutter': {}}
    replay_dir = Path('/path/to/replay')
    template_name = 'template.json'
    mock_file_content = json.dumps(context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        result = load(replay_dir, template_name)
        assert 'cookiecutter' in result


# LLM-generated content at query #49
#--------------------------

def test_dump_creates_file_with_cookiecutter_context():
    import json
    import tempfile
    from pathlib import Path
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists

    with tempfile.TemporaryDirectory() as tmp_dir:
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}
        dump(tmp_dir, template_name, context)
        
        replay_file = Path(tmp_dir) / f"{template_name}.json"
        assert replay_file.exists()
        
        with open(replay_file, encoding="utf-8") as infile:
            loaded_context = json.load(infile)
        assert loaded_context == context


# LLM-generated content at query #50
#--------------------------

```python
def test_load_returns_context_with_cookiecutter_key():
    replay_dir = "path/to/replay"
    template_name = "template"
    context = load(replay_dir, template_name)
    assert 'cookiecutter' in context


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_file_name_with_path_object_and_json_suffix():
    replay_dir = Path('/path/to/replay')
    template_name = 'template.json'
    result = get_file_name(replay_dir, template_name)
    assert result == '/path/to/replay/template.json'

def test_get_file_name_with_path_object_and_no_suffix():
    replay_dir = Path('/path/to/replay')
    template_name = 'template'
    result = get_file_name(replay_dir, template_name)
    assert result == '/path/to/replay/template.json'

def test_get_file_name_with_str_and_json_suffix():
    replay_dir = '/path/to/replay'
    template_name = 'template.json'
    result = get_file_name(replay_dir, template_name)
    assert result == '/path/to/replay/template.json'

def test_get_file_name_with_str_and_no_suffix():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    result = get_file_name(replay_dir, template_name)
    assert result == '/path/to/replay/template.json'


# LLM-generated content at query #2
#--------------------------

```python
def test_load_success():
    replay_dir = '/tmp/replay'
    template_name = 'test_template'
    test_data = {'cookiecutter': {'key': 'value'}}
    with open(f'/tmp/replay/test_template.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data

def test_load_missing_cookiecutter_key():
    replay_dir = '/tmp/replay'
    template_name = 'invalid_template'
    test_data = {'key': 'value'}
    with open(f'/tmp/replay/invalid_template.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    try:
        load(replay_dir, template_name)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_extension():
    replay_dir = '/tmp/replay'
    template_name = 'test_template.json'
    test_data = {'cookiecutter': {'key': 'value'}}
    with open(f'/tmp/replay/test_template.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data


# LLM-generated content at query #3
#--------------------------

```
def test_load_with_non_existent_file():
    import pytest
    from pathlib import Path
    from unittest.mock import patch

    with patch('builtins.open', side_effect=FileNotFoundError()):
        with pytest.raises(FileNotFoundError):
            load(Path('nonexistent_dir'), 'template_name')


# LLM-generated content at query #4
#--------------------------

```python
def test_dump_creates_directory_if_not_exists():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    assert Path(test_dir).exists()
    Path(test_dir).rmdir()

def test_dump_writes_correct_json_file():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    file_path = Path(test_dir) / f'{test_template}.json'
    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert content == test_context
    file_path.unlink()
    Path(test_dir).rmdir()

def test_dump_raises_error_without_cookiecutter_key():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'key': 'value'}
    try:
        dump(test_dir, test_template, test_context)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_not_in_context():
    replay_dir = "some_dir"
    template_name = "some_template"
    context = {"some_key": "some_value"}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #6
#--------------------------

```python
def test_load_context_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #7
#--------------------------

```
def test_load_with_valid_replay_file_opens_file_successfully():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    test_data = {'cookiecutter': {'key': 'value'}}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch('builtins.open', mock_file):
        result = load(Path('/test/dir'), 'template')
        assert result == test_data


# LLM-generated content at query #8
#--------------------------

```python
def test_load_with_valid_cookiecutter_key():
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    context_data = {'cookiecutter': {'key': 'value'}}
    with NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp_file:
        json.dump(context_data, temp_file)
        temp_file_path = temp_file.name

    result = load(temp_file_path, 'template_name')
    assert result == context_data
    Path(temp_file_path).unlink()

def test_load_without_cookiecutter_key_raises_error():
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    context_data = {'key': 'value'}
    with NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp_file:
        json.dump(context_data, temp_file)
        temp_file_path = temp_file.name

    try:
        load(temp_file_path, 'template_name')
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    Path(temp_file_path).unlink()


# LLM-generated content at query #9
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_not_in_context():
    context = {"key": "value"}
    try:
        load("dummy_dir", "dummy_template")
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #10
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_not_in_context():
    context = {"some_key": "some_value"}
    try:
        load("dummy_path", "dummy_template")
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #11
#--------------------------

```
def test_load_raises_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    import pytest

    class MockPath:
        def __init__(self, file_content):
            self.file_content = file_content

        def open(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(self.file_content)

        def write(self, *args):
            pass

    def mock_get_file_name(replay_dir, template_name):
        return replay_dir

    context = {"other_key": "value"}
    replay_dir = MockPath(context)
    template_name = "template"

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #12
#--------------------------

```python
def test_load_valid_file():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    mock_file_content = {'cookiecutter': {'key': 'value'}}
    mock_file_path = os.path.join(replay_dir, f'{template_name}.json')
    
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(mock_file_content))):
        result = load(replay_dir, template_name)
        assert result == mock_file_content

def test_load_invalid_file_missing_cookiecutter():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    mock_file_content = {'key': 'value'}
    mock_file_path = os.path.join(replay_dir, f'{template_name}.json')
    
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(mock_file_content))):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_without_json_extension():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    mock_file_content = {'cookiecutter': {'key': 'value'}}
    mock_file_path = os.path.join(replay_dir, f'{template_name}.json')
    
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(mock_file_content))):
        result = load(replay_dir, template_name)
        assert result == mock_file_content


# LLM-generated content at query #13
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = "/some/dir"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #14
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_missing():
    context = {"key": "value"}
    template_name = "test_template"
    replay_dir = "some/directory"
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #15
#--------------------------

```python
def test_load_valid_file():
    replay_dir = 'test_dir'
    template_name = 'valid_template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    file_path = os.path.join(replay_dir, f'{template_name}.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(expected_context, file)
    assert load(replay_dir, template_name) == expected_context

def test_load_file_without_cookiecutter_key():
    replay_dir = 'test_dir'
    template_name = 'invalid_template'
    file_path = os.path.join(replay_dir, f'{template_name}.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({'key': 'value'}, file)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_without_json_suffix():
    replay_dir = 'test_dir'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    file_path = os.path.join(replay_dir, f'{template_name}.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(expected_context, file)
    assert load(replay_dir, template_name) == expected_context

def test_load_file_with_json_suffix():
    replay_dir = 'test_dir'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    file_path = os.path.join(replay_dir, template_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(expected_context, file)
    assert load(replay_dir, template_name) == expected_context


# LLM-generated content at query #16
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_is_missing():
    from pathlib import Path
    import json
    import tempfile
    import os

    # Create a temporary file with JSON data missing the 'cookiecutter' key
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        json.dump({"key": "value"}, temp_file)
        temp_file_path = temp_file.name

    try:
        # Call the load function with the temporary file path
        load(Path(os.path.dirname(temp_file_path)), os.path.basename(temp_file_path))
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    finally:
        os.remove(temp_file_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_load_with_valid_cookiecutter_key():
    from pathlib import Path
    import json
    import tempfile

    template_name = "valid_template"
    context_data = {"cookiecutter": {"key": "value"}}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_file = Path(tmp_dir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as outfile:
            json.dump(context_data, outfile)
        
        result = load(tmp_dir, template_name)
        assert result == context_data

def test_load_without_cookiecutter_key_raises_error():
    from pathlib import Path
    import json
    import tempfile
    import pytest

    template_name = "invalid_template"
    context_data = {"key": "value"}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_file = Path(tmp_dir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as outfile:
            json.dump(context_data, outfile)
        
        with pytest.raises(ValueError):
            load(tmp_dir, template_name)


# LLM-generated content at query #18
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_missing():
    replay_dir = "some/path"
    template_name = "template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #19
#--------------------------

```python
def test_load_contains_cookiecutter_key():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    mock_file_content = '{"cookiecutter": {"key": "value"}}'
    mock_file_path = Path("fake_path") / "template.json"

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        context = load(mock_file_path, "template")
        assert 'cookiecutter' in context


# LLM-generated content at query #20
#--------------------------

```
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = '/path/to/replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #21
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_dir')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key


# LLM-generated content at query #22
#--------------------------

```python
def test_dump_raises_value_error_when_context_does_not_contain_cookiecutter():
    context = {'not_cookiecutter': 'value'}
    template_name = 'test_template'
    replay_dir = '/path/to/replay_dir'
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #23
#--------------------------

```
def test_load_with_invalid_file():
    try:
        load("non_existent_dir", "invalid_template")
        assert False, "Expected ValueError when file does not exist"
    except ValueError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_load_successfully_reads_json_file():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    expected_data = {'cookiecutter': {'key': 'value'}}
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(expected_data, f)
    result = load(replay_dir, template_name)
    assert result == expected_data

def test_load_raises_error_when_cookiecutter_key_missing():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    invalid_data = {'key': 'value'}
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(invalid_data, f)
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_appends_json_suffix_if_missing():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    expected_data = {'cookiecutter': {'key': 'value'}}
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(expected_data, f)
    result = load(replay_dir, template_name)
    assert result == expected_data


# LLM-generated content at query #25
#--------------------------

```python
def test_load_context_contains_cookiecutter_key():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    mock_data = '{"cookiecutter": {"key": "value"}}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        context = load(Path("fake_dir"), "fake_template")
        assert 'cookiecutter' in context


# LLM-generated content at query #26
#--------------------------

```python
def test_load_without_cookiecutter_key_raises_error():
    try:
        load("invalid_dir", "invalid_template")
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #27
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    test_dir = Path('test_dir')
    test_template = 'test_template'
    test_context = {'not_cookiecutter': 'value'}
    
    try:
        dump(test_dir, test_template, test_context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key


# LLM-generated content at query #28
#--------------------------

Here's the unit test for the predicate at line 5:


# LLM-generated content at query #29
#--------------------------

```
def test_load_without_cookiecutter_key_raises_value_error():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch

    test_data = '{"some_key": "some_value"}'
    mock_file = mock_open(read_data=test_data)
    
    with patch('builtins.open', mock_file), \
         patch('pathlib.Path.exists', return_value=True):
        try:
            load(Path('/fake/path'), 'template')
            assert False, "Expected ValueError but no exception was raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #30
#--------------------------

```
def test_load_with_valid_replay_file_opens_file_successfully():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    test_data = {'cookiecutter': {'key': 'value'}}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch('builtins.open', mock_file):
        result = load(Path('/test/dir'), 'template')
        assert result == test_data


# LLM-generated content at query #31
#--------------------------

```
def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch

    test_data = {"not_cookiecutter": "value"}
    mock_file = mock_open(read_data=json.dumps(test_data))
    
    with patch("builtins.open", mock_file):
        try:
            load(Path("test_dir"), "template")
            assert False, "Expected ValueError but no exception was raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #32
#--------------------------

```python
def test_dump_creates_directory_if_not_exists():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    try:
        dump(test_dir, test_template, test_context)
        assert Path(test_dir).exists()
    finally:
        if Path(test_dir).exists():
            Path(test_dir).rmdir()

def test_dump_writes_correct_content_to_file():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    try:
        dump(test_dir, test_template, test_context)
        file_path = Path(test_dir) / f'{test_template}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        assert content == test_context
    finally:
        if Path(test_dir).exists():
            for file in Path(test_dir).iterdir():
                file.unlink()
            Path(test_dir).rmdir()

def test_dump_raises_value_error_without_cookiecutter_key():
    test_dir = 'test_dir'
    test_template = 'test_template'
    test_context = {'key': 'value'}
    try:
        dump(test_dir, test_template, test_context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    else:
        assert False, 'Expected ValueError not raised'


# LLM-generated content at query #33
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_is_missing():
    context = {}
    try:
        load(context, "template_name")
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #34
#--------------------------

```python
def test_load_function_returns_valid_context_with_cookiecutter_key():
    from pathlib import Path
    import json
    import tempfile
    from unittest.mock import patch

    context = {"cookiecutter": {"key": "value"}}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(context, temp_file)
        temp_file_path = Path(temp_file.name)

    result = load(temp_file_path, "template_name")
    assert result == context
    assert 'cookiecutter' in result

def test_load_function_raises_error_when_cookiecutter_key_is_missing():
    from pathlib import Path
    import json
    import tempfile
    import pytest

    context = {"key": "value"}
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(context, temp_file)
        temp_file_path = Path(temp_file.name)

    with pytest.raises(ValueError):
        load(temp_file_path, "template_name")


# LLM-generated content at query #35
#--------------------------

```
def test_load_with_invalid_replay_file():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch

    invalid_json = '{"not_cookiecutter": "value"}'
    mock_file = mock_open(read_data=invalid_json)
    with patch('builtins.open', mock_file):
        try:
            load(Path('/fake/path'), 'template')
            assert False, "Expected ValueError but no exception was raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #36
#--------------------------

```python
def test_dump_creates_directory_if_not_exists():
    replay_dir = 'test_dir'
    template_name = 'template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)
    os.rmdir(replay_dir)

def test_dump_raises_value_error_if_cookiecutter_not_in_context():
    replay_dir = 'test_dir'
    template_name = 'template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    else:
        assert False, "Expected ValueError"

def test_dump_writes_correct_json_file():
    replay_dir = 'test_dir'
    template_name = 'template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #37
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_key_not_in_context():
    context = {"key": "value"}
    try:
        load("some_dir", "some_template")
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #38
#--------------------------

```python
def test_dump_creates_file_with_correct_content():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    
    dump(replay_dir, template_name, context)
    
    expected_file_path = os.path.join(replay_dir, f'{template_name}.json')
    assert os.path.exists(expected_file_path)
    
    with open(expected_file_path, 'r', encoding='utf-8') as infile:
        content = json.load(infile)
        assert content == context

def test_dump_raises_value_error_when_cookiecutter_key_missing():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_creates_directory_if_not_exists():
    replay_dir = 'non_existent_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(replay_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_load_context_contains_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with open("test_dir/test_template.json", "w", encoding="utf-8") as file:
        json.dump(expected_context, file)

    result = load(replay_dir, template_name)
    assert "cookiecutter" in result


# LLM-generated content at query #40
#--------------------------

```python
def test_load_raises_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    
    context = {}
    mock_file_content = json.dumps(context)
    mock_file = mock_open(read_data=mock_file_content)
    
    with patch('builtins.open', mock_file), \
         patch('pathlib.Path.exists', return_value=True):
        try:
            load("fake_dir", "fake_template")
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #41
#--------------------------

```python
def test_load_without_cookiecutter_key_raises_value_error():
    context = {}
    try:
        load("fake_dir", "fake_template")
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    else:
        assert False, "Expected ValueError to be raised"


# LLM-generated content at query #42
#--------------------------

```python
def test_dump_raises_value_error_when_context_does_not_contain_cookiecutter_key():
    replay_dir = 'some/dir'
    template_name = 'template'
    context = {'some_key': 'some_value'}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as error:
        assert str(error) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #43
#--------------------------

```python
def test_load_success():
    replay_dir = "/test/dir"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(expected_context))
    with unittest.mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter_key():
    replay_dir = "/test/dir"
    template_name = "template"
    invalid_context = {"key": "value"}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(invalid_context))
    with unittest.mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
        else:
            assert False, "Expected ValueError not raised"


# LLM-generated content at query #44
#--------------------------

def test_dump_creates_directory_if_not_exists():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    assert Path(test_dir).exists()
    Path(test_dir).rmdir()

def test_dump_raises_value_error_if_no_cookiecutter_key():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'key': 'value'}
    try:
        dump(test_dir, test_template, test_context)
        assert False
    except ValueError:
        assert True
    finally:
        if Path(test_dir).exists():
            Path(test_dir).rmdir()

def test_dump_creates_json_file_with_correct_content():
    test_dir = 'test_replay_dir'
    test_template = 'test_template'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    expected_file = Path(test_dir) / f"{test_template}.json"
    assert expected_file.exists()
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert content == test_context
    expected_file.unlink()
    Path(test_dir).rmdir()

def test_dump_handles_existing_json_suffix_correctly():
    test_dir = 'test_replay_dir'
    test_template = 'test_template.json'
    test_context = {'cookiecutter': {'key': 'value'}}
    dump(test_dir, test_template, test_context)
    expected_file = Path(test_dir) / test_template
    assert expected_file.exists()
    expected_file.unlink()
    Path(test_dir).rmdir()


# LLM-generated content at query #45
#--------------------------

```python
def test_load_returns_context_with_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    mock_file_content = '{"cookiecutter": {"key": "value"}}'
    mock_file_path = "test_dir/test_template.json"
    
    def mock_get_file_name(dir, name):
        return mock_file_path
    
    def mock_open(file_path, encoding):
        if file_path == mock_file_path and encoding == "utf-8":
            return mock_file_content
        raise FileNotFoundError
    
    original_get_file_name = get_file_name
    original_open = open
    
    get_file_name = mock_get_file_name
    open = mock_open
    
    context = load(replay_dir, template_name)
    
    get_file_name = original_get_file_name
    open = original_open
    
    assert 'cookiecutter' in context
    assert context['cookiecutter']['key'] == 'value'


# LLM-generated content at query #46
#--------------------------

```
def test_load_successful_with_json_suffix():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_content = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as f:
        json.dump(expected_content, f)
    result = load(replay_dir, template_name)
    assert result == expected_content


def test_load_successful_without_json_suffix():
    replay_dir = "/tmp"
    template_name = "template"
    expected_content = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as f:
        json.dump(expected_content, f)
    result = load(replay_dir, template_name)
    assert result == expected_content


def test_load_raises_value_error_when_missing_cookiecutter_key():
    replay_dir = "/tmp"
    template_name = "invalid_template.json"
    invalid_content = {"key": "value"}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as f:
        json.dump(invalid_content, f)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_raises_file_not_found_error_when_file_does_not_exist():
    replay_dir = "/nonexistent"
    template_name = "missing_template.json"
    try:
        load(replay_dir, template_name)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pas


# LLM-generated content at query #47
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_is_missing():
    replay_dir = "test_dir"
    template_name = "test_template"
    context = load(replay_dir, template_name)
    assert 'cookiecutter' in context


# LLM-generated content at query #48
#--------------------------

```
def test_load_raises_value_error_when_context_does_not_contain_cookiecutter_key():
    replay_dir = "fake_dir"
    template_name = "fake_template"
    context = {}
    expected_msg = "Context is required to contain a cookiecutter key"
    
    def mock_open(file, encoding):
        return context
    
    def mock_get_file_name(replay_dir, template_name):
        return "fake_file"
    
    original_open = open
    original_get_file_name = get_file_name
    open = mock_open
    get_file_name = mock_get_file_name
    
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == expected_msg
    else:
        assert False, "Expected ValueError but no exception was raised"
    
    open = original_open
    get_file_name = original_get_file_name


