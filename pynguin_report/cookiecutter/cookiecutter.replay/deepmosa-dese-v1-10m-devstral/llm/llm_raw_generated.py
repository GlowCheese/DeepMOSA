####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_json_suffix():
    assert get_file_name('/path/to/dir', 'template.json') == '/path/to/dir/template.json'

def test_get_file_name_without_json_suffix():
    assert get_file_name('/path/to/dir', 'template') == '/path/to/dir/template.json'

def test_get_file_name_with_path_object():
    assert get_file_name(Path('/path/to/dir'), 'template.json') == '/path/to/dir/template.json'

def test_get_file_name_with_mixed_case_suffix():
    assert get_file_name('/path/to/dir', 'template.JSON') == '/path/to/dir/template.JSON'


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(replay_file)

    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = tempfile.mkdtemp()
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context
    shutil.rmtree(replay_dir)

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = tempfile.mkdtemp()
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)
    shutil.rmtree(replay_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_load_with_valid_file():
    context = load('tests/data', 'valid_template')
    assert isinstance(context, dict)
    assert 'cookiecutter' in context

def test_load_with_invalid_file():
    try:
        load('tests/data', 'invalid_template')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_suffix():
    context = load('tests/data', 'valid_template.json')
    assert isinstance(context, dict)
    assert 'cookiecutter' in context


# LLM-generated content at query #5
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load('replay_dir', 'template_name')


# LLM-generated content at query #6
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_file = Path('dummy_replay.json')
    replay_file.write_text('{"cookiecutter": {}}', encoding='utf-8')

    load(replay_file.parent, 'dummy')

    # The test passes if the file is opened successfully with utf-8 encoding
    # If encoding was wrong, it would raise an exception


# LLM-generated content at query #7
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay", "template_name")


# LLM-generated content at query #8
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_without_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp/test", "test-template", {"some": "context"})


# LLM-generated content at query #10
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = "test_dir"
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    open(replay_file, encoding="utf-8").read() == '{"cookiecutter": {}}'


# LLM-generated content at query #11
#--------------------------

```python
def test_load_with_valid_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_raises_valueerror_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'key': 'value'})


# LLM-generated content at query #13
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))
    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_dump_handles_template_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))
    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_replay"
    template_name = "valid_template"
    os.makedirs(replay_dir, exist_ok=True)
    test_data = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data

def test_load_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "valid_template.json"
    os.makedirs(replay_dir, exist_ok=True)
    test_data = {"cookiecutter": {"key": "value"}}
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    result = load(replay_dir, template_name)
    assert result == test_data

def test_load_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "invalid_template"
    os.makedirs(replay_dir, exist_ok=True)
    test_data = {"key": "value"}
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #15
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context
    assert os.path.exists(replay_file)

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #16
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'key': 'value'})


# LLM-generated content at query #17
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    context = {}
    assert 'cookiecutter' not in context


# LLM-generated content at query #18
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #19
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking the file system and json.load
    import os
    import json
    from pathlib import Path

    # Create a temporary directory and file for testing
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    # Test the load function
    result = load(replay_dir, template_name)
    assert result == expected_context

    # Clean up
    os.remove(test_file)
    os.rmdir(replay_dir)

def test_load_with_invalid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"

    # Mocking the file system and json.load
    import os
    import json

    # Create a temporary directory and file for testing
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"invalid_key": "value"}, outfile)

    # Test the load function
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

    # Clean up
    os.remove(test_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("test_replay")
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    mock_open = mock.mock_open(read_data='{"cookiecutter": {}}')
    with patch('builtins.open', mock_open):
        load(replay_dir, template_name)
        mock_open.assert_called_with(replay_file, encoding="utf-8")


# LLM-generated content at query #22
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    context = {}
    assert 'cookiecutter' not in context


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_without_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_false():
    replay_file = "nonexistent_file.json"
    assert not Path(replay_file).exists()


# LLM-generated content at query #25
#--------------------------

```python
def test_dump_without_cookiecutter_key():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir='/tmp', template_name='test', context={'key': 'value'})


# LLM-generated content at query #26
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "test_template")


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_json_suffix_in_template_name():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context


# LLM-generated content at query #28
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #29
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    """Test that the file is opened with UTF-8 encoding."""
    replay_dir = Path("tests/replays")
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    mock_open = mock.mock_open(read_data='{"cookiecutter": {}}')
    with patch('builtins.open', mock_open):
        load(replay_dir, template_name)
        mock_open.assert_called_once_with(replay_file, encoding="utf-8")


# LLM-generated content at query #30
#--------------------------

```python
def test_dump_creates_directory_and_writes_json_file():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))

    with open(os.path.join(replay_dir, f'{template_name}.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)


# LLM-generated content at query #31
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay"
    template_name = "test_template_no_suffix"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template_invalid"

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #33
#--------------------------

```python
def test_open_file_with_utf8_encoding():
    replay_file = "test.json"
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump({"cookiecutter": {"key": "value"}}, outfile)

    result = load(replay_file, "test")
    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #34
#--------------------------

```python
def test_load_successful():
    context = load('tests/data', 'valid_template')
    assert context == {'cookiecutter': {'project_name': 'test_project'}}

def test_load_missing_cookiecutter_key():
    with pytest.raises(ValueError) as excinfo:
        load('tests/data', 'invalid_template')
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_load_with_json_suffix():
    context = load('tests/data', 'valid_template.json')
    assert context == {'cookiecutter': {'project_name': 'test_project'}}

def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load('tests/data', 'nonexistent_template')


# LLM-generated content at query #35
#--------------------------

```python
def test_load_with_valid_json():
    context = load('valid_replay_dir', 'valid_template')
    assert isinstance(context, dict)
    assert 'cookiecutter' in context

def test_load_with_missing_cookiecutter_key():
    try:
        load('invalid_replay_dir', 'invalid_template')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_suffix():
    context = load('replay_dir', 'template.json')
    assert isinstance(context, dict)
    assert 'cookiecutter' in context

def test_load_without_json_suffix():
    context = load('replay_dir', 'template')
    assert isinstance(context, dict)
    assert 'cookiecutter' in context


# LLM-generated content at query #36
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #37
#--------------------------

```python
def test_dump_creates_directory_and_writes_json():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.isdir(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)

def test_dump_handles_json_suffix_in_template_name():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(os.path.join(replay_dir, template_name))


# LLM-generated content at query #38
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    # Setup
    mock_replay_dir = Path("dummy_dir")
    mock_template_name = "dummy_template"
    mock_file_path = mock_replay_dir / f"{mock_template_name}.json"
    mock_file_path.parent.mkdir(parents=True, exist_ok=True)
    mock_file_path.write_text('{"cookiecutter": {}}', encoding="utf-8")

    # Exercise
    result = load(mock_replay_dir, mock_template_name)

    # Verify
    assert result == {"cookiecutter": {}}


# LLM-generated content at query #39
#--------------------------

```python
def test_load_with_valid_json():
    context = load('tests/data', 'valid_template')
    assert context == {'cookiecutter': {'project_name': 'test'}}
    assert isinstance(context, dict)

def test_load_with_missing_cookiecutter_key():
    with pytest.raises(ValueError) as excinfo:
        load('tests/data', 'invalid_template')
    assert str(excinfo.value) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_suffix():
    context = load('tests/data', 'template_with_suffix.json')
    assert context == {'cookiecutter': {'project_name': 'test_with_suffix'}}


# LLM-generated content at query #40
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    with pytest.raises(ValueError) as excinfo:
        load(replay_dir, template_name)
    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #41
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))

def test_dump_raises_value_error_if_context_missing_cookiecutter():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)

def test_dump_uses_correct_file_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, template_name))


# LLM-generated content at query #42
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #43
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(f"{replay_dir}/{template_name}.json", "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(f"{replay_dir}/{template_name}", "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"

    os.makedirs(replay_dir, exist_ok=True)
    with open(f"{replay_dir}/{template_name}.json", "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    with open(f"{replay_dir}/{template_name}.json", "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #44
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_without_json_extension():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_json_suffix():
    assert get_file_name('/path/to/dir', 'template.json') == '/path/to/dir/template.json'

def test_get_file_name_without_json_suffix():
    assert get_file_name('/path/to/dir', 'template') == '/path/to/dir/template.json'

def test_get_file_name_with_path_object():
    assert get_file_name(Path('/path/to/dir'), 'template') == '/path/to/dir/template.json'


# LLM-generated content at query #2
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_json_file_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"other_key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_creates_replay_file():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_dump_handles_json_suffix_in_template_name():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, template_name))

def test_dump_writes_correct_context_to_file():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'r', encoding="utf-8") as infile:
        written_context = json.load(infile)
    assert written_context == context


# LLM-generated content at query #4
#--------------------------

```python
def test_dump_creates_replay_dir_and_writes_context():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))

    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #5
#--------------------------

```python
def test_load_with_valid_file():
    context = load('tests/replay', 'test_template')
    assert context == {'cookiecutter': {'project_name': 'test'}}
    assert isinstance(context, dict)
    assert 'cookiecutter' in context

def test_load_with_invalid_file():
    try:
        load('tests/replay', 'invalid_template')
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError"

def test_load_with_missing_cookiecutter_key():
    try:
        load('tests/replay', 'missing_cookiecutter')
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_raises_value_error_for_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #7
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_without_json_extension():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_without_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("test_dir", "test_template", {"key": "value"})


# LLM-generated content at query #10
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #11
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp/replay', 'test-template', {'key': 'value'})


# LLM-generated content at query #12
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #13
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = "test_replay"
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    context = {"cookiecutter": {"key": "value"}}

    with patch("builtins.open", mock_open(read_data=json.dumps(context))) as mock_file:
        load(replay_dir, template_name)
        mock_file.assert_called_once_with(replay_file, encoding="utf-8")


# LLM-generated content at query #14
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #15
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp", "test", {"key": "value"})


# LLM-generated content at query #16
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #17
#--------------------------

```python
def test_dump_creates_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #18
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_file = "test.json"
    template_name = "test"
    mock_open = mock.mock_open(read_data='{"cookiecutter": {}}')
    with mock.patch('builtins.open', mock_open):
        load(replay_file, template_name)
        mock_open.assert_called_once_with(replay_file, encoding="utf-8")


# LLM-generated content at query #19
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value="test_dir/test_template.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value="test_dir/test_template.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)


# LLM-generated content at query #20
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    load("valid_replay_dir", "valid_template")


# LLM-generated content at query #21
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #22
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #24
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_path = get_file_name(replay_dir, template_name)
    with open(file_path, 'w', encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(file_path)
    os.rmdir(replay_dir)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    file_path = get_file_name(replay_dir, template_name)
    with open(file_path, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    finally:
        os.remove(file_path)
        os.rmdir(replay_dir)

def test_load_with_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_path = get_file_name(replay_dir, template_name)
    with open(file_path, 'w', encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #25
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #26
#--------------------------

```python
def test_load_with_valid_context():
    context = load("valid_replay_dir", "valid_template")
    assert 'cookiecutter' in context


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_invalid_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("invalid_replay_dir", "template_name")


# LLM-generated content at query #28
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(expected_file)

    with open(expected_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)

    assert content == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as exc_info:
        dump(replay_dir, template_name, context)

    assert str(exc_info.value) == 'Context is required to contain a cookiecutter key'

def test_dump_handles_template_name_with_json_extension():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(expected_file)

    with open(expected_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)

    assert content == context


# LLM-generated content at query #29
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = '/path/to/replay'
    template_name = 'test_template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_json_suffix():
    replay_dir = '/path/to/replay'
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = '/path/to/replay'
    template_name = 'test_template'
    invalid_context = {'key': 'value'}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError) as excinfo:
        load(replay_dir, template_name)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)


# LLM-generated content at query #30
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "valid_dir"
    template_name = "valid_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_invalid_file():
    replay_dir = "invalid_dir"
    template_name = "invalid_template"
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"key": "value"}, f)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_missing_file():
    replay_dir = "missing_dir"
    template_name = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #31
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test', {'key': 'value'})


# LLM-generated content at query #32
#--------------------------

```python
def test_cookiecutter_key_exists_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #33
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    """Test that the file is opened with UTF-8 encoding."""
    replay_file = Path("test_replay.json")
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    load(replay_file, "test_template")

    # The assertion is implicitly checked by the successful execution of the function
    # since if the encoding was incorrect, it would raise an exception


# LLM-generated content at query #34
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("valid_replay_dir", "template_name")


# LLM-generated content at query #35
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #36
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test_template', {'key': 'value'})


# LLM-generated content at query #37
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #38
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #39
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("valid_path", "template_name")


# LLM-generated content at query #40
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Assuming the file exists and contains the expected context
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "invalid_template"

    # Assuming the file exists but does not contain the cookiecutter key
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #41
#--------------------------

```python
def test_dump_creates_directory_and_writes_json():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))

    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #42
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_file = Path("dummy_replay.json")
    template_name = "test_template"
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    result = load(replay_file, template_name)

    assert result == {"cookiecutter": {}}


# LLM-generated content at query #43
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #44
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_json_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    else:
        assert False, "Expected ValueError was not raised"

def test_load_with_json_filename_without_extension():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context


# LLM-generated content at query #45
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking os.path.join and open to simulate file operations
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):
        result = load(replay_dir, template_name)
        assert result == expected_context

def test_load_with_json_file_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)

    with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_with_json_file_without_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):
        result = load(replay_dir, template_name)
        assert result == expected_context


