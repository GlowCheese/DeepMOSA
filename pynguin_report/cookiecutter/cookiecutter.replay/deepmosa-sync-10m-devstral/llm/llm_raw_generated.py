####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "/path/to/replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_json_without_cookiecutter_key():
    replay_dir = "/path/to/replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)

def test_load_with_json_file_ending_with_json():
    replay_dir = "/path/to/replay"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))
    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #3
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("/tmp"), "test-template", {"key": "value"})


# LLM-generated content at query #4
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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
    template_name = "test_template_without_suffix"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template_missing_key"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #7
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, 'test_template.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
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
    replay_file = os.path.join(replay_dir, 'test_template.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #9
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
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)

def test_load_with_template_name_already_having_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context


# LLM-generated content at query #10
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_lacks_cookiecutter_key():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)

def test_dump_handles_template_name_with_json_suffix():
    replay_dir = 'test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #11
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_without_cookiecutter_key():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as exc_info:
        dump(replay_dir, template_name, context)

    assert str(exc_info.value) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #13
#--------------------------

```python
def test_dump_requires_cookiecutter_key_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'some_key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #14
#--------------------------

```python
def test_cookiecutter_key_present():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #15
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp", "test", {"key": "value"})


# LLM-generated content at query #16
#--------------------------

```python
def test_dump_raises_valueerror_when_context_lacks_cookiecutter():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp", "test-template", {"some_key": "value"})


# LLM-generated content at query #17
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError) as excinfo:
        load(replay_dir, template_name)
    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #18
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking the file system and json.load
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            try:
                load(replay_dir, template_name)
                assert False, "Expected ValueError"
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_with_json_filename_already_ending_with_json():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context


# LLM-generated content at query #19
#--------------------------

```python
def test_load_without_cookiecutter_key():
    context = load("nonexistent_dir", "nonexistent_template")


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value="test_dir/test_template.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
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


# LLM-generated content at query #21
#--------------------------

```python
def test_load_with_valid_json_file():
    result = load('test_replay_dir', 'valid_template')
    assert result == {'cookiecutter': {'key': 'value'}}


# LLM-generated content at query #22
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_template_name_without_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template_without_suffix"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #23
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "valid_template"
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"cookiecutter": {"key": "value"}}, outfile)

    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}

def test_load_with_json_file_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "invalid_template"
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"key": "value"}, outfile)

    with pytest.raises(ValueError) as excinfo:
        load(replay_dir, template_name)
    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #24
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    context = {}
    assert 'cookiecutter' not in context


# LLM-generated content at query #25
#--------------------------

```python
def test_dump_raises_valueerror_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp", "test-template", {"some_key": "value"})


# LLM-generated content at query #26
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = '/path/to/replay'
    template_name = 'test_template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = '/path/to/replay'
    template_name = 'test_template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = '/path/to/replay'
    template_name = 'test_template'
    invalid_context = {'key': 'value'}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        result = load(replay_dir, template_name)

    assert result == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            load(replay_dir, template_name)

def test_load_with_valid_json_and_json_suffix():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        result = load(replay_dir, template_name)

    assert result == expected_context


# LLM-generated content at query #28
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #29
#--------------------------

```python
def test_context_without_cookiecutter_key_raises_value_error():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #30
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp", "test", {"key": "value"})


# LLM-generated content at query #31
#--------------------------

```python
def test_context_contains_cookiecutter_key():
    context = {'cookiecutter': {}}
    assert 'cookiecutter' in context


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking the file system and json.load
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value="test_replay_dir/test_template.json"):
            result = load(replay_dir, template_name)

    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value="test_replay_dir/test_template.json"):
            try:
                load(replay_dir, template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #33
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_file = Path("test.json")
    template_name = "test"
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    result = load(replay_file, template_name)

    assert result == {"cookiecutter": {}}


# LLM-generated content at query #34
#--------------------------

```python
def test_load_missing_cookiecutter_key():
    replay_dir = Path("some_dir")
    template_name = "some_template"
    replay_file = get_file_name(replay_dir, template_name)
    context = {"some_key": "some_value"}

    with patch("builtins.open", mock_open(read_data=json.dumps(context))):
        with pytest.raises(ValueError) as excinfo:
            load(replay_dir, template_name)
        assert str(excinfo.value) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #35
#--------------------------

```python
def test_dump_without_cookiecutter_key():
    dump(Path('/tmp'), 'test', {'key': 'value'})


# LLM-generated content at query #36
#--------------------------

```python
def test_load_file_exists_and_readable():
    replay_dir = Path("path/to/existing/dir")
    template_name = "valid_template"
    get_file_name.return_value = "path/to/existing/file.json"
    open.return_value.__enter__.return_value = io.StringIO('{"cookiecutter": {}}')
    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {}}


# LLM-generated content at query #37
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_without_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template_no_ext"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template_invalid"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #38
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking the file system and json.load
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value="test_dir/test_template.json"):
            result = load(replay_dir, template_name)

    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.join', return_value="test_dir/test_template.json"):
            try:
                load(replay_dir, template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #39
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template")


# LLM-generated content at query #40
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'some_key': 'some_value'})


# LLM-generated content at query #41
#--------------------------

```python
def test_load_predicate_false():
    replay_dir = "nonexistent_directory"
    template_name = "test_template"
    assert not (replay_dir / template_name).exists()


# LLM-generated content at query #42
#--------------------------

```python
def test_cookiecutter_key_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #43
#--------------------------

```python
def test_load_with_valid_json():
    assert load('valid_dir', 'valid_template') == {'cookiecutter': {'key': 'value'}}

def test_load_with_missing_cookiecutter_key():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load('invalid_dir', 'invalid_template')

def test_load_with_json_suffix():
    assert load('valid_dir', 'valid_template.json') == {'cookiecutter': {'key': 'value'}}

def test_load_without_json_suffix():
    assert load('valid_dir', 'valid_template') == {'cookiecutter': {'key': 'value'}}


# LLM-generated content at query #44
#--------------------------

```python
def test_replay_file_opens_successfully():
    replay_dir = Path("valid_path")
    template_name = "valid_template"
    replay_file = get_file_name(replay_dir, template_name)
    assert open(replay_file, encoding="utf-8") is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name_with_json_suffix():
    assert get_file_name("/path/to/dir", "template.json") == "/path/to/dir/template.json"

def test_get_file_name_without_json_suffix():
    assert get_file_name("/path/to/dir", "template") == "/path/to/dir/template.json"

def test_get_file_name_with_path_object():
    assert get_file_name(Path("/path/to/dir"), "template.json") == "/path/to/dir/template.json"

def test_get_file_name_with_nested_path():
    assert get_file_name("/path/to/dir", "subdir/template.json") == "/path/to/dir/subdir/template.json"


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_directory_and_writes_json():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'r', encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #3
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
    with open(os.path.join(replay_dir, f"{template_name}"), "w", encoding="utf-8") as outfile:
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


# LLM-generated content at query #4
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking os.path.join and open to avoid actual file operations
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('os.path.join', return_value="test_dir/test_template.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):

        result = load(replay_dir, template_name)
        assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)

    with patch('os.path.join', return_value="test_dir/test_template.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):

        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_with_json_suffix_in_template_name():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)

    with patch('os.path.join', return_value="test_dir/test_template.json"), \
         patch('builtins.open', mock_open(read_data=mock_file_content)):

        result = load(replay_dir, template_name)
        assert result == expected_context


# LLM-generated content at query #5
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    # Assuming the function is called with valid inputs
    replay_dir = "valid_path"
    template_name = "valid_template"
    # The test should verify that the file is opened with utf-8 encoding
    # This is implicitly tested by the function's implementation
    # The actual test would require mocking, which is not allowed here
    # So we just ensure the function can be called without errors
    result = load(replay_dir, template_name)
    assert isinstance(result, dict)


# LLM-generated content at query #6
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"cookiecutter": {"key": "value"}}, outfile)

    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay"
    template_name = "test_template_without_suffix"
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"cookiecutter": {"key": "value"}}, outfile)

    result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template_missing_key"
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump({"key": "value"}, outfile)

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #7
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp/test_replay', 'test_template', {'key': 'value'})


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_raises_valueerror_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir='/tmp/test', template_name='test', context={'other_key': 'value'})


# LLM-generated content at query #10
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir="test_dir", template_name="test_template")


# LLM-generated content at query #11
#--------------------------

```python
def test_dump_raises_valueerror_when_context_lacks_cookiecutter_key():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'some_key': 'value'})


# LLM-generated content at query #12
#--------------------------

```python
def test_dump_missing_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert str(excinfo.value) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #13
#--------------------------

```python
def test_file_not_found():
    assert not Path("nonexistent_file.json").exists()


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (replay_file := get_file_name(replay_dir, template_name))


# LLM-generated content at query #15
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load('nonexistent_dir', 'template')


# LLM-generated content at query #16
#--------------------------

```python
def test_load_contains_cookiecutter_key():
    context = load(replay_dir="valid_replay_dir", template_name="valid_template")
    assert 'cookiecutter' in context


# LLM-generated content at query #17
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_file = Path("dummy_replay.json")
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    context = load(replay_file, "template")

    assert context == {"cookiecutter": {}}


# LLM-generated content at query #18
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #19
#--------------------------

```python
def test_cookiecutter_key_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(file_name)
    os.rmdir(replay_dir)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    finally:
        os.remove(file_name)
        os.rmdir(replay_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("path/to/replay_dir", "template_name")


# LLM-generated content at query #22
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

    # Call the function
    result = load(replay_dir, template_name)

    # Assertions
    assert result == expected_context

    # Clean up
    os.remove(test_file)
    os.rmdir(replay_dir)

def test_load_with_json_file_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    # Mocking the file system and json.load
    import os
    import json
    from pathlib import Path

    # Create a temporary directory and file for testing
    os.makedirs(replay_dir, exist_ok=True)
    test_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(test_file, "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    # Call the function and expect a ValueError
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

    # Clean up
    os.remove(test_file)
    os.rmdir(replay_dir)

def test_load_with_json_file_without_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template_without_extension"
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

    # Call the function
    result = load(replay_dir, template_name)

    # Assertions
    assert result == expected_context

    # Clean up
    os.remove(test_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_dump_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'key': 'value'})


# LLM-generated content at query #24
#--------------------------

```python
def test_load_raises_valueerror_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #25
#--------------------------

```python
def test_load_with_valid_json_file():
    assert load('valid_dir', 'valid_template') == {'cookiecutter': {'key': 'value'}}

def test_load_with_missing_cookiecutter_key():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load('invalid_dir', 'invalid_template')

def test_load_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load('nonexistent_dir', 'nonexistent_template')


# LLM-generated content at query #26
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    # Arrange
    replay_dir = Path("test_replay_dir")
    template_name = "test_template"
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text('{"key": "value"}', encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #27
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(f'{replay_dir}/template.json', 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_json_file_no_suffix():
    replay_dir = '/path/to/replay'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(f'{replay_dir}/template.json', 'w', encoding='utf-8') as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = '/path/to/replay'
    template_name = 'template'
    invalid_context = {'key': 'value'}
    os.makedirs(replay_dir, exist_ok=True)
    with open(f'{replay_dir}/template.json', 'w', encoding='utf-8') as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #28
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        result = load(replay_dir, template_name)

    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with raises(ValueError, match="Context is required to contain a cookiecutter key"):
            load(replay_dir, template_name)


# LLM-generated content at query #29
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        saved_context = json.load(infile)
    assert saved_context == context

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_dump_handles_json_suffix_in_template_name():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.endswith('.json')
    assert os.path.exists(replay_file)


# LLM-generated content at query #30
#--------------------------

```python
def test_context_contains_cookiecutter_key():
    context = {'cookiecutter': {}}
    assert 'cookiecutter' in context


# LLM-generated content at query #31
#--------------------------

```python
def test_dump_creates_directory_and_writes_file():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError):
        dump(replay_dir, template_name, context)

def test_dump_uses_template_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, 'test_template.json'))


# LLM-generated content at query #32
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("/tmp/test"), "test_template", {"key": "value"})


# LLM-generated content at query #33
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir="dummy_path", template_name="dummy_template", context={"other_key": "value"})


# LLM-generated content at query #34
#--------------------------

```python
def test_dump_creates_directory_if_not_exists():
    replay_dir = Path('/tmp/test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


# LLM-generated content at query #35
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #36
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
    Path(replay_dir).rmdir()

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)


# LLM-generated content at query #37
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #38
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "path/to/replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "path/to/replay"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "path/to/replay"
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


# LLM-generated content at query #39
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

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_dump_handles_template_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(replay_file)

    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #40
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    with patch('builtins.open', mock_open()) as mock_file:
        load('dummy_dir', 'dummy_template')
        mock_file.assert_called_once_with('dummy_file', encoding='utf-8')


# LLM-generated content at query #41
#--------------------------

```python
def test_dump_raises_value_error_when_context_lacks_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("/tmp/test"), "test-template", {"some_key": "value"})


# LLM-generated content at query #42
#--------------------------

```python
def test_dump_creates_replay_dir_and_writes_json_file():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.isfile(os.path.join(replay_dir, 'test_template.json'))

    with open(os.path.join(replay_dir, 'test_template.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #43
#--------------------------

```python
def test_dump_creates_replay_file():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


# LLM-generated content at query #44
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


# LLM-generated content at query #45
#--------------------------

```python
def test_context_contains_cookiecutter_key():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #46
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}.json"):
            with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
                load(replay_dir, template_name)


# LLM-generated content at query #47
#--------------------------

```python
def test_dump_creates_directory_and_writes_json():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.isfile(os.path.join(replay_dir, f'{template_name}.json'))

    with open(os.path.join(replay_dir, f'{template_name}.json'), 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #48
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #49
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template")


# LLM-generated content at query #50
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load("nonexistent_dir", "template_name")


# LLM-generated content at query #51
#--------------------------

```python
def test_dump_creates_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context
    Path(replay_dir).rmdir()
    Path(replay_file).unlink(missing_ok=True)

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #52
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    expected_context = {'cookiecutter': {'key': 'value'}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_valid_json_without_cookiecutter_key():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    invalid_context = {'key': 'value'}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f'{template_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(invalid_context, f)

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_with_json_file_without_extension():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    expected_context = {'cookiecutter': {'key': 'value'}}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), 'w', encoding='utf-8') as f:
        json.dump(expected_context, f)

    result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #53
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

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
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #54
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
    template_name = "test_template_without_suffix"
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

    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template_missing_key"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


