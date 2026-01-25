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

def test_get_file_name_with_nested_path():
    assert get_file_name('/path/to/dir', 'nested/template') == '/path/to/dir/nested/template.json'


# LLM-generated content at query #2
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
    template_name = "invalid_template"

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        outfile.write("invalid json content")

    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "missing_cookiecutter_template"

    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump({"key": "value"}, outfile)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #3
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #4
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
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
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #5
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = os.path.join(replay_dir, f'{template_name}.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context
    os.remove(replay_file)
    os.rmdir(replay_dir)

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError) as exc_info:
        dump(replay_dir, template_name, context)

    assert str(exc_info.value) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'key': 'value'})


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_raises_valueerror_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(Path('/tmp'), 'test-template', {'other_key': 'value'})


# LLM-generated content at query #8
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

def test_dump_raises_value_error_if_context_missing_cookiecutter():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError):
        dump(replay_dir, template_name, context)

def test_dump_raises_os_error_if_replay_dir_cannot_be_created():
    replay_dir = '/invalid/path/that/cannot/be/created'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    with pytest.raises(OSError):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #9
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

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #10
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump('/tmp', 'test-template', {'some_key': 'value'})


# LLM-generated content at query #11
#--------------------------

```python
def test_dump_raises_valueerror_when_context_lacks_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("/tmp"), "test_template", {"other_key": "value"})


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_false():
    assert not (replay_file := get_file_name(replay_dir, template_name))


# LLM-generated content at query #13
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

def test_dump_handles_template_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    with open(replay_file, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context


# LLM-generated content at query #14
#--------------------------

```python
def test_load_with_valid_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_false():
    assert not isinstance(replay_file, str)


# LLM-generated content at query #16
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {}}
    assert 'cookiecutter' in context


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(Path('/tmp'), 'test-template', {'key': 'value'})


# LLM-generated content at query #19
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "path/to/replay"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_json_file_without_json_extension():
    replay_dir = "path/to/replay"
    template_name = "template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, template_name), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "path/to/replay"
    template_name = "template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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

def test_load_without_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template_without_suffix"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter_key():
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


# LLM-generated content at query #22
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

def test_load_with_json_file_ending_with_json():
    replay_dir = "test_dir"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    with patch('builtins.open', mock_open(read_data=json.dumps(expected_context))):
        with patch('os.path.join', return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context


# LLM-generated content at query #23
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

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #24
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path(tempfile.mkdtemp())
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


# LLM-generated content at query #25
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = "test_replay"
    template_name = "test_template"
    replay_file = Path(replay_dir) / f"{template_name}.json"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    result = load(replay_dir, template_name)

    assert result == {"cookiecutter": {}}


# LLM-generated content at query #26
#--------------------------

```python
def test_load_with_valid_file():
    replay_dir = "tests/replay"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"project_name": "test"}}
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "tests/replay"
    template_name = "invalid_template"
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_with_json_suffix():
    replay_dir = "tests/replay"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"project_name": "test"}}
    assert load(replay_dir, template_name) == expected_context


# LLM-generated content at query #27
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

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #28
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #29
#--------------------------

```python
def test_replay_file_opened_successfully():
    replay_dir = "/path/to/replay"
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    assert os.access(replay_file, os.R_OK)


# LLM-generated content at query #30
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "/tmp/test"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking the file system and json.load
    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("os.path.join", return_value=f"{replay_dir}/{template_name}.json"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_json_suffix_in_template_name():
    replay_dir = "/tmp/test"
    template_name = "test_template.json"
    expected_context = {"cookiecutter": {"key": "value"}}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(expected_context)
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("os.path.join", return_value=f"{replay_dir}/{template_name}"):
            result = load(replay_dir, template_name)
            assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "/tmp/test"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    import os
    import json
    from unittest.mock import patch, mock_open

    mock_file_content = json.dumps(invalid_context)
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("os.path.join", return_value=f"{replay_dir}/{template_name}.json"):
            try:
                load(replay_dir, template_name)
                assert False, "Expected ValueError"
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #31
#--------------------------

```python
def test_dump_creates_replay_dir_and_writes_json():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    assert os.path.exists(replay_dir)
    assert os.path.isfile(os.path.join(replay_dir, f'{template_name}.json'))

    with open(os.path.join(replay_dir, f'{template_name}.json'), 'r', encoding="utf-8") as infile:
        written_context = json.load(infile)
    assert written_context == context

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #32
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_without_json_extension():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #33
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    load("valid_replay_dir", "template_name")


# LLM-generated content at query #34
#--------------------------

```python
def test_dump_raises_value_error_when_context_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("/tmp"), "test-template", {"some_key": "value"})


# LLM-generated content at query #35
#--------------------------

```python
def test_dump_ensures_cookiecutter_key_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert 'cookiecutter' in context


# LLM-generated content at query #36
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load('nonexistent_file.json', 'template')


# LLM-generated content at query #37
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    assert replay_file.is_file()


# LLM-generated content at query #38
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    # Setup
    test_dir = Path("test_replay")
    test_file = test_dir / "test_template.json"
    test_dir.mkdir(exist_ok=True)
    test_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")

    # Exercise
    result = load(test_dir, "test_template")

    # Verify
    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #39
#--------------------------

```python
def test_load_with_valid_json_and_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mocking os.path.join to return a known path
    import os
    os.path.join = lambda *args: "test_replay_dir/test_template.json"

    # Mocking open and json.load to return expected context
    import json
    import builtins
    builtins.open = lambda *args, **kwargs: expected_context
    json.load = lambda *args, **kwargs: expected_context

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_without_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    # Mocking os.path.join to return a known path
    import os
    os.path.join = lambda *args: "test_replay_dir/test_template.json"

    # Mocking open and json.load to return invalid context
    import json
    import builtins
    builtins.open = lambda *args, **kwargs: invalid_context
    json.load = lambda *args, **kwargs: invalid_context

    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #40
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


# LLM-generated content at query #41
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Mock the file system and json.load
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

    # Mock the file system and json.load
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


# LLM-generated content at query #42
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("test_replay")
    template_name = "test_template"
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    context = load(replay_dir, template_name)

    assert context == {"cookiecutter": {}}


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

def test_get_file_name_with_nested_path():
    assert get_file_name('/path/to/dir', 'nested/template') == '/path/to/dir/nested/template.json'


# LLM-generated content at query #2
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, "r", encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #3
#--------------------------

```python
def test_load_with_valid_json():
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

def test_load_without_cookiecutter_key():
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


# LLM-generated content at query #4
#--------------------------

```python
def test_cookiecutter_key_present_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #5
#--------------------------

```python
def test_load_raises_value_error_when_cookiecutter_key_missing():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(Path("path/to/replay_dir"), "template_name")


# LLM-generated content at query #6
#--------------------------

```python
def test_dump_creates_replay_file_with_correct_content():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, 'test_template.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context

def test_dump_raises_value_error_if_no_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)

def test_dump_handles_template_name_with_json_extension():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = os.path.join(replay_dir, 'test_template.json')
    with open(replay_file, 'r', encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #7
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


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
    assert os.path.exists(replay_file)

def test_dump_raises_value_error_if_context_missing_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #9
#--------------------------

```python
def test_dump_raises_valueerror_when_context_missing_cookiecutter_key():
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(Path("some_dir"), "template", {"some_key": "value"})


# LLM-generated content at query #10
#--------------------------

```python
def test_dump_creates_replay_file():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


# LLM-generated content at query #11
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("test_replays")
    template_name = "test_template"
    replay_file = Path("test_replays/test_template.json")
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text('{"cookiecutter": {"key": "value"}}', encoding="utf-8")

    result = load(replay_dir, template_name)

    assert result == {"cookiecutter": {"key": "value"}}


# LLM-generated content at query #12
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    file_name = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_file_without_suffix():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    os.makedirs(replay_dir, exist_ok=True)
    file_name = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)

    result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_invalid_json_file():
    replay_dir = "test_replay_dir"
    template_name = "test_template"

    os.makedirs(replay_dir, exist_ok=True)
    file_name = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_name, "w", encoding="utf-8") as outfile:
        outfile.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    invalid_context = {"key": "value"}

    os.makedirs(replay_dir, exist_ok=True)
    file_name = os.path.join(replay_dir, f"{template_name}.json")
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #13
#--------------------------

```python
def test_load_opens_file_with_correct_encoding():
    assert open("dummy_file.json", encoding="utf-8").encoding == "utf-8"


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_cookiecutter_key_exists_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #16
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("some_dir")
    template_name = "some_template"
    replay_file = get_file_name(replay_dir, template_name)
    open(replay_file, encoding="utf-8")


# LLM-generated content at query #17
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    assert replay_file.is_file()


# LLM-generated content at query #18
#--------------------------

```python
def test_cookiecutter_key_exists_in_context():
    context = {'cookiecutter': {'key': 'value'}}
    assert 'cookiecutter' in context


# LLM-generated content at query #19
#--------------------------

```python
def test_dump_creates_replay_file():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, f'{template_name}.json'))

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)

def test_dump_handles_json_suffix():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, template_name))


# LLM-generated content at query #20
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    file_name = os.path.join(replay_dir, f'{template_name}.json')
    context = {'cookiecutter': {'key': 'value'}}
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile)
    result = load(replay_dir, template_name)
    assert result == context

def test_load_with_json_file_without_suffix():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    file_name = os.path.join(replay_dir, f'{template_name}.json')
    context = {'cookiecutter': {'key': 'value'}}
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile)
    result = load(replay_dir, template_name)
    assert result == context

def test_load_with_invalid_json_file():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    file_name = os.path.join(replay_dir, f'{template_name}.json')
    with open(file_name, 'w', encoding="utf-8") as outfile:
        outfile.write('invalid json')
    with pytest.raises(json.JSONDecodeError):
        load(replay_dir, template_name)

def test_load_with_missing_cookiecutter_key():
    replay_dir = 'test_dir'
    template_name = 'test_template'
    file_name = os.path.join(replay_dir, f'{template_name}.json')
    context = {'key': 'value'}
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile)
    with pytest.raises(ValueError) as excinfo:
        load(replay_dir, template_name)
    assert 'Context is required to contain a cookiecutter key' in str(excinfo.value)


# LLM-generated content at query #21
#--------------------------

```python
def test_load_contains_cookiecutter_key():
    context = load("valid_replay_dir", "valid_template")
    assert 'cookiecutter' in context


# LLM-generated content at query #22
#--------------------------

```python
def test_dump_creates_replay_file_with_context():
    replay_dir = Path('test_replay_dir')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()


# LLM-generated content at query #23
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("test_replay")
    template_name = "test_template"
    replay_file = replay_dir / f"{template_name}.json"
    replay_file.parent.mkdir(exist_ok=True)
    replay_file.write_text('{"cookiecutter": {}}', encoding="utf-8")

    result = load(replay_dir, template_name)

    assert result == {"cookiecutter": {}}


# LLM-generated content at query #24
#--------------------------

```python
def test_load_with_valid_json_file():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Assuming the file exists and contains the expected context
    assert load(replay_dir, template_name) == expected_context

def test_load_with_missing_cookiecutter_key():
    replay_dir = "test_dir"
    template_name = "invalid_template"

    # Assuming the file exists but does not contain the 'cookiecutter' key
    try:
        load(replay_dir, template_name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #25
#--------------------------

```python
def test_load_with_valid_json():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"{template_name}.json"), "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile)
    result = load(replay_dir, template_name)
    assert result == expected_context

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


# LLM-generated content at query #26
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


# LLM-generated content at query #27
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


# LLM-generated content at query #28
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

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'key': 'value'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, context)

def test_dump_handles_template_name_with_json_suffix():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_load_opens_file_with_utf8_encoding():
    replay_dir = Path("test_dir")
    template_name = "test_template"
    replay_file = get_file_name(replay_dir, template_name)
    open(replay_file, encoding="utf-8").close()


# LLM-generated content at query #30
#--------------------------

```python
def test_cookiecutter_key_exists_in_context():
    context = {'cookiecutter': {}}
    assert 'cookiecutter' in context


