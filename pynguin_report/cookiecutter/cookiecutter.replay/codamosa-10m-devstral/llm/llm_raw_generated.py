####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name():
    # Test with Path object
    from pathlib import Path
    replay_dir = Path("/tmp/replay")
    template_name = "test_template"
    expected = "/tmp/replay/test_template.json"
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path
    replay_dir = "/tmp/replay"
    template_name = "test_template"
    expected = "/tmp/replay/test_template.json"
    assert get_file_name(replay_dir, template_name) == expected

    # Test with template_name already ending in .json
    replay_dir = "/tmp/replay"
    template_name = "test_template.json"
    expected = "/tmp/replay/test_template.json"
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_get_file_name():
    # Test with Path object and template name without .json suffix
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    template_name = 'test_template'
    expected = '/tmp/replay/test_template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name without .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test_template'
    expected = '/tmp/replay/test_template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with Path object and template name with .json suffix
    replay_dir = Path('/tmp/replay')
    template_name = 'test_template.json'
    expected = '/tmp/replay/test_template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name with .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test_template.json'
    expected = '/tmp/replay/test_template.json'
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_get_file_name():
    # Test with Path object and template name without .json suffix
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    template_name = 'test'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name without .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with Path object and template name with .json suffix
    replay_dir = Path('/tmp/replay')
    template_name = 'test.json'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name with .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test.json'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_get_file_name():
    # Test with Path object and template name without .json suffix
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    template_name = 'test-template'
    expected = '/tmp/replay/test-template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name without .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test-template'
    expected = '/tmp/replay/test-template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with Path object and template name with .json suffix
    replay_dir = Path('/tmp/replay')
    template_name = 'test-template.json'
    expected = '/tmp/replay/test-template.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path and template name with .json suffix
    replay_dir = '/tmp/replay'
    template_name = 'test-template.json'
    expected = '/tmp/replay/test-template.json'
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_get_file_name():
    # Test with Path object
    from pathlib import Path
    replay_dir = Path('/tmp/replay')
    template_name = 'test'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with string path
    replay_dir = '/tmp/replay'
    template_name = 'test'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected

    # Test with template_name already ending in .json
    replay_dir = '/tmp/replay'
    template_name = 'test.json'
    expected = '/tmp/replay/test.json'
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Write test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #7
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #8
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #9
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {
        'project_name': 'test_project',
        'author': 'test_author'
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #10
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #12
#--------------------------

```python
def test_dump(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_with_json_suffix():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    # Execute & Verify
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #13
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with .json suffix
    template_name_with_suffix = "test_template.json"
    replay_file_with_suffix = get_file_name(replay_dir, template_name_with_suffix)

    with open(replay_file_with_suffix, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_with_suffix = load(replay_dir, template_name_with_suffix)
    assert loaded_context_with_suffix == context

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"project_name": "test_project"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")

    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #15
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay_dir"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #16
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #17
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #18
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_dir = tmp_path / "replay"
    test_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(test_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(test_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #19
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #20
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    dump(replay_dir, template_name, context)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_file = tmp_path / f'{template_name}.json'

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #22
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #23
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    expected_file = os.path.join(replay_dir, f'{template_name}.json')

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    test_template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, test_template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, test_template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {"project_name": "test_project"}
    test_template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, test_template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, test_template_name)


# LLM-generated content at query #25
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #26
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #27
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #29
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_without_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #30
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    test_template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, test_template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, test_template_name)

    # Assert
    assert result == test_context

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f'{test_template_name}.json')
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    result_json = load(replay_dir, f'{test_template_name}.json')
    assert result_json == test_context

    # Test missing cookiecutter key
    invalid_context = {'invalid_key': 'value'}
    invalid_replay_file = get_file_name(replay_dir, 'invalid_template')
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'invalid_template')


# LLM-generated content at query #31
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #32
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    expected_file = os.path.join(replay_dir, 'test_template.json')

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Teardown
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #33
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #34
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #36
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #37
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #38
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Test
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / template_name

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #2
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == context

    # Test error case: missing cookiecutter key
    invalid_context = {"key": "value"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #3
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    try:
        # Execute
        dump(replay_dir, template_name, context)

        # Verify
        replay_file = get_file_name(replay_dir, template_name)
        assert os.path.exists(replay_file)

        with open(replay_file, encoding="utf-8") as infile:
            loaded_context = json.load(infile)

        assert loaded_context == context
    finally:
        # Cleanup
        shutil.rmtree(replay_dir)


# LLM-generated content at query #4
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #6
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #8
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #9
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import os
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #11
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(expected_file):
        os.remove(expected_file)
    if os.path.exists(replay_dir):
        os.rmdir(replay_dir)


# LLM-generated content at query #13
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_load(tmp_path):
    # Setup test data
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        },
        'other_data': 'some_value'
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test loading the file
    loaded_context = load(replay_dir, template_name)

    # Verify the loaded data matches the original
    assert loaded_context == test_context

    # Test with .json extension
    template_name_with_ext = 'test_template.json'
    replay_file_with_ext = get_file_name(replay_dir, template_name_with_ext)
    with open(replay_file_with_ext, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    loaded_context_with_ext = load(replay_dir, template_name_with_ext)
    assert loaded_context_with_ext == test_context

    # Test error case - missing cookiecutter key
    invalid_context = {'other_data': 'some_value'}
    invalid_replay_file = get_file_name(replay_dir, 'invalid_template')
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'invalid_template')


# LLM-generated content at query #15
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test context with the required 'cookiecutter' key
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug"
        }
    }

    # Write the test context to a file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == test_context


# LLM-generated content at query #16
#--------------------------

```python
def test_dump():
    import tempfile
    import os
    from pathlib import Path

    # Setup
    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    expected_file = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #17
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    test_template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, test_template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, test_template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {'project_name': 'test_project'}
    test_template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, test_template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test & Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, test_template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    test_template_name = 'test_template.json'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, test_template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, test_template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #18
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #19
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "non_existent_template"
    replay_dir = tmp_path

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #20
#--------------------------

```python
def test_dump(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #21
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    os.makedirs(replay_dir, exist_ok=True)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #22
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_non_existent_file(tmp_path):
    # Setup
    template_name = "non_existent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #23
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #24
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    context = {'cookiecutter': {'project_name': 'test'}}
    dump(tmp_path, 'test', context)
    assert os.path.exists(os.path.join(tmp_path, 'test.json'))

    # Test with .json suffix
    dump(tmp_path, 'test.json', context)
    assert os.path.exists(os.path.join(tmp_path, 'test.json'))

    # Test context validation
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(tmp_path, 'test', {'invalid': 'context'})

    # Test file content
    dump(tmp_path, 'test_content', context)
    with open(os.path.join(tmp_path, 'test_content.json'), encoding="utf-8") as f:
        content = json.load(f)
    assert content == context

    # Test directory creation
    non_existent_dir = tmp_path / 'new_dir'
    dump(non_existent_dir, 'test', context)
    assert os.path.exists(non_existent_dir / 'test.json')


# LLM-generated content at query #25
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    replay_dir = tmp_path / "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    # Verify content
    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json suffix in template name
    template_name_with_suffix = "test_template.json"
    dump(replay_dir, template_name_with_suffix, context)

    expected_file_with_suffix = replay_dir / template_name_with_suffix
    assert expected_file_with_suffix.exists()

    # Test missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test directory creation
    non_existent_dir = tmp_path / "non_existent" / "nested" / "dir"
    dump(non_existent_dir, template_name, context)
    assert (non_existent_dir / f"{template_name}.json").exists()


# LLM-generated content at query #26
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    test_template_name = 'test_template'
    replay_file = tmp_path / f'{test_template_name}.json'

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(tmp_path, test_template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #27
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import shutil
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    try:
        # Execute
        dump(replay_dir, template_name, context)

        # Verify
        replay_file = get_file_name(replay_dir, template_name)
        assert Path(replay_file).exists()

        with open(replay_file, encoding="utf-8") as infile:
            loaded_context = json.load(infile)

        assert loaded_context == context

    finally:
        # Cleanup
        shutil.rmtree(replay_dir)


# LLM-generated content at query #28
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    context = {'cookiecutter': {'project_name': 'test_project'}}
    dump(tmp_path, 'test_template', context)

    # Verify file was created
    expected_file = tmp_path / 'test_template.json'
    assert expected_file.exists()

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json suffix
    dump(tmp_path, 'test_template.json', context)
    expected_file = tmp_path / 'test_template.json'
    assert expected_file.exists()

    # Test ValueError when cookiecutter key is missing
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(tmp_path, 'test_template', {'other_key': 'value'})

    # Test directory creation
    non_existent_dir = tmp_path / 'new_dir'
    dump(non_existent_dir, 'test_template', context)
    assert (non_existent_dir / 'test_template.json').exists()


# LLM-generated content at query #29
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Test
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #30
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #31
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == context

    # Test error handling for missing cookiecutter key
    invalid_context = {"key": "value"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #32
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context), encoding="utf-8")

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #33
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    loaded_context = load(tmp_path, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #34
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
        }
    }

    # Write the context to a file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "project_slug": "test_project",
    }

    # Write the context to a file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #36
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Call the function
    dump(replay_dir, template_name, context)

    # Verify the file was created and contains the correct data
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #37
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / template_name

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #38
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug"
        }
    }

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #39
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


